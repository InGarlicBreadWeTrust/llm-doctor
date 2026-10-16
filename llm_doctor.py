import argparse
import json
import re
import socket
import statistics
import subprocess
import time
import urllib.parse
import urllib.request

_WORD = "the quick brown fox jumps over the lazy dog " * 12


def post_json(url, payload, timeout=60):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, time.time() - t0, resp.read()


def get_json(url, timeout=15):
    req = urllib.request.Request(url,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def norm_endpoint(endpoint):
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    return endpoint


def probe_kind(endpoint):
    base = norm_endpoint(endpoint).rstrip("/")
    if base.endswith("/v1"):
        return "chat_completions", base
    if base.endswith("/api"):
        return "ollama_native", base
    try:
        r = urllib.request.urlopen(base + "/api/tags", timeout=6)
        if r.status == 200 and "models" in r.read().decode("utf-8", "ignore"):
            return "ollama_native", base
    except Exception:
        pass
    try:
        r = urllib.request.urlopen(base + "/v1/models", timeout=6)
        if r.status == 200:
            return "chat_completions", base
    except Exception:
        pass
    return "ollama_native", base


def _strip_suffix(base, suffix):
    b = base.rstrip("/")
    return b[: -len(suffix)] if b.endswith(suffix) else b


def chat_url(base, kind):
    b = _strip_suffix(base, "/v1")
    if kind == "chat_completions":
        return b + "/v1/chat/completions"
    return _strip_suffix(b, "/api") + "/api/chat"


def list_url(base, kind):
    b = _strip_suffix(base, "/v1")
    if kind == "chat_completions":
        return b + "/v1/models"
    return _strip_suffix(b, "/api") + "/api/tags"


def make_prompt(input_tokens):
    if input_tokens <= 0:
        return "hi"
    words = int(input_tokens)
    text = (_WORD * (words + 1))[: words * 12]
    return text + " say the single word: pong"


def check_ping(host, port, n):
    lats = []
    for _ in range(n):
        t0 = time.time()
        try:
            s = socket.create_connection((host, port), timeout=5)
            s.close()
            lats.append((time.time() - t0) * 1000.0)
        except Exception:
            pass
    if len(lats) < max(5, n // 10):
        print("    x  most of %d TCP connects failed - host reachable?" % n)
        return False
    lats.sort()
    n = len(lats)
    print("       p50: %7.1f ms" % statistics.median(lats))
    print("       p90: %7.1f ms" % lats[int(n * 0.90)])
    print("       p99: %7.1f ms" % lats[int(min(n - 1, n * 0.99))])
    print("       max: %7.1f ms" % lats[-1])
    link_ok = lats[int(n * 0.90)] < 15 and lats[-1] < 200
    print()
    if link_ok:
        print("   verdict: link is NOT the bottleneck")
    elif lats[int(min(n - 1, n * 0.99))] > 100:
        print("   verdict: network latency is the bottleneck")
    else:
        print("   verdict: borderline - check both link and context")
    return link_ok


def _stream_ttfb(base, kind, model, prompt, ctx_tokens, timeout=300):
    if kind != "ollama_native":
        payload = {
             "model": model,
             "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 1,
             "num_ctx": ctx_tokens,
             "stream": False,
         }
        try:
            status, elapsed, _ = post_json(
                chat_url(base, kind), payload, timeout=timeout)
            return 0.0, elapsed, status == 200
        except Exception:
            return 0.0, 0.0, False
    payload = {
         "model": model,
         "messages": [{"role": "user", "content": prompt}],
         "num_ctx": ctx_tokens,
         "stream": True,
         "options": {"num_predict": 1},
     }
    req = urllib.request.Request(
        chat_url(base, kind), data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    total = 0.0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if not p or p == "[DONE]":
                    continue
                try:
                    json.loads(p)
                    if first is None:
                        first = time.time() - t0
                except Exception:
                    pass
            total = time.time() - t0
        return (first if first is not None else total), total, True
    except Exception:
        return 0.0, 0.0, False


def check_context_scaling(base, kind, model, base_ctx=8192, mults=(1, 2, 4, 8)):
    results = []
    for mult in mults:
        ctx = base_ctx * mult
        head = "   ctx=%8d   in~%8d tok      (%2dx)" % (ctx, ctx // 2, mult)
        print(head)
        ttfb, total, ok = _stream_ttfb(
            base, kind, model, make_prompt(ctx // 2), ctx)
        results.append((ctx, ttfb, ok))
        print("        TTFB = %7.2fs" % ttfb if ok else "        FAIL")
    good = [(c, t) for c, t, ok in results if ok and t > 0.01]
    if len(good) < 3:
        print("\n   not enough successful requests to fit a line")
        return
    xs = [c / base_ctx for c, _ in good]
    ys = [t for _, t in good]
    n, sx, sy = len(xs), sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    slope = (n * sxy - sx * sy) / max(n * sxx - sx * sx, 1e-9)
    yb = (sy - slope * sx) / n
    ys_hat = [slope * x + yb for x in xs]
    ss_res = sum((y - yhat) ** 2 for y, yhat in zip(ys, ys_hat))
    ss_tot = sum((y - sy / n) ** 2 for y in ys) or 1e-9
    r2 = 1.0 - ss_res / ss_tot
    med = statistics.median(ys)
    print()
    print("   fit: TTFB = %+.3f s x (ctx/%d) + %+.1f s    R^2 = %.2f"
           % (slope, base_ctx, yb, r2))
    if slope > 1.5:
        print("   verdict: TTFB climbs steeply with input length - the engine")
        print("            reuses nothing and rebuilds its KV cache every turn.")
        print("            That IS your 'network crawl.' It is a context-size")
        print("            problem, NOT the network.")
        print("            Fix: pin num_ctx to a window the model holds resident,")
        print("            rebuild it on the server, then re-run 'kv-reuse'.")
    elif med < 1.5:
        print("   verdict: HEALTHY - first token is fast and does not scale with")
        print("            context. Prefill is not your bottleneck.")
    elif slope >= 0.5:
        print("   verdict: TTFB scales partially - check both link and context")
    else:
        print("   verdict: slow but flat - looks like a cold start or a")
        print("            connection-setup cost, not context growth.")


def check_kv_reuse(base, kind, model, n_turns=4, ctx=65536):
    prompt = make_prompt(ctx // 2)
    print("   prompt ~%d tokens   x%d requests   ctx=%d"
          % (ctx // 2, n_turns, ctx))
    results = []
    for i in range(n_turns):
        ttfb, total, ok = _stream_ttfb(base, kind, model, prompt, ctx, 300)
        results.append((i + 1, ttfb, ok))
        if ok:
            print("       turn %2d:  TTFB = %7.2fs     OK" % (i + 1, ttfb))
        else:
            print("       turn %2d:     timed out     FAIL" % (i + 1,))
    ok_ttfb = [t for _, t, ok in results if ok and t > 0]
    if len(ok_ttfb) < 2:
        print("\n   not enough successful turns to compare")
        return
    first_t = ok_ttfb[0]
    rest_avg = statistics.mean(ok_ttfb[1:])
    ratio = first_t / max(rest_avg, 1e-3)
    print()
    print("   first cold  TTFB: %7.2fs" % first_t)
    print("   reused avg  TTFB: %7.2fs     (%.0fx slower when cold)"
          % (rest_avg, ratio))
    if rest_avg < 1.5:
        print("   verdict: KV cache IS being reused - one warm-up turn, then")
        print("            sub-second. Healthy.")
    else:
        print("   verdict: KV cache is NOT being reused - the engine rebuilds")
        print("             %d tokens of KV every single turn." % ctx)
        print("            Fix: pin num_ctx to the model's real context window")
        print("            (e.g. 65536, not 262144) and rebuild on the server.")


def pick_model(base, kind, fallback="qwen3.8:27b-mlx"):
    try:
        d = get_json(list_url(base, kind), timeout=10)
        if kind == "ollama_native" and d.get("models"):
            return d["models"][0]["name"]
        if kind == "chat_completions" and d.get("data"):
            return d["data"][0]["id"]
    except Exception:
        pass
    return fallback


def show_models(base, kind):
    try:
        d = get_json(list_url(base, kind), timeout=10)
        if kind == "ollama_native":
            for m in d.get("models", []):
                size = m.get("size", 0) / 1e9
                ctx = m.get("details", {}).get("context_length", "?")
                print("        %-42s    %5.1f GB  ctx=%s"
                      % (m.get("name", "?"), size, ctx))
        else:
            for m in d.get("data", []):
                print("        %s" % m.get("id", "?"))
    except Exception as e:
        print("      failed to list: %s" % e)


def ps_url(base, kind):
    """Ollama-native /api/ps (list of resident models) - the memory truth.

    Ollama-only. For an OpenAI-compatible endpoint (vLLM/LM Studio served over
     /v1) this may 404; the check reports that and points you at --ssh instead.
    """
    b = _strip_suffix(base, "/v1")
    if kind == "chat_completions":
        return b + "/api/ps"
    return _strip_suffix(base, "/api") + "/api/ps"


# ---------------------------------------------------------------------------
# IDLE-MEMORY DIAGNOSTIC     (subcommand: "mem")
#
# HTTP-FIRST by design. Ollama's /api/ps already tells you exactly which model
# is resident and how big, with NO SSH -- that answers "what is being used when
# idle" in the overwhelming majority of cases, and it needs no credentials. The
# --ssh path is opt-in and only pulled in for OS-level memory (vm_stat /
# free / memory_pressure, the ollama process RSS) and for a server-side fix
# (changing OLLAMA_KEEP_ALIVE). Honours the least-privilege stance:
#     * diagnosis            -> HTTP only, no root, no SSH
#     * server-side change   -> SSH, key-pinned, non-root, no sudo
# ---------------------------------------------------------------------------

# Tight, non-interactive ssh flags. No passwords / interactive prompts: if the
# key is not authorised the command fails fast (rc!=0) instead of hanging. No
# su/sudo/full shell: we hand bash ONE script on stdin (`bash -s`) and parse
# only the tagged KEY=VALUE lines it emits.
_SSH_SAFE_FLAGS = [
    "-o", "BatchMode=yes",        # never block on a password prompt
    "-o", "ConnectTimeout=8",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "PasswordAuthentication=no",
    "-o", "KbdInteractiveAuthentication=no",
    "-o", "NumberOfPasswordPrompts=0",
]

# Runs on the remote host, passed on STDIN to `ssh ... 'bash -s'` -- no shell
# quoting / base64 to fight. Emits tagged lines; the Python side parses only the
# keys it knows. Cross-platform (macOS + Linux). Never touches sudo.
REMOTE_MEM = u"""set +e
U=$(uname -s)
echo HOST=$(hostname)
echo UNAME=$U
case "$U" in
  Darwin)
    echo HW_MODEL=$(sysctl -n hw.model 2>/dev/null)
    echo MEM_TOTAL=$(sysctl -n hw.memsize 2>/dev/null)
    echo MEM_PRESSURE=$(memory_pressure -Q 2>/dev/null | awk '/overall:/{print; exit}')
    echo TOPMEM=$(top -l1 -s 0.3 2>/dev/null | awk '/PhysMem/{print; exit}')
    ;;
  Linux)
    echo HW_MODEL=$(cat /sys/class/dmi/id/product_name 2>/dev/null || hostname)
    echo MEM_TOTAL=$(awk '/MemTotal/{print $2*1024}' /proc/meminfo)
    echo MEM_AVAIL=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
    ;;
esac
echo OLLAMA_RSS=$(ps -axo rss=,etime=,comm= 2>/dev/null | awk 'tolower($3) ~ /ollama/ {printf "%d %s %s",$1,$2,$3; exit}')
for e in OLLAMA_KEEP_ALIVE OLLAMA_MAX_LOADED_MODELS OLLAMA_HOST OLLAMA_NUM_CTX OLLAMA_NGPU OLLAMA_FLASH_ATTENTION; do
    v=$(printenv $e 2>/dev/null); [ -n "$v" ] && echo OLLAMA_ENV_$e=$v
done
if [ -f "$HOME/.ollama.env" ]; then
    i=0
    while IFS= read -r ln; do
        [ -z "$ln" ] && continue
        case "$ln" in
            *=*assword*|*=*secret*=*|*=*token*=*|*=*api_key*=*|*=*API_KEY*=*|*=*SECRET*=*|*=*TOKEN*=*)
                echo "OLLAMA_ENVFILE=$i: <redacted>";;
            *)
                echo "OLLAMA_ENVFILE=$i: $ln";;
        esac
        i=$((i+1))
    done < "$HOME/.ollama.env"
fi
"""


def _expires_note(expires_at):
    """Human note for an Ollama expires_at timestamp.

    Ollama sets a far-future sentinel (year thousands out) when keep_alive is -1
    or a huge value -- the "pinned resident, never unloads" state, which is the
    idle-memory footprint we care about. Tolerant of the 9-digit-microsecond /
    signed-offset format Ollama emits (datetime parsing is not 3.9-safe for it),
    so we only inspect the leading year.
    """
    if not expires_at:
        return "n/a"
    m = re.match(r"^(\d{4})", str(expires_at))
    if m and int(m.group(1)) >= 2123:
        return "NEVER (keep_alive pinned: -1 or huge value)"
    return "expires " + str(expires_at)


def _remote_probe(ssh_target, key, port):
    """Run the memory probe on the remote host over ssh. Returns the parsed
    tagged lines as a dict, or a dict with("_error") describing why it could not
    run (key not authorised, host down, ssh client missing). Never raises."""
    cmd = ["ssh"] + list(_SSH_SAFE_FLAGS)
    if port and int(port) != 22:
        cmd += ["-p", str(port)]
    if key:
        cmd += ["-i", key]
    # target may be "user@host" or "host"; ssh handles both.
    cmd += [ssh_target, "bash", "-s"]
    try:
        p = subprocess.run(cmd, input=REMOTE_MEM, capture_output=True,
                           text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return {"_error": "ssh timed out after 30s (host unreachable or hung)"}
    except FileNotFoundError:
        return {"_error": "no `ssh` client on this machine"}
    except Exception as e:  # defensive
        return {"_error": "ssh error: %s" % e}

    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        return {"_error": err or ("ssh exited rc=%d" % p.returncode)}

    d = {}
    env_files = []
    keep = {"HOST", "UNAME", "HW_MODEL", "MEM_TOTAL", "MEM_USED",
            "MEM_PRESSURE", "MEM_AVAIL", "TOPMEM", "OLLAMA_RSS"}
    for line in p.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.startswith("OLLAMA_ENVFILE"):
            env_files.append(v.strip())
        elif k.startswith("OLLAMA_ENV_") or k in keep:
            d[k] = v.strip()
    if env_files:
        d["OLLAMA_ENVFILE"] = env_files
    return d


def _print_os_memory(info, total_vram):
    if "MEM_PRESSURE" in info:
        raw = info["MEM_PRESSURE"]
        m = re.search(r"(\d+)\s*%", raw)
        pc = int(m.group(1)) if m else None
        note = "   <-- HIGH (throttling / swapping likely)" if (pc is not None and pc >= 75) else ""
        print("    macOS memory pressure: %s%s" % (raw, note))
    if "MEM_AVAIL" in info:
        try:
            avail = float(info["MEM_AVAIL"])
            tot = float(info.get("MEM_TOTAL") or total_vram or avail)
            used = max(tot - avail, 0.0)
            pct = (used / tot * 100.0) if tot else 0.0
            print("    Linux: %6.1f GB used / %.1f GB total (%.1f%%)"
                  % (used / 1e9, tot / 1e9, pct))
        except (ValueError, TypeError):
            pass
    if "TOPMEM" in info:
        print("    top: %s" % info["TOPMEM"])
    if "OLLAMA_RSS" in info:
        print("    ollama proc: %s" % info["OLLAMA_RSS"])
    if "HW_MODEL" in info:
        print("    host: %-30s %s" % (info.get("HOST", "?"), info.get("HW_MODEL", "?")))
    ka = info.get("OLLAMA_ENV_OLLAMA_KEEP_ALIVE")
    for k in ("OLLAMA_ENV_OLLAMA_MAX_LOADED_MODELS", "OLLAMA_ENV_OLLAMA_HOST",
              "OLLAMA_ENV_OLLAMA_NUM_CTX"):
        if k in info:
            print("    env: %-38s %s" % (k, info[k]))
    print("    env: OLLAMA_KEEP_ALIVE = %s"
          % (ka if ka else "(unset -> Ollama default: 5m per request)"))
    for ln in (info.get("OLLAMA_ENVFILE") or []):
        print("      ~/.ollama.env: %s" % ln)


def check_mem(base, kind, endpoint, model=None,
        ssh_target=None, ssh_key=None, ssh_port=22,
        fix=False, apply=False, keep_alive="15m"):
    print()
    print("--- RESIDENT MODELS (Ollama /api/ps - HTTP, no SSH) ---")

    resident = []
    catalog = []
    ps_err = None
    try:
        ps = get_json(ps_url(base, kind), timeout=10)
        resident = ps.get("models", []) if isinstance(ps, dict) else []
    except Exception as e:
        ps_err = e
    try:
        tags = get_json(list_url(base, kind), timeout=10)
        if kind == "chat_completions":
            data = tags.get("data", []) or []
            catalog = [x.get("id") or x.get("name") or "?" for x in data]
        else:
            catalog = [x.get("name") or "?" for x in (tags.get("models", []) or [])]
    except Exception:
        pass

    if ps_err:
        print("      /api/ps unavailable: %s" % ps_err)
        if kind == "chat_completions":
            print("    (OpenAI-compatible endpoint; /api/ps is Ollama-only)")
            print("    pass --ssh user@host to read OS-level memory instead")

    total_vram = 0
    pinned = []
    for m in resident:
        name = m.get("model") or m.get("name") or "?"
        vram = m.get("size_vram") or m.get("size") or 0
        ctx = m.get("context_length", "?")
        det = m.get("details") or {}
        q = det.get("quantization_level", "?")
        fmt = det.get("format", "?")
        note = _expires_note(m.get("expires_at"))
        total_vram += vram
        if note.startswith("NEVER"):
            pinned.append(name)
        print("        %-40s %7.1f GB vram  ctx=%-7s   %-11s %-8s %s"
              % (name, vram / 1e9, ctx, fmt, q, note))
    if catalog:
        print("    on disk (%d): %s" % (len(catalog), ", ".join(catalog)))
    print("    resident TOTAL: %.1f GB held in unified/VRAM" % (total_vram / 1e9))

    # OS-level (opt-in, SSH)
    info = None
    if ssh_target:
        print()
        print("--- OS MEMORY (ssh to %s, key-pinned, no password) ---" % ssh_target)
        info = _remote_probe(ssh_target, ssh_key, ssh_port)
        if "_error" in info:
            err = info["_error"].lower()
            print("    could not reach remote OS metrics: %s" % info["_error"])
            if "permission denied" in err or "publickey" in err:
                key = ssh_key or "~/.ssh/id_ed25519_llmdoctor"
                if "@" in ssh_target:
                    user, _, host = ssh_target.partition("@")
                else:
                    user, host = "", ssh_target
                grant = user + ("@" + host if host else "")
                print("    -> the llm-doctor key is not authorised on this host yet.")
                print("       one-time grant, no root (key-pinned, non-root):")
                print("        ssh-copy-id -i %s.pub %s" % (key, grant or host))
                print("     ...or append %s.pub to that host's ~/.ssh/authorized_keys." % key)
        else:
            _print_os_memory(info, total_vram)

    # ---- verdict ----
    print()
    print("--- VERDICT ---")
    if pinned:
        print("   FOUND: %d model(s) PINNED resident forever (keep_alive=-1/huge): %s"
              % (len(pinned), ", ".join(pinned)))
        print("   That %.1f GB stays resident even while the agent is idle - this is"
              % (total_vram / 1e9))
        print("    your 'used when idle'. macOS marks the file-backed weights as")
        print("    'reclaimable', so Activity Monitor under-reports it, but Ollama")
        print("    keeps them mapped. max_loaded=1 means only ONE big model is")
        print("    resident at a time (the other is on disk, not the cost).")
    elif total_vram == 0 and resident:
        print("   No model is currently resident - idle memory is clean")
        print("    (Ollama default keep_alive=5m already unloads them).")
    elif total_vram == 0:
        print("   Nothing resident (or /api/ps was unreachable) - nothing to reclaim.")
    else:
        print("    %.1f GB resident with bounded keep_alive - unloads after idle. Fine."
              % (total_vram / 1e9))

    # security nudge
    if info and info.get("OLLAMA_ENV_OLLAMA_HOST"):
        h = info["OLLAMA_ENV_OLLAMA_HOST"]
        if h not in ("127.0.0.1", "localhost"):
            print()
            print("   SECURITY: OLLAMA_HOST=%s exposes the UNAUTHENTICATED Ollama API" % h)
            print("    to the whole LAN (any host on 192.168.12.x can run your models).")
            print("    Set OLLAMA_HOST=127.0.0.1 unless you mean it.")

    # ---- remediation ----
    if fix or pinned:
        print()
        print("--- RECOMMENDED FIXES ---")
        if pinned:
            print("    1) Auto-unload idle models after inactivity instead of forever")
            print("        (edit ~/.ollama.env, then restart Ollama; cost = one warmup):")
            print("        echo 'OLLAMA_KEEP_ALIVE=%s' >> ~/.ollama.env" % keep_alive)
            print("         # macOS:  launchctl kickstart -k gui/$(id -u) org.ollama.app")
            print("         # linux:  systemctl --user restart ollama   (if a service unit)")
        if total_vram > 0:
            print("    2) Reclaim the %d resident model(s) RIGHT NOW over HTTP (no SSH):"
                  % len(resident))
            b = _strip_suffix(_strip_suffix(base, "/api"), "/v1")
            unload = b + "/api/generate"
            for m in resident:
                nm = m.get("model") or m.get("name") or "?"
                print("        curl -s %s -d '{\"model\":%s,\"keep_alive\":0}' >/dev/null"
                      % (unload, json.dumps(nm)))
            print("       WARN: unloading the model your running agent uses forces the same")
            print("       slow warmup on the next request. Run this only when idle.")
            if apply:
                print()
                print("    --apply: NOT auto-unloading - you are likely the client that needs")
                print("    this model. Run command 2 by hand when it is safe.")
    elif not fix:
        print()
        print("   Re-run with --fix to print the exact remediation commands.")
    print()



def main():
    parser = argparse.ArgumentParser(
        prog="llm-doctor",
        description="diagnose local LLM inference without SSH or root")
    parser.add_argument("check",
        choices=["detect", "list", "ping", "context-scaling",
                 "kv-reuse", "sweep", "mem"],
        help="which check to run")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434",
                       help="Ollama /api or OpenAI /v1 URL; "
                            "bare host:port auto-detects")
    parser.add_argument("--model", default=None,
                       help="model name (auto-discovered if omitted)")
    parser.add_argument("--host", default=None,
                       help="host for ping (parsed from --endpoint if omitted)")
    parser.add_argument("--port", type=int, default=11434,
                       help="TCP port for ping (default 11434)")
    parser.add_argument("--ping-n", type=int, default=50,
                       help="TCP round-trips to measure (default 50)")
    parser.add_argument("--base-ctx", type=int, default=8192,
                       help="base context for context-scaling (default 8192)")
    # "mem" subcommand options -- all opt-in; diagnosis is HTTP-only by default.
    parser.add_argument("--ssh", default=None, metavar="USER@HOST",
                       help="mem: read OS-level memory on the remote host over a "
                             "key-pinned, non-root ssh (no passwords, no sudo)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519_llmdoctor",
                       help="mem: ssh key file (default ~/.ssh/id_ed25519_llmdoctor)")
    parser.add_argument("--ssh-port", type=int, default=22,
                       help="mem: ssh port (default 22)")
    parser.add_argument("--fix", action="store_true",
                       help="mem: print the remediation commands")
    parser.add_argument("--apply", action="store_true",
                       help="mem: run the HTTP-only unload right now "
                             "(dangerous while your agent is live: forces a 60s warmup)")
    parser.add_argument("--keep-alive", default="15m",
                       help="mem: value to write for OLLAMA_KEEP_ALIVE (default 15m)")
    args = parser.parse_args()

    kind, base = probe_kind(args.endpoint)
    print("llm-doctor    endpoint=%s    kind=%s\n" % (args.endpoint, kind))

    if args.check == "detect":
        print("auto-detected kind:      %s" % kind)
        print("chat endpoint:         %s" % chat_url(base, kind))
        print("model-list endpoint:   %s" % list_url(base, kind))
        return
    if args.check == "list":
        print("models on %s:" % args.endpoint)
        show_models(base, kind)
        return

    model = args.model or pick_model(base, kind)
    print("using model: %s\n" % model)
    parsed = urllib.parse.urlsplit(norm_endpoint(args.endpoint))
    host = args.host or parsed.hostname or "127.0.0.1"
    port = parsed.port or args.port

    if args.check == "ping":
        check_ping(host, port, args.ping_n)
    elif args.check == "context-scaling":
        check_context_scaling(base, kind, model, base_ctx=args.base_ctx)
    elif args.check == "kv-reuse":
        check_kv_reuse(base, kind, model, ctx=args.base_ctx * 8)
    elif args.check == "sweep":
        print("--- PING (link) ---")
        check_ping(host, port, args.ping_n)
        print("\n--- CONTEXT-SCALING (KV) ---")
        check_context_scaling(base, kind, model, base_ctx=args.base_ctx)
        print("\n--- KV-REUSE ---")
        check_kv_reuse(base, kind, model, ctx=args.base_ctx * 8)
        print("\n--- SUMMARY ---")
        print("  If PING says 'link OK' but CONTEXT-SCALING shows TTFB")
        print("  climbing with input length, it is your context window, not")
        print("  the network. Pin num_ctx to the model's real context and")
        print("  confirm reuse with 'kv-reuse'.")
    elif args.check == "mem":
        # Idle-memory diagnostic. HTTP-only by default; the --ssh and --fix
        # paths are opt-in so a remote key never has to exist to diagnose.
        check_mem(base, kind, args.endpoint, model=model,
                   ssh_target=args.ssh, ssh_key=args.ssh_key,
                   ssh_port=args.ssh_port, fix=args.fix,
                   apply=args.apply, keep_alive=args.keep_alive)


if __name__ == "__main__":
    main()

