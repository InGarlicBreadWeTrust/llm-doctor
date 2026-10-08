import argparse
import json
import socket
import statistics
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


def main():
    parser = argparse.ArgumentParser(
        prog="llm-doctor",
        description="diagnose local LLM inference without SSH or root")
    parser.add_argument("check",
        choices=["detect", "list", "ping", "context-scaling",
                 "kv-reuse", "sweep"],
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


if __name__ == "__main__":
    main()

