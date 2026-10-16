# Real-World Cases

## 2026-10-08 Apple M5 Pro, Ollama 0.34.4, qwen3.8:27b-mlx

### Symptoms

Agent using Ollama on an Apple M5 Pro via WiFi reported:
- "60 second wait for first stream chunk" on every turn
- "Network traffic crawl" during generation
- TTFB ~57s cold, ~15s even after warmup

First instinct: "my WiFi is bad." Second instinct: "the keep-alive
setting is wrong." Neither was the issue.

### Diagnosis

1. **Ping** (`ping --ping-n 600`)
     - p50 = 2.6ms, p90 = 31ms, p99 = 145ms
     - Verdict: "link is NOT the bottleneck."
     - The 30ms p90 spikes are WiFi power-save cadence (iwl DTIM100),
       not TCP-level latency. TCP connects were flat at p50=2.6ms,
       p90=4.5ms.

2. **context-scaling**
     - 8k window: TTFB 4.5s
     - 16k window: TTFB 9.4s
     - 32k window: TTFB 7.7s
     - 262144 window (client's num_ctx): TTFB ~57s cold every turn
     - R^2 = 0.97 - TTFB tracks input length.
     - Verdict: "a KV/ctx problem, not the network."

3. **kv-reuse**
     - At num_ctx=262144: turn 1 = 57s, turn 2 = 15s.
       The engine rebuilds an oversized KV that cannot fit in the
       48GB unified memory with the model resident.
     - At num_ctx=65536: turn 1 = 53s (one-time warmup), turn 2-4 = 0.24s.
       Verdict: "healthy."

### Root cause

The agent's config pinned num_ctx=262144, 8x the model's 32768 real
context window. Ollama's KV cache for 262144 tokens did not fit in the
memory left after loading the model, so it was evicted and rebuilt on
every turn.

### Fix

1. Pin num_ctx to the model's real context in the client config.
2. Rebuild the model on the server with num_ctx 65536 in the Modelfile.
3. Drop the client's endpoint-probe cache so it does not re-detect at
   the old context size.

### Result

- TTFB: 0.30s (was 57s) - 190x improvement
- ~19 tok/s sustained generation
- Both algebra problems answered correctly in ~7s total

### Lesson

The "network crawl" was never the network. It was the engine recomputing
an oversized KV cache on every turn. The tell is a TTFB that tracks
num_ctx, not one that tracks Wi-Fi quality.

## 2026-10-16 Apple M5 Pro, Ollama 0.34.4, qwen3.8:27b-mlx -- idle memory

### Symptoms

User reported ~10% of the 48GB unified memory stayed "used" even when the
agent was not running. Activity Monitor looked low, but the machine felt
thrashed and cold starts were slow.

### Diagnosis

`mem --endpoint 192.168.12.10:11434` (HTTP-only, no SSH, no password):

1. **/api/ps** -- one model resident: qwen3.8:27b-mlx, 20.8G VRAM,
   expires_at = 2319-01-25 -> keep_alive = "never unload".
   This is the smoking gun: with `max_loaded=1` a single forever-pinned
   model keeps ~21-32GB mapped and macOS shows most of it as
   "reclaimable", so Activity Monitor under-reports it.

2. No second model resident (the 29GB uncensored one sits on disk).

### Root cause

Ollama was launched once with `keep_alive` set to never, so the resident
model was never evicted. macOS unified memory classifies the file-backed
model as reclaimable, so the Activity Monitor number (10%) is lower than
the true resident footprint, but the pinned model still costs RAM and
forces warmups.

### Fix

1. Bound the idle residency: add `OLLAMA_KEEP_ALIVE=15m` to
   `~/.ollama.env` and restart the Ollama service (one-time warmup cost).
2. To reclaim immediately over HTTP (no SSH):
   `curl -s http://192.168.12.10:11434/api/generate -d '{"model":"qwen3.8:27b-mlx","keep_alive":0}'`
   -- safe only when the agent is idle.
3. `--ssh --fix` is opt-in for OS-level view (vm_stat / memory_pressure)
   only; diagnosis itself needs no SSH and no password.

### Result

`mem` prints the resident model, the keep_alive sentinel, the real
VRAM footprint, and a copy-paste remediation -- all over plain HTTP.

### Lesson

"Used when idle" is usually not a leak. It is one model pinned forever by
keep_alive that macOS hides as reclaimable. Ollama's `/api/ps` exposes it
with zero credentials; SSH is only needed for the OS-level view.
