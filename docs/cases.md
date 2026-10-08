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
