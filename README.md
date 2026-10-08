<div align="center">

# 🩺 llm-doctor

**Is your local LLM actually slow — or is your agent just asking for too much context?**

A backend-agnostic diagnostic tool for anyone whose local model feels sluggish,
stalls on "waiting for first token," or crawls over the network. It runs the
experiments a tired engineer would eventually try by hand — and tells you the
verdict, in plain words.

</div>

---

## The problem it solves

Local LLM inference has three completely different failure modes that all look
the same to a user:

| Symptom you see | What it usually *isn't* | What it often *is* |
|---|---|---|
| "Network is crawling between agent and server" | A bad WiFi link / router | KV-cache blowup from an oversized context window |
| "57s waiting for the first token, every turn" | A cold-start reload | The engine rebuilding a KV cache that can't fit, instead of reusing it |
| "The model's slow" | A weak / mis-quantized model | Your client asking for more context than the engine will ever keep resident |

In most of these cases **the network is fine and the model is fine** — the
*conversation context* is the variable that's broken, and it's the one nobody
looks at first because it lives in their agent's config, not in a log line.

`llm-doctor` checks the thing people skip: it measures **first-token latency as a
function of the context you ask the model to keep alive**, then tells you whether
that scaling is linear-with-context (a KV/prefill problem, *not* a link problem).

---

## Install & run

```bash
# clone, then run any check directly (no deps, Python 3.9+)
git clone https://github.com/youruser/llm-doctor.git
cd llm-doctor
python -m llm_doctor checks --endpoint http://YOUR_HOST:11434/v1
```

Point `--endpoint` at **any** OpenAI-compatible `/v1` API — or the Ollama native
`/api` one:

```bash
python -m llm_doctor checks --endpoint http://192.0.2.1:11434
      # 192.0.2.1 is an RFC 5737 documentation placeholder - use your server's LAN IP
python -m llm_doctor checks --endpoint http://vllm.yourhost:8000/v1  # vLLM / OpenAI-compat
python -m llm_doctor ping     --endpoint http://YOUR_HOST:11434      # network baseline
```

No root, no SSH, no passwords. Just the HTTP endpoint your agent already talks to.

---

## What it measures

```
llm-doctor checks          # first-token latency as a function of context window
llm-doctor ping            # round-trip time distribution, p50/p90/p99
llm-doctor kv-reuse        # does the model actually reuse its KV cache?
llm-doctor sweep           # all of the above, plus a plain-English verdict
```

Each check runs a **real request** against your endpoint and prints:

- The numbers (TTFB, tok/s, context size, cache-hit ratio)
- A **verdict** in plain words
- **The exact command to fix it**

---

## Why this exists

A user reported "traffic between my agent and my local model crawls — 60s waiting
for the first chunk." We pinged, we measured TCP to the port, we `ss`'d the
sockets — the link was **flat at 2.6ms**. The real cause: an agent had auto-
detected a context window 8× larger than the model would ever keep resident, so
the inference engine rebuilt an oversized KV cache *every single turn* instead of
reusing it. The "network crawl" was the client waiting on that recompute.

The diagnostic that fixed it — *measure first-token latency as a function of the
requested context window* — generalizes. This repo is that diagnostic, packaged.

---

## Why it's safe

`llm-doctor` is **read-only**. Every check sends ordinary chat/ping requests and
parses the response. It does not modify anything on the server, does not store
your model weights, does not touch your credentials. It only *observes*.

---

## Contributing

Found a new symptom? A better model, a new backend, a check that's missing?
PRs welcome — see `.github/ISSUE_TEMPLATE/`.

---

## License

MIT.
