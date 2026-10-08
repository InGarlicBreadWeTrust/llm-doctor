# Kind Auto-Detection Notes

llm-doctor auto-detects the backend flavor from the endpoint shape. Two common
gotchas:

## Ollama's /v1 bridge vs /api/tags

Ollama serves both `/v1/chat/completions` (OpenAI-compatible bridge) and
`/api/chat` (native). A naive probe that tries `/v1/models` first will find
Ollama and *think* it is an OpenAI-compatible server, when the `num_ctx` and
`keep_alive` flags it sends are Ollama-specific.

llm-doctor probes `/api/tags` (Ollama-specific) **first**, then falls back to
`/v1/models` (OpenAI-compatible). This avoids misidentifying Ollama as a
generic OpenAI-compatible server.

## Bare host:port

A bare `host:port` without a scheme is normal: most people copy the address
from `ollama ps` or a README. `norm_endpoint()` auto-prefixes `http://`
so both work:

    python3 llm_doctor.py detect --endpoint 192.0.2.1:11434
    python3 llm_doctor.py detect --endpoint http://192.0.2.1:11434

Both produce the same output.

## vLLM / SGLang / OpenAI

For vLLM or SGLang backends, pass the `/v1` suffix explicitly:

    python3 llm_doctor.py list --endpoint http://gpu:8000/v1

## MLX: no context_length in /api/tags

`ollama list` may show `ctx=0` or omit it for MLX builds (Ollama on
Apple Silicon does not stamp `context_length` in the `details` block).
For GGUF models, a `context_length` of 262144 is the classic oversized-
context bug - pin your client's `num_ctx` to 65536 or the model's real
window, rebuild on the server, and re-test.
