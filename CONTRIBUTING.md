# Contributing

Thanks for helping make LLM diagnosis easier.

## What this tool is

A single-file Python 3 utility that turns the question "is the network the
bottleneck?" into a real measurement. No dependencies, no root, no SSH.

## What we want

- **Real-world diagnosis reports.** File a `Diagnosis report` issue with the
  output of `detect`, `ping`, `context-scaling`, and `kv-reuse` on your
  endpoint. We collect these for [docs/cases.md](docs/cases.md) and to
  sharpen the verdicts.
- **Bug fixes and new checks.** The check functions are independent; add a
  new one, make it pass the test suite, and open a PR.
- **Backend flavor support.** Ollama and OpenAI-compatible are the two shapes
  we know well. If you know vLLM, SGLang, LM Studio, or vllm-mlx, add a kind
  and a test case.

## Running checks

```
# compile + unit tests (stdlib only, no endpoint needed)
python3 -m unittest -v tests/test_llm_doctor.py

# live run against your own endpoint
python3 llm_doctor.py sweep --endpoint http://192.0.2.1:11434
      # 192.0.2.1 is an RFC 5737 documentation placeholder - use your server's LAN IP
```

## Style

- stdlib only - no third-party deps
- Python 3.9+
- 4-space indent
- No secrets in code, tests, or examples
