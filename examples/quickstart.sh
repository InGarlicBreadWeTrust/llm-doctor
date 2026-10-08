#!/usr/bin/env bash
# llm-doctor quickstart: auto-detect, list, ping, context-scaling
# Usage: ./examples/quickstart.sh http://192.0.2.1:11434
set -e
EP="${1:-http://127.0.0.1:11434}"
echo "Endpoint: $EP"
echo
echo "=== DETECT ==="
python3 ../llm_doctor.py detect --endpoint "$EP"
echo
echo "=== LIST ==="
python3 ../llm_doctor.py list --endpoint "$EP"
echo
echo "=== PING (100 TCP round-trips) ==="
python3 ../llm_doctor.py ping --endpoint "$EP" --ping-n 100
echo
echo "=== CONTEXT-SCALING (base 4096, 1x/2x/4x/8x) ==="
python3 ../llm_doctor.py context-scaling --endpoint "$EP" --base-ctx 4096
echo
echo "=== KV-REUSE (4 turns @ 16k ctx) ==="
python3 ../llm_doctor.py kv-reuse --endpoint "$EP" --base-ctx 2048
