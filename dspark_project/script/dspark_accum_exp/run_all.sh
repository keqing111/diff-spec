#!/bin/bash
# Orchestrator: (1) start vLLM server on card 15, (2) wait for /v1/models,
# (3) launch both trainers (accum1 on card 13, accum12 on card 14) in parallel.
# Logs under processed_data/dspark_data/dspark_accum/{logs,accum1/logs,accum12/logs}.
set -e
cd "$(dirname "$0")"

bash serve_vllm.sh

# Wait for the hidden-state server to accept requests (Qwen3-4B load can take a
# couple of minutes on Ascend).
echo "Waiting for vLLM server ..."
for i in $(seq 1 120); do
    if curl -sf --noproxy '*' http://80.48.17.178:1123/v1/models >/dev/null 2>&1; then
        echo "vLLM server is up after ~${i}0s"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "vLLM server did not become ready in time; check logs." >&2
        exit 1
    fi
    sleep 10
done

bash train_accum1.sh
bash train_accum12.sh
echo "Both trainers launched. Watch with:"
echo "  tail -f /home/y50063564/processed_data/dspark_data/dspark_accum/accum1/logs/train_accum1_*.log"
echo "  tail -f /home/y50063564/processed_data/dspark_data/dspark_accum/accum12/logs/train_accum12_*.log"
