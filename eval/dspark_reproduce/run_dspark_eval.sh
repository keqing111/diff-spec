#!/bin/bash
# 对 math_reasoning 跑 DSpark acceptance 评测(需先起 launch_dspark_eval.sh 且 vLLM 就绪)
set -euo pipefail
cd /home/y50063564/dspark_project/speculators/scripts/evaluate
exec python evaluate.py \
    --target http://localhost:1125/v1 \
    --dataset RedHatAI/speculator_benchmarks \
    sweep \
    --subsets math_reasoning
