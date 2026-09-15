#!/bin/bash
# 一键复现：DataLoader pack 在各 DP rank 之间的负载不均。
#
#   bash repro/run_all.sh            # 全部（离线复现 + 录制数据验证）
#   bash repro/run_all.sh --quick    # 只跑离线复现（约 1 分钟）
#
# 依赖：python3（numpy、datasets）、训练代码在
#       /home/y50063564/dspark_project/speculators
#       数据集在 /home/y50063564/data/open_perfectblend_qwen3_4b_700k
# 都不需要 GPU / server / 训练。
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY=/usr/local/python3.12.13/bin/python3

cd "$ROOT"

echo "########################################################################"
echo "# 1. 离线复现：直接用真实 sampler + 数据集生成各 rank 的 pack"
echo "########################################################################"
$PY repro/01_pack_imbalance.py --steps 1500 --compare 2>&1 | grep -v UserWarning | grep -v "from scipy"

if [ "${1:-}" != "--quick" ]; then
  echo
  echo "########################################################################"
  echo "# 2. 录制数据验证：真实训练 run 里的 pack 不均及其后果"
  echo "########################################################################"
  $PY repro/02_validate_on_recorded_run.py 2>&1 | grep -v UserWarning | grep -v "from scipy"
fi

echo
echo "复现完成。结论见 README.md。"
