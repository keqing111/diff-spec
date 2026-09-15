#!/bin/bash
# ============================================================================
# 01 复现【问题现象】
#
#   多 DP + FSDP2 训练时，某个 rank 周期性取数饥饿，经集合同步放大成全体
#   forward 尖峰。
#
# 默认（离线重算，秒级完成，不启动任何服务，只读本文件夹 data/）：
#     bash repro/01_reproduce_problem.sh
#
# 真实重跑（约 35 分钟；需要卡 11-14 训练 + 卡 10 起 server，且
# /home/y50063564/data/open_perfectblend_qwen3_4b_700k 存在）：
#     bash repro/01_reproduce_problem.sh --run
#
# 参考配置：train dp4(卡11-14) + --num-workers 2 + --prefetch-factor 4
#           + server 单副本(卡10, dp1, port 1132) + 700k 数据 + SAMP_SEED=42
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW="$(dirname "$HERE")"
ORIG=/home/y50063564/dataloader_diag
PY=/usr/local/python3.12.13/bin/python3

RUN_MODE=0
[ "${1:-}" = "--run" ] && RUN_MODE=1

hdr(){ printf '\n\033[1m%s\033[0m\n' "$*"; }
note(){ printf '  → %s\n' "$*"; }

# ---- 选择数据源：data/（已有精简数据）或 exp/（重跑后） -------------------
DATA_DIR="$NEW/data"
if [ "$RUN_MODE" = "1" ]; then
  hdr "== 真实重跑参考配置（E1）=="
  bash "$NEW/scripts/experiment/run_exp.sh" E1_ref_w2_pf4 \
       --workers 2 --prefetch 4 --max-steps 1500 --server-dp 1 --samp-seed 42 --ag-max 6
  hdr "== 真实重跑 all-gather 等待插桩（E1c）=="
  bash "$NEW/scripts/experiment/run_exp.sh" E1c_agwait_ref \
       --workers 2 --prefetch 4 --max-steps 400 --server-dp 1 --samp-seed 42 --ag-max 6
  DATA_DIR="$ORIG/exp"
fi

[ -d "$DATA_DIR/E1_ref_w2_pf4" ] || { echo "缺少 $DATA_DIR/E1_ref_w2_pf4，先跑 --run 或检查数据"; exit 1; }

hdr "0. 参考配置（问题复现条件）"
sed -n '1,20p' "$DATA_DIR/E1_ref_w2_pf4/config.txt" | sed 's/^/     /'

# ---------------------------------------------------------------- 症状 1
hdr "1. 症状：某个 rank 系统性取数饥饿（边①/③）"
note "看 [1] 节：docs/batch 最重的 rank 同时也是 load_p50 最高、慢步率最高的那个"
note "看 [1] 节末行：<1ms 与 >200ms 两模中间几乎为空 ⇒ 双峰，不是队列渐变耗尽"
$PY "$HERE/print_key_metrics.py" "$DATA_DIR/E1_ref_w2_pf4"

# ---------------------------------------------------------------- 症状 2
hdr "2. 症状：慢步呈周期≈2（边③）"
note "看 [3] 节 r3 行：lag1 显著为负、lag2 显著为正 ⇒ 快→慢→快→慢 交替"
note "机制含义：有缓存批次就立刻取到，没有就等整整一轮服务（约 1 个 RTT）"

# ---------------------------------------------------------------- 传导
hdr "3. 传导：全体 rank 等这个饥饿 rank（边④/⑤）"
note "看 [4] 节：ready_skew p50 应约 1s；最后到达者集中在同一个 rank"
note "关键判别：受害者自己的 fwd 最短（~100ms），其余三个 rank 的 fwd 被拉到 ~1.1s"
note "看 [4] 节 gap_before_ag2：等待不在 all-gather 调用本身（该调用仅 ~0.1ms）"
$PY "$HERE/print_key_metrics.py" "$DATA_DIR/E1c_agwait_ref"

# ---------------------------------------------------------------- 对照
hdr "4. 对照：现象在 W≥6 时消失（同一份数据、同一个 seed）"
note "E1(W=2) vs E3_w8(W=8)：step/s 0.79→1.32，全体慢步率 14.9%→6.8%，"
note "ready_skew p50 1067ms→33ms ⇒ 尖峰消失，但这不是数据变均了，是供给变够了"
$PY "$HERE/print_key_metrics.py" \
    "$DATA_DIR/E1_ref_w2_pf4" "$DATA_DIR/E3_w8_pf4" "$DATA_DIR/E3_w12_pf4"

hdr "小结"
echo "  ✅ 已复现：pack 样本数不均 → 最重 rank 慢性饥饿 → 该 rank 晚进 forward"
echo "             → 其余 rank 在「首次消费被聚合参数」处等待 ≈ 迟到量 → 全体 fwd 尖峰"
echo "  ⚠️ 未复现为因果：单步 pack 里样本更多 ⇒ 该步更慢。实测 rank 内"
echo "             corr(docs, log load_ms)≈0.07，慢步率随 docs 仅 47%→62%，"
echo "             慢/快是双峰，与当步 docs 数几乎无关（见本节 [2] 节末尾的反证表）。"
