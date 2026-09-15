#!/bin/bash
# ============================================================================
# 02 复现【结论链】
#
#   逐个干预实验，说明"谁是受害者 / 为什么卡 / 天花板在哪 / 为什么压方差不管用"。
#
# 默认（离线重算，秒级完成，只读本文件夹 data/）：
#     bash repro/02_reproduce_conclusions.sh
#
# 真实重跑全部 5 组实验（约 1.5 小时；需要卡 10-14 空闲）：
#     bash repro/02_reproduce_conclusions.sh --run
#
# 结论索引：
#   §1 受害者选择机制：docs/batch 最大的 rank 恒为饥饿者，且可用流置换反转
#   §2 均衡只换受害者，不减少尖峰总量（甚至更差）
#   §3 瓶颈切换点：W≤2 卡在 client 侧在飞数；W≥6 卡在 server token 速率
#   §4 server 天花板不是 token 预算、也不是并发槽位
#   §5 压请求尺寸方差：只摊平负担、不减总量；且"每步样本数相同"从未被真正测试
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

DATA_DIR="$NEW/data"
if [ "$RUN_MODE" = "1" ]; then
  hdr "== 真实重跑：§1§2 流置换 / 均衡 / 不均衡（chaained，含 server 启停） =="
  bash "$NEW/scripts/experiment/run_E6_E5.sh"
  hdr "== 真实重跑：§3 worker 扫参 W=1/4/6/8/12 =="
  bash "$NEW/scripts/experiment/run_E3.sh"
  hdr "== 真实重跑：§4 token 预算与并发槽位 =="
  bash "$NEW/scripts/experiment/run_E4C.sh"
  bash "$NEW/scripts/experiment/run_E4D.sh"
  hdr "== 真实重跑：§5 窄带语料（统一请求尺寸） =="
  bash "$NEW/scripts/experiment/run_E7.sh"
  bash "$NEW/scripts/experiment/run_E7b.sh"
  DATA_DIR="$ORIG/exp"
fi

[ -d "$DATA_DIR/E1_ref_w2_pf4" ] || { echo "缺少 $DATA_DIR/E1_ref_w2_pf4"; exit 1; }

# ============================================================ §1 + §2
hdr "§1 受害者选择机制 + §2 均衡只换受害者（E1 / E6 / E5 / E5'，同 seed 同配置）"
note "四组唯一的区别是 sampler 的流分配："
note "  E1  自然不均          docs/batch = 4.67 / 5.49 / 5.82 / 5.90"
note "  E6  SAMP_RANK_PERM=3,2,1,0  精确反流 → 重流从 r3 搬到 r0"
note "  E5  SAMP_BALANCE=docs       四 rank 全部 5.47"
note "  E5' SAMP_BALANCE=skew_docs  人为拉大到 4.59 / 5.41 / 5.83 / 6.19"
note ""
note "预期观测："
note "  (a) 每组里 docs/batch 最高的 rank 恒为饥饿者 —— 剂量-反应"
note "  (b) E6 换流后 r0 变饥饿、原受害者 r3 完全健康 ⇒ 否定「物理 rank/CPU 固定歧视」"
note "  (c) E5 均衡后慢性受害者消失（p50 897→0.1ms），但总慢步率 14.9%→22.4% 反而升"
note "      ⇒ 均衡改的是「谁挨饿」，不是「为什么挨饿」"
$PY "$HERE/print_key_metrics.py" \
    "$DATA_DIR/E1_ref_w2_pf4" "$DATA_DIR/E6_perm_rev" \
    "$DATA_DIR/E5_balance_docs" "$DATA_DIR/E5b_skew_docs"

# ============================================================ §3
hdr "§3 瓶颈切换点（E3 worker 扫参，同 server，seed42，prefetch4）"
note "预期观测：吞吐 W=1/2/4/6/8/12 → 11.3 / 17.4 / 26.2 / 29.0 / 29.0 / 28.7 docs/s"
note "  W≥6 完全饱和 ⇒ 瓶颈由 client 侧转移到 server 侧"
note "  waiting 单调堆积（4→16→25→41）而 running 只到 3-5（上限 8）"
note "  W=2 时 server 尚未饱和：running p50=0 / p90=3 / max=6，waiting max=4"
note "      ⇒ 参考配置下的瓶颈是 client 侧在飞请求数（worker 用同步 client，"
note "        每 worker 只有 1 个在飞），server 算力被浪费 —— 可视为实现缺陷"
$PY "$HERE/print_key_metrics.py" \
    "$DATA_DIR/E3_w1_pf4" "$DATA_DIR/E1_ref_w2_pf4" "$DATA_DIR/E3_w4_pf4" \
    "$DATA_DIR/E3_w6_pf4" "$DATA_DIR/E3_w8_pf4" "$DATA_DIR/E3_w12_pf4"

# ============================================================ §4
hdr "§4 server 天花板既不是 token 预算、也不是并发槽位（E4-C / E4-D）"
note "E4-C：W=8 固定，--max-num-batched-tokens 4096 → 16384 → 32768"
note "  预期：吞吐/tokens per s 完全不变（19243 / 19243 / 18981）"
note "  开关已验证生效（server 启动日志出现 'max_num_batched_tokens': 16384，4096 警告消失）"
note "E4-D：W=8 固定，--max-num-seqs 8 → 16 → 32"
note "  预期：并发 3→6、排队 25→0，但吞吐与 tokens/s 完全不变（19243 / 20115 / 18972）"
note "  ⇒ 天花板 = 每 token 固定成本导致的 token 速率上限（≈19-20k prompt tok/s ≈ 29 docs/s）"
$PY "$HERE/print_key_metrics.py" \
    "$DATA_DIR/E3_w8_pf4" "$DATA_DIR/E4C_w8_tok16384" "$DATA_DIR/E4C_w8_tok32768" \
    "$DATA_DIR/E4D_w8_seqs16" "$DATA_DIR/E4D_w8_seqs32"

# ============================================================ §5
hdr "§5 压请求尺寸方差：只摊平负担、不减总量（E7 / E7b，窄带语料 600-800）"
note "语料换成 doc 长度 {600,800} 带内（请求尺寸 CV 0.82 → ≈0），docs/batch 天然四 rank 相等"
note "⚠️ 口径提示：EVIDENCE.md L2 判读 1 说「统一尺寸显著更稳（支持）」，"
note "   但那是拿 E1(W=2) 比 E7(W=4/8)——同时在飞数也变了，是未匹配的中间结论。"
note "   REPORT.md §6 做了匹配对照（固定 W 只换语料），本脚本采用后者。见 README §6.2。"
note "预期观测（匹配 ρ 与在飞数后对比）："
note "  W8：窄带 6.2%  vs  变长 6.8%   ⇒ 没降"
note "  W2：窄带 17.8% vs  变长 14.9%  ⇒ 反而更差"
note "  ⇒ 它的效果是「消灭固定受害者 + 提高 fast 占比」，不是降低总停顿"
note "  ⇒ 与 §2 的均衡结论一致：改的是时间分布，不是总量"
note ""
note "⚠️ 重要边界：E7/E7b 统一的是请求【尺寸】，不是 pack 的【样本数】。"
note "   窄带下 docs/batch 仍是 {5:68%, 6:32%} —— 32% 的步没有对齐。"
note "   严格的「每步 pack 样本数完全相同」在 dataloader_diag 中【从未实施】，"
note "   仅出现在 REPORT.md §8 的可行性分析表里。见 README.md「未决问题」。"
$PY "$HERE/print_key_metrics.py" \
    "$DATA_DIR/E7_band_w8" "$DATA_DIR/E3_w8_pf4" "$DATA_DIR/E7b_band_w2" "$DATA_DIR/E1_ref_w2_pf4"

# ============================================================ §6
hdr "§6 为什么「改善 pack 不均」减不了尖峰：分支锁存（本文件新增分析）"
note "E7b 四个 rank 请求尺寸相同、docs/batch 相同(5.31-5.34)、"
note "RTT(389-407ms) 与 CPU/doc(112-124ms) 逐项相同，慢步率却是 0.1% vs 38.1%。"
note "⇒ 差异不在生产侧，而在「prefetch 缓冲里有没有现成批次」："
note "   这是一条双稳态分支——先拉开缓冲的 rank 此后恒定命中，"
note "   而 DP 每步集合通信要等齐，快 rank 等待时缓冲不被消耗，"
note "   慢 rank 又每步清空缓冲 ⇒ 开局随机领先被固化，追不回来。"
note "   领先者会易主（E7_band_w8: r2→r1→r3→r1→r3）⇒ 不是固定 rank 属性。"
note ""
note "这解释了 §2/§5 的全部结果："
note "  · 均衡/统一尺寸消除的是「谁更可能输」的**偏置**，不是**双稳态本身**"
note "  · 只要 ρ 还在刀刃附近，总有人输 ⇒ 总停顿不降、只是受害者随机化"
echo
$PY "$HERE/print_key_metrics.py" "$DATA_DIR/E7b_band_w2" 2>&1 | sed -n '/\[5\]/,/\[6\]/p'
echo
echo "  对照 E1（变长 + doc 不均）：最重的 r3 开局就掉队，全程锁死在慢分支"
$PY "$HERE/print_key_metrics.py" "$DATA_DIR/E1_ref_w2_pf4" 2>&1 | sed -n '/\[5\]/,/\[6\]/p'

hdr "结论汇总"
cat <<'EOF'
  ✅ 支持  ① docs/batch 水平差 → 决定谁是受害者（可被流置换反转）
  ✅ 支持  ④ rank 晚进 forward（迟到 p50 882ms、标签 0% 不符）
  ✅ 支持  ⑤ 集合同步把迟到量放大成全体 fwd 尖峰（slope 0.97-0.98、R²=1.00）
  ❌ 否定  ② 累计 docs → prefetch 耗尽 → 若干步后断粮（无 Δ≥1 显著互相关格点）
  ❌ 否定  ②′ 当期批次成本解释等待（load_ms ~ docs 的 R²≈0.013）
  ❌ 否定  H4 物理 rank/CPU 固定歧视（E6：饥饿随流走）
  ❌ 否定  「均衡 docs 即可解决」（E5：受害者消失但总量 14.9%→22.4%）
  ❌ 否定  「token 预算是天花板」（E4-C）
  ❌ 否定  「并发槽位是天花板」（E4-D）
  ❌ 否定  「压方差 ⇒ 高 ρ 稳定」（E7/E7b：只摊平，不减总量）
  ✅ 判据  ρ = 需求/能力：ρ≲0.90 → 慢步≤3%；ρ≈0.92-0.95 → 6%-54%（16 个 rank-样本一致）
EOF
