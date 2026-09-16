#!/bin/bash
# Wait for both ctx-only A/B arms to exit, then print the epoch-2 comparison
# against the reference runs. Zombie-safe (a finished torchrun stays <defunct>
# and `kill -0` still succeeds on it).
ROOT=/home/y50063564/processed_data/dspark_data/dspark_ctxonly_ab
REF=/home/y50063564/processed_data/dspark_data/dspark_accum
OUT=/home/y50063564/processed_data/dspark_data/dspark_ctxonly_ab/AB_RESULT.txt

log() { echo "[$(date -Is)] $*"; }

pids=()
for ARM in fix legacy; do
    p=$(cat "$ROOT/$ARM/logs/train.pid" 2>/dev/null || echo "")
    [ -n "$p" ] && pids+=("$p")
done
log "watching pids: ${pids[*]}"

deadline=$(( $(date +%s) + 14 * 3600 ))
for p in "${pids[@]}"; do
    while :; do
        st=$(ps -o stat= -p "$p" 2>/dev/null) || break
        case "$st" in *Z*) break ;; esac
        [ "$(date +%s)" -gt "$deadline" ] && { log "TIMEOUT on $p"; break 2; }
        sleep 60
    done
done
log "both arms exited"

val() {  # val <run_root> -> "ep0 ep1 ep2" accept_len
    for e in 0 1 2; do
        f="$1/checkpoints/$e/val_metrics.json"
        if [ -f "$f" ]; then
            python3 -c "import json;print('%.4f'%json.load(open('$f'))['accept_len_epoch'])" 2>/dev/null
        else
            echo "-"
        fi
    done | tr '\n' ' '
}

{
    echo "ctx-only anchor-leak A/B  —  $(date -Is)"
    echo
    echo "5w (open_perfectblend_qwen3_4b_50k), 3 epochs, accum=12, single card."
    echo "Only variable: --ctx-only-anchor-in-context (legacy) vs default (fix)."
    echo
    printf "%-34s %-9s %-9s %-9s\n" "run" "ep0" "ep1" "ep2"
    printf "%-34s %-9s %-9s %-9s\n" "---" "---" "---" "---"
    printf "%-34s %s\n" "fix    (card 11, no leak)"   "$(val "$ROOT/fix")"
    printf "%-34s %s\n" "legacy (card 12, leaking)"  "$(val "$ROOT/legacy")"
    echo
    echo "同一份 5w 的历史参照（老代码）:"
    printf "%-34s %s\n" "plain GQA accum12 (accum12)"  "$(val "$REF/accum12")"
    printf "%-34s %s\n" "ctx-only leaky (gqa_ctx_l0_a12)" "$(val "$REF/gqa_ctx_l0_a12")"
    echo
    echo "读法:"
    echo "  legacy 应复现 gqa_ctx_l0_a12 的 ~4.05；复现不出则本框架与老代码不等价。"
    echo "  fix 若回落到 plain GQA (~3.67) 附近 -> 增益主要是泄漏。"
    echo "  fix 若明显高于 plain GQA -> 去掉 block 内自条件化本身也有部分真实收益。"
    echo
    echo "逐微步指标: $ROOT/{fix,legacy}/metrics/train_metrics.csv（需自行从日志抽取）"
    echo "原始日志  : $ROOT/{fix,legacy}/logs/"
} | tee "$OUT"
log "wrote $OUT"
