#!/usr/bin/env bash
# GLM-5.2 DSpark 最简跑通训练脚本（配已跑的 dummy vLLM serve）。
#
# 直接调用 $TRAIN_PY（绝对路径，可覆盖），模型类型 = glm_dspark（GLM-5.2 MLA）。
# serve 用 connector 方式（extract_hidden_states）把 HS 写到 $HS_DIR。
#
# 用法（目标机器，任意目录）:
#   bash <此脚本的路径>
# 覆盖项都是环境变量，见下方定义。
set -eo pipefail
export ASCEND_RT_VISIBLE_DEVICES=4
TRAIN_NPUS="4"
NUM_TRAIN_NPUS=1
_ep_host="$(printf '%s' "$ENDPOINT" | sed -E 's#^https?://([^:/]+).*#\1#')"
export NO_PROXY=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
export no_proxy=localhost,127.0.0.1,80.48.17.186,80.48.17.185,80.48.17.187,80.48.17.188,80.48.17.178
# ================= 路径（可覆盖） =================
REPO_ROOT="${SPECULATORS_ROOT:-/home/y50063564/dspark_project/speculators-for-glm52}"
TRAIN_PY="$REPO_ROOT/scripts/train.py"
VERIFIER="${VERIFIER:-/home/y50063564/dspark_project/fakemodelconfig}"   # serve 的模型（dummy，仅 config）
HS_DIR="${HS_DIR:-/home/y50063564/dspark_project/processed_data}"         # serve 的 shared_storage_path（HS 写这里）
ENDPOINT="${ENDPOINT:-http://80.48.17.178:1123/v1}"                       # serve 地址
DATA="${DATA:-/home/y50063564/inputs/processed_data}"       # Arrow 数据集（token_ids 要和 HS 对齐）
SERVED_MODEL="${SERVED_MODEL:-/home/y50063564/dspark_project/fakemodelconfig}"  # serve 的 OpenAI model id（serve 未加 --served-model-name，注册的是完整路径）

# ================= 训练参数 =================
NUM_LAYERS="${NUM_LAYERS:-1}"            # 草稿层数
TARGET_LAYERS="${TARGET_LAYERS:-1 2}"  # 必须匹配 serve 的 eagle_aux_hidden_state_layer_ids
BLOCK="${BLOCK:-5}"                      # γ
MASK_TOKEN="${MASK_TOKEN:-154856}"
SWA_WINDOW="${SWA_WINDOW:-128}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-1}"
MAX_ANCHORS="${MAX_ANCHORS:-64}"
SEQLEN="${SEQLEN:-3072}"
NPROC="${NPROC:-1}"

# ================= 输出目录 + PID =================
TS="$(date +%Y%m%d_%H%M%S)"
RUN="${RUN:-/home/y50063564/glm_dspark_minimal_run}"
mkdir -p "$RUN"
LOG="$RUN/train_${TS}.log"
PID_FILE="$RUN/train.pid"
SAVE_PATH="${SAVE_PATH:-$RUN/ckpt_${TS}}"
rm -f "$HS_DIR"/hs_*.safetensors*   # 清掉旧 HS，避免和新的 data row 索引撞

# ================= 校验 + 打印实际命令 =================
[ -f "$TRAIN_PY" ] || { echo "!! 没有 $TRAIN_PY —— 用 SPECULATORS_ROOT=<speculators 根目录> 指定"; exit 2; }
[ -d "$DATA" ] || echo "!! 警告: DATA 不存在 $DATA（训练会因无数据失败，先确认 Arrow 数据集路径）"

echo "==============================================================="
echo " GLM-5.2 DSpark MINIMAL"
echo " train.py   = $TRAIN_PY"
echo " speculator = glm_dspark  num_layers=$NUM_LAYERS  aux=$TARGET_LAYERS"
echo " verifier   = $VERIFIER"
echo " hs_dir     = $HS_DIR"
echo " data       = $DATA"
echo " endpoint   = $ENDPOINT  served_model=$SERVED_MODEL"
echo " log        = $LOG"
echo " save       = $SAVE_PATH"
echo "==============================================================="

nohup env \
  PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
  torchrun --nproc_per_node "$NPROC" "$TRAIN_PY" \
    --speculator-type glm_dspark --served-model-name "$SERVED_MODEL" \
    --num-layers "$NUM_LAYERS" \
    --block-size "$BLOCK" --target-layer-ids $TARGET_LAYERS --max-anchors "$MAX_ANCHORS" \
    --sliding-window "$SWA_WINDOW" --sliding-window-non-causal \
    --total-seq-len "$SEQLEN" --mask-token-id "$MASK_TOKEN" \
    --loss-fn '{"ce":0.1,"tv":1.8}' \
    --optimizer adamw --lr "$LR" --epochs "$EPOCHS" \
    --on-missing generate --on-generate delete \
    --num-workers 4 --prefetch-factor 4 \
    --hidden-states-path "$HS_DIR" --vllm-endpoint "$ENDPOINT" \
    --verifier-name-or-path "$VERIFIER" --data-path "$DATA" \
    --save-path "$SAVE_PATH" --log-dir "$RUN" \
  > "$LOG" 2>&1 &
echo $! > "$PID_FILE"

echo ">>> started PID $!  |  tail -f $LOG"
echo ">>> stop: kill \$(cat $PID_FILE)"
