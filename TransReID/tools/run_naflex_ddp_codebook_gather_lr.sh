#!/usr/bin/env bash
set -Eeuo pipefail

LR=${1:-5e-6}
case "$LR" in
  5e-6) RUN_TAG=lr_5e6 ;;
  1e-5) RUN_TAG=lr_1e5 ;;
  2e-5) RUN_TAG=lr_2e5 ;;
  *) echo "usage: $0 {5e-6|1e-5|2e-5}" >&2; exit 2 ;;
esac

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT=${RUNTIME_ROOT:-/home/zhangwq/ReID_2026_runtime/autodl-tmp}
PYTHON=${PYTHON:-/home/zhangwq/miniconda3/envs/ReIDEnv/bin/python}
GPU_IDS=${GPU_IDS:-0,2}
EXPECTED_BRANCH=codex/naflex-reid
STATE_DIR="$RUNTIME_ROOT/logs/ddp_global_loss_gather/m_to_ms/caption_codebook_delayed/$RUN_TAG/seed_1234"
OUTPUT_DIR="$RUNTIME_ROOT/experiments/ddp_global_loss_gather/m_to_ms/caption_codebook_delayed/$RUN_TAG/seed_1234"
TRAIN_LOG="$STATE_DIR/train.stdout.log"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "M->MS Caption-C + Codebook; GPUs=$GPU_IDS; global/local batch=256/128"
  echo "60 epochs; LR=$LR; warmup=10; steps=30,50; seed=1234"
  echo "Caption=0.1; Codebook=0.05 from epoch6; Relation=OFF"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi
[[ "$GPU_IDS" =~ ^[0-7],[0-7]$ && "${GPU_IDS:0:1}" != "${GPU_IDS:2:1}" ]] || {
  echo "GPU_IDS must identify two distinct GPUs" >&2; exit 30;
}
# Query failures are fatal. A visible process or >1 GiB allocated blocks launch.
for gpu in ${GPU_IDS//,/ }; do
  apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ -n "$apps" || "$used" -gt 1024 ]]; then
    echo "GPU $gpu occupied: memory=${used}MiB processes=$apps" >&2; exit 31;
  fi
done
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/launch.lock"
flock -n 9 || { echo "experiment already running" >&2; exit 32; }
[[ "$(git -C "$PROJECT_DIR" branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
  echo "wrong branch" >&2; exit 10;
}
[[ -z "$(git -C "$PROJECT_DIR" status --porcelain)" ]] || {
  echo "dirty worktree" >&2; exit 11;
}
[[ -x "$PYTHON" ]] || { echo "missing ReIDEnv Python: $PYTHON" >&2; exit 12; }
[[ -d "$RUNTIME_ROOT/datasets" ]] || { echo "missing datasets" >&2; exit 13; }
[[ -s "$RUNTIME_ROOT/Avada_ReID/TransReID/caption_tools/output/final/v2.4/captions.jsonl" ]] || {
  echo "missing Caption JSONL" >&2; exit 14;
}
[[ ! -e "$STATE_DIR/complete.json" ]] || { echo "already complete"; exit 0; }
[[ ! -e "$STATE_DIR/failed.json" ]] || { echo "failed state exists" >&2; exit 15; }

resume_opts=()
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if [[ -s "$OUTPUT_DIR/checkpoint_latest.pth.tar" ]]; then
    resume_opts=(SOLVER.RESUME_TRAIN True SOLVER.RESUME_PATH "$OUTPUT_DIR/checkpoint_latest.pth.tar")
  else
    echo "refusing non-resumable non-empty output: $OUTPUT_DIR" >&2
    exit 16
  fi
fi

mkdir -p "$OUTPUT_DIR"
HEAD_SHA=$(git -C "$PROJECT_DIR" rev-parse HEAD)
printf '{"run":"caption_codebook_delayed_gather","lr":"%s","global_batch":256,"world_size":2,"gpu_ids":"%s","head":"%s","started_at":"%s"}\n' \
  "$LR" "$GPU_IDS" "$HEAD_SHA" "$(date -Is)" >"$STATE_DIR/running.json"

failed=1
on_exit() {
  code=$?
  if [[ $failed -ne 0 ]]; then
    printf '{"run":"caption_codebook_delayed_gather","lr":"%s","exit_code":%d,"failed_at":"%s"}\n' \
      "$LR" "$code" "$(date -Is)" >"$STATE_DIR/failed.json"
  fi
  rm -f "$STATE_DIR/running.json"
}
trap on_exit EXIT

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_DIR"
export HF_HOME="$RUNTIME_ROOT/hf_cache"
export HF_HUB_CACHE="$RUNTIME_ROOT/hf_cache/hub"
export TORCH_HOME="$RUNTIME_ROOT/hf_cache/torch"
export XDG_CACHE_HOME="$RUNTIME_ROOT/hf_cache/xdg"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJECT_DIR"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/train.py \
  --config_file configs/experiments/base/siglip2_naflex_caption_codebook.yml \
  MODEL.DIST_TRAIN True \
  MODEL.BACKBONE.PRETRAINED_NAME "$RUNTIME_ROOT/pretrained/siglip2/siglip2-base-patch16-naflex" \
  DATASETS.SOURCES market1501 \
  DATASETS.TARGETS msmt17 \
  DATASETS.COMBINEALL False \
  DATASETS.ROOT_DIR "$RUNTIME_ROOT/datasets" \
  OBJECTIVE.CAPTION.FILE "$RUNTIME_ROOT/Avada_ReID/TransReID/caption_tools/output/final/v2.4/captions.jsonl" \
  OBJECTIVE.CAPTION.GATHER_ACROSS_RANKS True \
  OBJECTIVE.TRIPLET.GATHER_ACROSS_RANKS True \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.CAPTION_FILE "$RUNTIME_ROOT/Avada_ReID/TransReID/caption_tools/output/final/v2.4/captions.jsonl" \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.PHRASE_BANK "$RUNTIME_ROOT/precomputed/attribute_codebooks/market1501_pr1/phrase_bank.pt" \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.CODEBOOK "$RUNTIME_ROOT/precomputed/attribute_codebooks/market1501_pr1/codebook.pt" \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.MANIFEST "$RUNTIME_ROOT/precomputed/attribute_codebooks/market1501_pr1/manifest.json" \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.START_EPOCH 6 \
  OBJECTIVE.ATTRIBUTE_CODEBOOK.WEIGHT 0.05 \
  OBJECTIVE.ATTRIBUTE_RELATION.ENABLED False \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.MAX_EPOCHS 60 \
  SOLVER.BASE_LR "$LR" \
  SOLVER.WARMUP_ITERS 10 \
  SOLVER.STEPS '[30, 50]' \
  SOLVER.CHECKPOINT_PERIOD 2 \
  SOLVER.EVAL_PERIOD 2 \
  SOLVER.LOG_PERIOD 20 \
  SOLVER.SEED 1234 \
  OUTPUT_DIR "$OUTPUT_DIR" "${resume_opts[@]}" >"$TRAIN_LOG" 2>&1

[[ -s "$OUTPUT_DIR/model_best.pth.tar" ]] || { echo "missing model_best" >&2; exit 20; }
[[ -s "$OUTPUT_DIR/train_log.txt" ]] || { echo "missing train_log" >&2; exit 21; }

deleted="$STATE_DIR/deleted_non_best_weights.tsv"
: >"$deleted"
while IFS= read -r -d '' file; do
  printf '%s\t%s\n' "$(stat -c %s "$file")" "$file" >>"$deleted"
  rm -f -- "$file"
done < <(
  find "$OUTPUT_DIR" -maxdepth 1 -type f \
    \( -name '*_epoch*.pth*' -o -name 'checkpoint_latest.pth.tar' -o -name 'model_last.pth.tar' \) \
    -print0
)

failed=0
rm -f "$STATE_DIR/failed.json"
printf '{"run":"caption_codebook_delayed_gather","lr":"%s","global_batch":256,"world_size":2,"gpu_ids":"%s","head":"%s","completed_at":"%s","retained":"model_best.pth.tar"}\n' \
  "$LR" "$GPU_IDS" "$HEAD_SHA" "$(date -Is)" >"$STATE_DIR/complete.json"
