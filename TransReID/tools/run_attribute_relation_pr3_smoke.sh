#!/usr/bin/env bash
set -Eeuo pipefail

MODE=${1:-all}
case "$MODE" in
  s1|p2|all) ;;
  *) echo "usage: $0 [s1|p2|all]" >&2; exit 2 ;;
esac

PROJECT=/root/autodl-tmp/Avada_ReID/TransReID
PYTHON=/root/miniconda3/bin/python
CONFIG=configs/experiments/siglip2_naflex_caption_attribute_codebook_relation_m_to_ms.yml
ASSET_ROOT=/root/autodl-tmp/precomputed/attribute_codebooks
LOGROOT=/root/autodl-tmp/logs/attribute_relation_pr3
OUTROOT=/root/autodl-tmp/experiments/attribute_relation_pr3_smoke

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TORCH_HOME=/root/autodl-tmp/hf_cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/hf_cache/xdg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$LOGROOT"
cd "$PROJECT"

run_s1() {
  output="$OUTROOT/s1_m_to_ms"
  log="$LOGROOT/smoke_s1.log"
  [[ ! -e "$output" ]] || { echo "refusing existing $output" >&2; return 10; }
  "$PYTHON" tools/train.py --config_file "$CONFIG" \
    SOLVER.MAX_EPOCHS 1 SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 32 \
    DATALOADER.NUM_WORKERS 2 SOLVER.EVAL_PERIOD 1 \
    SOLVER.CHECKPOINT_PERIOD 1 SOLVER.LOG_PERIOD 20 \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.START_EPOCH 1 \
    OBJECTIVE.ATTRIBUTE_RELATION.START_EPOCH 1 \
    OBJECTIVE.ATTRIBUTE_RELATION.QUEUE_SIZE 64 \
    OUTPUT_DIR "$output" >"$log" 2>&1
  grep -q 'attribute_relation' "$log"
  grep -q 'relation_graphs' "$log"
  [[ -s "$output/model_best.pth.tar" ]]
}

run_p2() {
  asset="$ASSET_ROOT/market1501_msmt17_cuhksysu_pr1"
  output="$OUTROOT/p2_m_ms_cs_to_c3"
  log="$LOGROOT/smoke_p2.log"
  [[ ! -e "$output" ]] || { echo "refusing existing $output" >&2; return 20; }
  "$PYTHON" tools/validate_attribute_codebook.py "$asset" \
    --forbid-dataset cuhk03 --verify-caption-file \
    >"$LOGROOT/smoke_p2.asset.json"
  "$PYTHON" tools/train.py --config_file "$CONFIG" \
    DATASETS.SOURCES market1501,msmt17,cuhksysu DATASETS.TARGETS cuhk03 \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.PHRASE_BANK "$asset/phrase_bank.pt" \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.CODEBOOK "$asset/codebook.pt" \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.MANIFEST "$asset/manifest.json" \
    OBJECTIVE.ATTRIBUTE_RELATION.DOMAIN_KEY dataset \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.START_EPOCH 1 \
    OBJECTIVE.ATTRIBUTE_RELATION.START_EPOCH 1 \
    OBJECTIVE.ATTRIBUTE_RELATION.QUEUE_SIZE 64 \
    SOLVER.MAX_EPOCHS 1 SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 32 \
    DATALOADER.NUM_WORKERS 2 SOLVER.EVAL_PERIOD 1 \
    SOLVER.CHECKPOINT_PERIOD 1 SOLVER.LOG_PERIOD 20 \
    OUTPUT_DIR "$output" >"$log" 2>&1
  grep -q 'attribute_relation' "$log"
  grep -q 'relation_graphs' "$log"
  [[ -s "$output/model_best.pth.tar" ]]
}

if [[ "$MODE" == s1 || "$MODE" == all ]]; then run_s1; fi
if [[ "$MODE" == p2 || "$MODE" == all ]]; then run_p2; fi
