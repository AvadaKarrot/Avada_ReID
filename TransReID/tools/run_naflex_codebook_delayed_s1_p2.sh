#!/usr/bin/env bash
set -Eeuo pipefail

MODE=${1:-all}
case "$MODE" in
  build|smoke|train|all) ;;
  *) echo "usage: $0 [build|smoke|train|all]" >&2; exit 2 ;;
esac

REPO=/root/autodl-tmp/Avada_ReID
PROJECT="$REPO/TransReID"
PYTHON=/root/miniconda3/bin/python
BASE_CONFIG=configs/experiments/base/siglip2_naflex_caption_codebook.yml
CAPTION_FILE="$PROJECT/caption_tools/output/final/v2.4/captions.jsonl"
ASSET_ROOT=/root/autodl-tmp/precomputed/attribute_codebooks
LOGROOT=/root/autodl-tmp/logs/codebook_delayed_s1_p2
S1_OUT=/root/autodl-tmp/experiments/attribute_codebook_delayed_s1
P2_OUT=/root/autodl-tmp/experiments/protocol2_siglip2_naflex_caption_attribute_codebook_delayed_30ep
SMOKE_OUT=/root/autodl-tmp/experiments/attribute_codebook_delayed_s1_p2_smoke
EXPECTED_BRANCH=codex/codebook-delayed-s1-p2

mkdir -p "$LOGROOT"
exec >>"$LOGROOT/supervisor.log" 2>&1
echo $$ >"$LOGROOT/supervisor.pid"
echo "[$(date -Is)] start mode=$MODE pid=$$"

finished=0
failure_reason="pipeline did not complete"
on_exit() {
  code=$?
  rm -f "$LOGROOT/pipeline.running"
  if [[ $code -ne 0 || $finished -ne 1 ]]; then
    printf '{"mode":"%s","exit_code":%d,"failed_at":"%s","reason":"%s"}\n' \
      "$MODE" "$code" "$(date -Is)" "$failure_reason" >"$LOGROOT/pipeline.failed"
  fi
}
trap on_exit EXIT

[[ "$(git -C "$REPO" branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
  failure_reason="wrong branch"; exit 10;
}
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  failure_reason="dirty worktree"; git -C "$REPO" status --short; exit 11;
}
nvidia-smi -L | grep -q 'RTX PRO 6000' || {
  failure_reason="GPU unavailable"; exit 12;
}
if pgrep -af 'python.*tools/train.py|python.*tools/build_attribute_codebook.py' >/dev/null; then
  failure_reason="training or Codebook build process already exists"; exit 13;
fi
[[ -s "$CAPTION_FILE" ]] || { failure_reason="missing Caption JSONL"; exit 14; }

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TORCH_HOME=/root/autodl-tmp/hf_cache/torch
export XDG_CACHE_HOME=/root/autodl-tmp/hf_cache/xdg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
rm -f "$LOGROOT/pipeline.failed" "$LOGROOT/pipeline.complete"
printf '{"mode":"%s","pid":%d,"started_at":"%s","head":"%s"}\n' \
  "$MODE" "$$" "$(date -Is)" "$HEAD_SHA" >"$LOGROOT/pipeline.running"

validate_asset() {
  asset_dir=$1
  target=$2
  "$PYTHON" "$PROJECT/tools/validate_attribute_codebook.py" \
    "$asset_dir" --forbid-dataset "$target" --verify-caption-file
}

build_asset() {
  name=$1
  sources=$2
  domain_mode=$3
  min_coverage=$4
  asset_dir="$ASSET_ROOT/$name"
  state="$LOGROOT/assets/$name"
  mkdir -p "$state"
  if [[ -s "$asset_dir/manifest.json" && -s "$asset_dir/phrase_bank.pt" && -s "$asset_dir/codebook.pt" ]]; then
    "$PYTHON" "$PROJECT/tools/validate_attribute_codebook.py" \
      "$asset_dir" --verify-caption-file >"$state/validation.json"
    echo "[$(date -Is)] asset verified $name"
    return
  fi
  if [[ -e "$asset_dir" && -n "$(find "$asset_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    failure_reason="partial asset directory exists: $asset_dir"
    return 20
  fi
  mkdir -p "$asset_dir"
  printf '{"name":"%s","sources":"%s","started_at":"%s"}\n' \
    "$name" "$sources" "$(date -Is)" >"$state/running.json"
  cd "$PROJECT"
  set +e
  "$PYTHON" tools/build_attribute_codebook.py \
    --config-file "$BASE_CONFIG" \
    --output-dir "$asset_dir" \
    --caption-file "$CAPTION_FILE" \
    --device cuda \
    --batch-size 256 \
    --seed 1234 \
    --stability-seed 2345 \
    --stability-seed 3456 \
    --max-codes 64 \
    --merge-similarity 0.90 \
    --max-compactness-drop 0.01 \
    --min-support 20 \
    --min-domain-coverage "$min_coverage" \
    --domain-mode "$domain_mode" \
    DATASETS.SOURCES "$sources" \
    DATASETS.COMBINEALL False >"$state/build.log" 2>&1
  code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    rm -f "$state/running.json"
    printf '{"name":"%s","exit_code":%d,"failed_at":"%s"}\n' \
      "$name" "$code" "$(date -Is)" >"$state/failed.json"
    failure_reason="asset build failed: $name"
    return "$code"
  fi
  "$PYTHON" "$PROJECT/tools/validate_attribute_codebook.py" \
    "$asset_dir" --verify-caption-file >"$state/validation.json"
  rm -f "$state/running.json"
  printf '{"name":"%s","sources":"%s","completed_at":"%s"}\n' \
    "$name" "$sources" "$(date -Is)" >"$state/complete.json"
}

build_all_assets() {
  build_asset msmt17_pr1 msmt17 camera 1
  build_asset cuhk03_pr1 cuhk03 camera 1
  build_asset market1501_msmt17_cuhksysu_pr1 market1501,msmt17,cuhksysu dataset 2
  build_asset market1501_cuhksysu_cuhk03_pr1 market1501,cuhksysu,cuhk03 dataset 2
  build_asset msmt17_cuhksysu_cuhk03_pr1 msmt17,cuhksysu,cuhk03 dataset 2
  printf '{"completed_at":"%s"}\n' "$(date -Is)" >"$LOGROOT/assets.complete"
}

cleanup_non_best() {
  output=$1
  manifest=$2
  : >"$manifest"
  while IFS= read -r -d '' file; do
    printf '%s\t%s\n' "$(stat -c %s "$file")" "$file" >>"$manifest"
    rm -f -- "$file"
  done < <(
    find "$output" -maxdepth 1 -type f \
      \( -name '*_epoch*.pth*' -o -name 'checkpoint_latest.pth.tar' -o -name 'model_last.pth.tar' \) \
      -print0
  )
  [[ -s "$output/model_best.pth.tar" ]]
}

train_command() {
  sources=$1
  target=$2
  asset_dir=$3
  output=$4
  shift 4
  "$PYTHON" tools/train.py --config_file "$BASE_CONFIG" \
    DATASETS.SOURCES "$sources" \
    DATASETS.TARGETS "$target" \
    DATASETS.COMBINEALL False \
    SOLVER.SEED 1234 \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.WEIGHT 0.05 \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.START_EPOCH 6 \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.DECAY_START_EPOCH 0 \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.FINAL_WEIGHT 0.0 \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.PHRASE_BANK "$asset_dir/phrase_bank.pt" \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.CODEBOOK "$asset_dir/codebook.pt" \
    OBJECTIVE.ATTRIBUTE_CODEBOOK.MANIFEST "$asset_dir/manifest.json" \
    OUTPUT_DIR "$output" "$@"
}

run_smoke() {
  name=$1
  sources=$2
  target=$3
  asset_name=$4
  output="$SMOKE_OUT/$name"
  log="$LOGROOT/smoke/$name.log"
  mkdir -p "$LOGROOT/smoke"
  validate_asset "$ASSET_ROOT/$asset_name" "$target" >"$LOGROOT/smoke/$name.asset.json"
  rm -rf -- "$output"
  cd "$PROJECT"
  train_command "$sources" "$target" "$ASSET_ROOT/$asset_name" "$output" \
    SOLVER.MAX_EPOCHS 1 \
    SOLVER.IMS_PER_BATCH 16 \
    TEST.IMS_PER_BATCH 32 \
    DATALOADER.NUM_WORKERS 2 \
    SOLVER.EVAL_PERIOD 1 \
    SOLVER.CHECKPOINT_PERIOD 1 >"$log" 2>&1
  [[ -s "$output/model_best.pth.tar" ]] || {
    failure_reason="smoke missing model_best: $name"; return 30;
  }
  printf '{"name":"%s","completed_at":"%s","target_caption_input":false}\n' \
    "$name" "$(date -Is)" >"$LOGROOT/smoke/$name.complete.json"
  rm -rf -- "$output"
}

run_smokes() {
  run_smoke s1_ms_to_m msmt17 market1501 msmt17_pr1
  run_smoke p2_m_ms_cs_to_c3 market1501,msmt17,cuhksysu cuhk03 market1501_msmt17_cuhksysu_pr1
  printf '{"completed_at":"%s"}\n' "$(date -Is)" >"$LOGROOT/smoke.complete"
}

run_one() {
  stage=$1
  name=$2
  sources=$3
  target=$4
  asset_name=$5
  output=$6
  state="$LOGROOT/$stage/$name"
  asset_dir="$ASSET_ROOT/$asset_name"
  mkdir -p "$state"
  if [[ -e "$state/complete.json" ]]; then
    echo "[$(date -Is)] already complete $stage/$name"
    return
  fi
  if [[ -e "$state/failed.json" ]]; then
    failure_reason="failed state exists: $stage/$name"
    return 40
  fi
  validate_asset "$asset_dir" "$target" >"$state/asset_validation.json"
  resume_opts=()
  if [[ -d "$output" && -n "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if [[ -s "$output/checkpoint_latest.pth.tar" ]]; then
      resume_opts=(SOLVER.RESUME_TRAIN True SOLVER.RESUME_PATH "$output/checkpoint_latest.pth.tar")
    else
      failure_reason="non-resumable output exists: $output"
      return 41
    fi
  fi
  mkdir -p "$output"
  printf '{"stage":"%s","name":"%s","sources":"%s","target":"%s","seed":1234,"started_at":"%s","output":"%s"}\n' \
    "$stage" "$name" "$sources" "$target" "$(date -Is)" "$output" >"$state/running.json"
  cd "$PROJECT"
  set +e
  train_command "$sources" "$target" "$asset_dir" "$output" \
    "${resume_opts[@]}" >"$state/train.log" 2>&1
  code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    rm -f "$state/running.json"
    printf '{"stage":"%s","name":"%s","exit_code":%d,"failed_at":"%s"}\n' \
      "$stage" "$name" "$code" "$(date -Is)" >"$state/failed.json"
    failure_reason="training failed: $stage/$name"
    return "$code"
  fi
  [[ -s "$output/model_best.pth.tar" ]] || {
    failure_reason="missing model_best: $stage/$name"; return 50;
  }
  [[ -s "$output/model_last.pth.tar" ]] || {
    failure_reason="missing model_last: $stage/$name"; return 51;
  }
  [[ -s "$output/train_log.txt" ]] || {
    failure_reason="missing train_log: $stage/$name"; return 52;
  }
  cleanup_non_best "$output" "$state/deleted_non_best_weights.tsv"
  rm -f "$state/running.json"
  printf '{"stage":"%s","name":"%s","sources":"%s","target":"%s","seed":1234,"completed_at":"%s","output":"%s","non_best_weights_cleaned":true,"target_caption_input":false}\n' \
    "$stage" "$name" "$sources" "$target" "$(date -Is)" "$output" >"$state/complete.json"
}

run_formal() {
  run_one s1 ms_to_m msmt17 market1501 msmt17_pr1 "$S1_OUT/ms_to_m/seed_1234"
  run_one s1 ms_to_c3 msmt17 cuhk03 msmt17_pr1 "$S1_OUT/ms_to_c3/seed_1234"
  run_one s1 c3_to_ms cuhk03 msmt17 cuhk03_pr1 "$S1_OUT/c3_to_ms/seed_1234"
  run_one s1 c3_to_m cuhk03 market1501 cuhk03_pr1 "$S1_OUT/c3_to_m/seed_1234"
  run_one s1 m_to_c3 market1501 cuhk03 market1501_pr1 "$S1_OUT/m_to_c3/seed_1234"
  printf '{"completed_at":"%s"}\n' "$(date -Is)" >"$LOGROOT/s1.complete"

  run_one p2 m_ms_cs_to_c3 market1501,msmt17,cuhksysu cuhk03 market1501_msmt17_cuhksysu_pr1 "$P2_OUT/m_ms_cs_to_c3/seed_1234"
  run_one p2 m_cs_c3_to_ms market1501,cuhksysu,cuhk03 msmt17 market1501_cuhksysu_cuhk03_pr1 "$P2_OUT/m_cs_c3_to_ms/seed_1234"
  run_one p2 ms_cs_c3_to_m msmt17,cuhksysu,cuhk03 market1501 msmt17_cuhksysu_cuhk03_pr1 "$P2_OUT/ms_cs_c3_to_m/seed_1234"
  printf '{"completed_at":"%s"}\n' "$(date -Is)" >"$LOGROOT/p2.complete"
}

build_all_assets
if [[ "$MODE" == smoke || "$MODE" == all ]]; then
  run_smokes
fi
if [[ "$MODE" == train || "$MODE" == all ]]; then
  run_formal
fi

finished=1
rm -f "$LOGROOT/pipeline.running"
printf '{"mode":"%s","completed_at":"%s","head":"%s","non_best_weights_cleaned":true}\n' \
  "$MODE" "$(date -Is)" "$HEAD_SHA" >"$LOGROOT/pipeline.complete"
echo "[$(date -Is)] complete mode=$MODE"
