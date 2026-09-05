#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/root/autodl-tmp}"
GPU_IDS="${GPU_IDS:-0,2}"
MODE="${MODE:-dry-run}"
RUN_NAME="${RUN_NAME:-image_only}"

cd "$PROJECT_DIR"
exec python tools/run_naflex_legacy_bs256_gate.py \
  --mode "$MODE" \
  --only "$RUN_NAME" \
  --runtime-root "$RUNTIME_ROOT" \
  --gpu-ids "$GPU_IDS" \
  "$@"
