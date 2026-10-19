#!/usr/bin/env bash
# Detached CPU concat of per-video crop metadata. Safe to re-run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python}"
EMB="${EMB:-/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full}"
CATS="${CATS:-$SCRIPT_DIR/../data/included_categories_valid129.txt}"
LOGDIR="${EMB}/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/index.log"
echo "===== index $(date -Is) =====" >>"$LOG"
setsid nohup env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SCRIPT_DIR" \
  "$PY" -uB "$SCRIPT_DIR/run_cdi_crop_pipeline.py" \
    --stage index \
    --categories-file "$CATS" \
  </dev/null >>"$LOG" 2>&1 &
echo "pid $!  log $LOG"
