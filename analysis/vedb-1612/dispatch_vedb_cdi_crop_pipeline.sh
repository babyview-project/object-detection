#!/usr/bin/env bash
# Regenerate VEDB's CDI crop+CLIP-filter+embed pipeline over all 717 sessions
# (the original run only covered the 80-session pinned subset). Mirrors
# image-embedding/dispatch_cdi_crop_pipeline.sh (the BabyView version) but
# points at vedb-1612's frames/detections/output paths instead of the
# defaults, and skips annotated-crop tars (not needed for the BV-vs-VEDB
# embedding comparison; the original run's annotated tars cost ~24GB for 80
# sessions, which would scale to ~215GB for all 717).
#
# Per-video .done markers under $EMB/_status/ make this resumable: the 80
# already-embedded sessions are skipped automatically, only the new ~637 get
# processed.
#
# Usage:
#   bash dispatch_vedb_cdi_crop_pipeline.sh            # auto-pick GPUs with <8GB used
#   bash dispatch_vedb_cdi_crop_pipeline.sh 2,3,5,6     # explicit GPU indices
set -euo pipefail

REPO="/home/j7yang/babyview-projects/vss2026/object-detection"
SCRIPT_DIR="${REPO}/image-embedding"
PY="${PY:-/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python}"

FRAMES="/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps"
CDI_ROOT="/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi"
EMB="/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/image_embeddings/vedb_full"
LOGDIR="${EMB}/logs"
mkdir -p "$LOGDIR"

export HF_HOME="${HF_HOME:-/ccn2/u/khaiaw/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PREFETCH="${PREFETCH:-8}"

if [[ $# -ge 1 && -n "${1}" ]]; then
  IFS=',' read -r -a GPUS <<< "$1"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '$2 < 8000 {print $1}')
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No free GPUs (<8GB used). Currently:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  exit 1
fi

N=${#GPUS[@]}
echo "Launching $N workers on GPUs: ${GPUS[*]}"
echo "frames=$FRAMES"
echo "cdi_root=$CDI_ROOT"
echo "emb_root=$EMB"
echo "Logs: $LOGDIR"

i=0
for G in "${GPUS[@]}"; do
  log="$LOGDIR/embed_gpu${G}_shard${i}of${N}.log"
  {
    echo "===== launch $(date -Is) gpu=$G shard=${i}/${N} batch=$BATCH_SIZE prefetch=$PREFETCH ====="
  } >>"$log"
  setsid nohup env CUDA_VISIBLE_DEVICES="$G" HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SCRIPT_DIR" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS" \
    PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    "$PY" -uB "$SCRIPT_DIR/run_cdi_crop_pipeline.py" \
      --stage embed --shard "${i}/${N}" \
      --frames-root "$FRAMES" --cdi-root "$CDI_ROOT" --emb-root "$EMB" \
      --annotated-format none \
      --batch-size "$BATCH_SIZE" --prefetch-frames "$PREFETCH" \
      </dev/null >>"$log" 2>&1 &
  echo "  gpu $G  shard ${i}/${N}  pid $!  log $log"
  i=$((i + 1))
done

echo "Monitor:  tail -f $LOGDIR/embed_gpu*_shard*.log"
echo "When every shard prints 'finished':"
echo "  $PY $SCRIPT_DIR/run_cdi_crop_pipeline.py --stage index --frames-root $FRAMES --cdi-root $CDI_ROOT --emb-root $EMB"
