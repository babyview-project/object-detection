#!/bin/bash
# YOLOE CDI detections on the VEDB 1 fps frames.
# Writes <video>/bounding_box_predictions.csv under the detections dir.
# Annotated JPEGs are not kept (save_frame_every is huge); the CLIP stage draws passers later.
set -u
REPO="/home/j7yang/babyview-projects/vss2026/object-detection"
FRAMES="/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps"
DET="/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi"
CKPT="${REPO}/yoloe/pretrain/yoloe-v8l-seg.pt"
# Manuscript detections (2025.2 CDI, owned by tanawm) used this env:
# ultralytics 8.3.39, torch 2.7.0+cu126. Do not pip-install into it.
PY="/data/tanawm/miniconda3/envs/yoloe/bin/python"
LOGDIR="${REPO}/analysis/vedb-1612/logs"
mkdir -p "$DET" "$LOGDIR"

if [[ ! -x "$PY" ]]; then
  echo "missing $PY" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "missing $CKPT" >&2
  exit 1
fi

# Shared node: never take more than 4 GPUs, and only start when at least 3
# are free so at least 2 stay available for other people.
MAX_GPUS=4
while true; do
  mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$2+0 < 8000 {gsub(/ /,"",$1); print $1}')
  nfree=${#FREE[@]}
  if (( nfree >= 3 )); then
    break
  fi
  echo "$(date -Is) only ${nfree} free GPU(s) (${FREE[*]:-none}); waiting" >&2
  sleep 120
done
USE=("${FREE[@]:0:MAX_GPUS}")
N=${#USE[@]}
echo "free GPUs: ${FREE[*]}  using: ${USE[*]}  (cap ${MAX_GPUS}, leaving the rest)"

mapfile -t NAMES < <("$PY" - << 'PY'
import pandas as pd
from pathlib import Path
csv = Path("/home/j7yang/babyview-projects/vss2026/object-detection/yoloe/tools/MCDI_items_with_AoA.csv")
df = pd.read_csv(csv)
nouns = df[df["lexical_category"] == "nouns"]
nouns = nouns[~nouns["uni_lemma"].astype(str).str.contains(" ", na=False)]
words = nouns["uni_lemma"].astype(str).tolist() + ["person"]
print("\n".join(words))
PY
)
echo "prompts ${#NAMES[@]}"

cd "${REPO}/yoloe"
pids=()
for i in "${!USE[@]}"; do
  gpu="${USE[$i]}"
  echo "launch rank ${i} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    YOLO_AUTOINSTALL=false \
    nice -n 19 "$PY" -u run_yoloe.py \
      --source "$FRAMES" \
      --output "$DET" \
      --checkpoint "$CKPT" \
      --device cuda \
      --rank_id "$i" \
      --num_parallel "$N" \
      --confidence 0.1 \
      --save_frame_every 1000000 \
      --names "${NAMES[@]}" \
      > "${LOGDIR}/yoloe_rank${i}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "rank pid ${pid} failed"
    fail=1
  fi
done
ncsv=$(find "$DET" -name bounding_box_predictions.csv | wc -l)
echo "yoloe end $(date -Is) csvs=${ncsv} fail=${fail}"
exit "$fail"
