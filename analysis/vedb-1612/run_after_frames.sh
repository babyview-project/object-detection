#!/bin/bash
# Wait until every VEDB video has frames and the YOLOE env exists, then detect.
set -u
REPO="/home/j7yang/babyview-projects/vss2026/object-detection"
FRAMES="/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps"
VIDEOS="/ccn2a/dataset/babyview/related_data/vedb_1612_world"
PY="/data2/j7yang/venvs/yoloe/bin/python"
LOGDIR="${REPO}/analysis/vedb-1612/logs"
mkdir -p "$LOGDIR"
expected=$(find "$VIDEOS" -maxdepth 1 -type f -iname '*.mp4' | wc -l)
echo "waiting for ${expected} frame dirs and ${PY} $(date -Is)"
while true; do
  ndone=$(find "$FRAMES" -name .done | wc -l)
  if (( ndone >= expected )) && [[ -x "$PY" && -f "${REPO}/yoloe/pretrain/yoloe-v8l-seg.pt" ]]; then
    break
  fi
  echo "$(date -Is) frames ${ndone}/${expected} py=$([[ -x $PY ]] && echo yes || echo no)"
  sleep 60
done
echo "starting yoloe $(date -Is)"
exec bash "${REPO}/analysis/vedb-1612/run_yoloe_vedb.sh"
