#!/bin/bash
# 1 fps JPEG frames for the 80-session VEDB world subset.
# Matches yoloe/extract_frames.py (fps=1, qscale 1, frame PTS names).
set -u
REPO="/home/j7yang/babyview-projects/vss2026/object-detection"
VIDEOS="/ccn2a/dataset/babyview/related_data/vedb_1612_world"
FRAMES="/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps"
LOGDIR="${REPO}/analysis/vedb-1612/logs"
WORKERS="${WORKERS:-2}"
mkdir -p "$FRAMES" "$LOGDIR"

extract_one() {
  local video="$1"
  local name out
  name="$(basename "$video" .mp4)"
  out="${FRAMES}/${name}"
  if [[ -f "${out}/.done" ]]; then
    echo "skip ${name}"
    return 0
  fi
  rm -rf "$out"
  mkdir -p "$out"
  if ffmpeg -i "$video" -vf fps=1 -qscale:v 1 -video_track_timescale 1000 -frame_pts 1 \
      "${out}/%08d.jpg" -hide_banner -loglevel error; then
    touch "${out}/.done"
    echo "done ${name} $(find "$out" -name '*.jpg' | wc -l) frames"
  else
    echo "FAIL ${name}"
    return 1
  fi
}
export -f extract_one
export FRAMES

echo "extract start $(date -Is) workers=${WORKERS}"
mapfile -t VIDEOS_LIST < <(find "$VIDEOS" -maxdepth 1 -type f -iname '*.mp4' | sort)
echo "videos ${#VIDEOS_LIST[@]}"
printf '%s\n' "${VIDEOS_LIST[@]}" | xargs -n 1 -P "$WORKERS" bash -c 'extract_one "$1"' _
fail=0
for video in "${VIDEOS_LIST[@]}"; do
  name="$(basename "$video" .mp4)"
  if [[ ! -f "${FRAMES}/${name}/.done" ]]; then
    echo "missing ${name}"
    fail=1
  fi
done
njpg=$(find "$FRAMES" -name '*.jpg' | wc -l)
echo "extract end $(date -Is) jpgs=${njpg} fail=${fail}"
exit "$fail"
