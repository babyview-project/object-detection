#!/bin/bash
# CLIP + DINOv3 embeddings for the unpacked imagenet-1k and lvis images on
# /data2, using the existing image-embedding/create_image_embeddings.py.
# Matches the CLIP/DINO checkpoints already used for vedb/manuscript analyses
# (analysis/vedb-1612/vedb_config.py) so embeddings are directly comparable.
#
# --input_image_dir points at the dataset's *parent* extracted/ dir (both
# train/ and val/ under it): create_image_embeddings.py derives each image's
# id from its parent-dir name + filename, so train/val images naturally get
# distinct ids (train_..., val_...) in one combined run/output dir.
#
# Shared node: uses Ray's fractional-GPU scheduling (not whole-GPU claims
# like run_yoloe_vedb.sh) on GPUs 2,3,5 -- the same GPUs the vedb detection
# job is lightly using (6-24% compute util), avoiding 0,1,4,6,7 which other
# users have pegged at 79-100% right now. Re-check nvidia-smi before reusing
# this script later; the free set will have moved on.
set -u

REPO="/home/j7yang/babyview-projects/vss2026/object-detection"
PY="/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python"
GPUS="${GPUS:-2,3,5}"
NUM_PROCESSES="${NUM_PROCESSES:-24}"   # 3 GPUs x 8 workers/GPU at the default 0.125 gpus_per_worker
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$LOGDIR"

CLIP_MODEL="openai/clip-vit-base-patch32"
DINO_MODEL="facebook/dinov3-vitb16-pretrain-lvd1689m"

if [[ ! -x "$PY" ]]; then
  echo "missing $PY" >&2
  exit 1
fi

run_one() {
  local input_dir="$1" out_dir="$2" model="$3" tag="$4"
  mkdir -p "$out_dir"
  echo "$(date -Is) start ${tag} model=${model}"
  CUDA_VISIBLE_DEVICES="$GPUS" nice -n 19 "$PY" -u "${REPO}/image-embedding/create_image_embeddings.py" \
    --model_name "$model" \
    --input_image_dir "$input_dir" \
    --out_dir "$out_dir" \
    --data_type images \
    --max_images 987654321 \
    --num_processes "$NUM_PROCESSES" \
    > "${LOGDIR}/embed_${tag}.log" 2>&1
  local rc=$?
  echo "$(date -Is) end ${tag} rc=${rc}"
  return "$rc"
}

IMAGENET_IN="/data2/j7yang/imagenet-1k/extracted"
IMAGENET_OUT="/ccn2a/dataset/babyview/related_data/imagenet_1k/outputs/image_embeddings"
LVIS_IN="/data2/j7yang/lvis/extracted"
LVIS_OUT="/ccn2a/dataset/babyview/related_data/lvis/outputs/image_embeddings"

run_one "$IMAGENET_IN" "$IMAGENET_OUT" "$CLIP_MODEL" "imagenet_clip"  || exit 1
run_one "$IMAGENET_IN" "$IMAGENET_OUT" "$DINO_MODEL" "imagenet_dino" || exit 1
run_one "$LVIS_IN"     "$LVIS_OUT"     "$CLIP_MODEL" "lvis_clip"     || exit 1
run_one "$LVIS_IN"     "$LVIS_OUT"     "$DINO_MODEL" "lvis_dino"     || exit 1

echo "$(date -Is) all embedding runs done"
