#!/bin/bash
# CLIP + DINOv3 embeddings of LVIS v1 ground-truth box crops (train + val annotations),
# for every LVIS category mapped to a CDI word by WordNet hyponym pooling
# (notebooks/01_cdi_mapping.ipynb with DATASET = "lvis" -> outputs/mapping/lvis_crop_manifest.csv).
#
# Human-annotated boxes: no detector, no CLIP filter, no box-size filter. The
# manifest's xmin..ymax columns make create_image_embeddings.py crop each box
# before embedding; output files are <out>/<model_tag>/lvis_<split>_<ann_id>.npy.
#
# The whole-image LVIS embeddings from run_embeddings_imagenet_lvis.sh are NOT used
# for category analyses (an LVIS image is a multi-object scene) -- crops go to a
# separate crop_embeddings/ tree.
#
# HF_HOME points at the lab cache that holds the gated DINOv3 weights (same as
# image-embedding/dispatch_cdi_crop_pipeline.sh), so the checkpoints match BV/VEDB/ImageNet.
# Shared node: fractional-GPU Ray workers on two GPUs that were free at launch
# (re-check nvidia-smi before reusing; leave others for other users).
set -u

REPO="/home/j7yang/babyview-projects/vss2026/object-detection"
PY="/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python"
GPUS="${GPUS:-3,6}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"   # 2 GPUs x 8 workers at the default 0.125 gpus_per_worker
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
MANIFEST="${REPO}/analysis/comparison-datasets/outputs/mapping/lvis_crop_manifest.csv"
OUT="/ccn2a/dataset/babyview/related_data/lvis/outputs/crop_embeddings"
mkdir -p "$LOGDIR" "$OUT"

export HF_HOME="${HF_HOME:-/ccn2/u/khaiaw/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm}"

run_one() {
  local model="$1" tag="$2"
  echo "$(date -Is) start ${tag} model=${model}"
  CUDA_VISIBLE_DEVICES="$GPUS" nice -n 19 "$PY" -u "${REPO}/image-embedding/create_image_embeddings.py" \
    --model_name "$model" \
    --image_manifest_csv "$MANIFEST" \
    --out_dir "$OUT" \
    --num_processes "$NUM_PROCESSES" \
    > "${LOGDIR}/embed_${tag}.log" 2>&1
  local rc=$?
  echo "$(date -Is) end ${tag} rc=${rc}"
  return "$rc"
}

run_one "openai/clip-vit-base-patch32" "lvis_crops_clip" || exit 1
run_one "facebook/dinov3-vitb16-pretrain-lvd1689m" "lvis_crops_dino" || exit 1
echo "$(date -Is) all LVIS crop embedding runs done"
touch "${LOGDIR}/lvis_crops.DONE"
