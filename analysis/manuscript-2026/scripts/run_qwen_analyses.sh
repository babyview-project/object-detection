#!/usr/bin/env bash
# Manuscript analyses on Qwen3.8-27B's boxes (full 2025.2 run, every 5th 1 fps frame), mapped to valid129
# (annotation/qwen_pilot/build_qwen_packed.py). Two versions, run one after the other:
#   2025.2_qwen       every format (real, toy, picture, screen), as YOLOE + CLIP counted depictions
#   2025.2_qwen_real  real objects only
# No CLIP filter (passed=True, clip_sim=1 in the packed copy). Frames are already one per 5 s, so compare with 2025.2_5s.
# Stage 0 (exemplar tables) then Stage 1 (run_2025_2_analyses.sh) per version; THINGS tables reused from 2025.2.
#   bash analysis/manuscript-2026/scripts/run_qwen_analyses.sh   (VERSIONS="2025.2_qwen" to run one)
set -u
cd "$(dirname "$0")/.."
export BV_FRAME_STRIDE=1 BV_FRAME_OFFSET=0 BV_EMBED_SOURCE=packed BV_ANALYSIS_MODELS=clip,dinov3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python
STORE=/data2/dataset/babyview/2025.2/outputs/object_detections
for tag in ${VERSIONS:-2025.2_qwen 2025.2_qwen_real}; do
  case $tag in 2025.2_qwen) src=$STORE/qwen_packed_valid129_all ;; 2025.2_qwen_real) src=$STORE/qwen_packed_valid129_real ;; esac
  export BV_OUTPUT_TAG=$tag BV_EMBEDDINGS_BASE=$src BV_EXEMPLAR_EMBED_ROOT="$PWD/exemplar_set_embeddings/$tag"
  echo "STAGE0 $tag start $(date -Is)"
  nice -n 19 $PY -uB scripts/exemplar_set_zscore_embeddings.py --source packed --all-category-sets > logs/stage0_packed_$tag.log 2>&1 \
    || { echo "STAGE0 $tag FAILED $(date -Is)"; continue; }
  for s in valid129 valid85; do cp -p exemplar_set_embeddings/2025.2/$s/things_* "$BV_EXEMPLAR_EMBED_ROOT/$s/"; done
  echo "STAGE0 $tag done $(date -Is)"
  nice -n 19 bash scripts/run_2025_2_analyses.sh > logs/analyses_$tag.log 2>&1
  echo "STAGE1 $tag exit $? $(date -Is)"
done
echo "ALL DONE $(date -Is)"
