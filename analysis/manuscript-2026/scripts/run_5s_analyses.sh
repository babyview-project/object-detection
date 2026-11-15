#!/usr/bin/env bash
# 5-second (stride 5, offset 0) reproduction of the 2025.2 analyses: Stage 0, then Stage 1.
# Run from anywhere: bash analysis/manuscript-2026/scripts/run_5s_analyses.sh  (SKIP_STAGE0=1 to rerun Stage 1 only)
set -u
cd "$(dirname "$0")/.."
export BV_FRAME_STRIDE=5 BV_FRAME_OFFSET=0 BV_OUTPUT_TAG=2025.2_5s
export BV_EMBEDDINGS_BASE=/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full
export BV_EXEMPLAR_EMBED_ROOT="$PWD/exemplar_set_embeddings/2025.2_5s"
export BV_EMBED_SOURCE=packed BV_ANALYSIS_MODELS=clip,dinov3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python
if [[ "${SKIP_STAGE0:-0}" == 1 ]]; then echo "STAGE0 skipped"; else
echo "STAGE0 start $(date -Is)"
nice -n 19 $PY -uB scripts/exemplar_set_zscore_embeddings.py --source packed --all-category-sets > logs/stage0_packed_2025.2_5s.log 2>&1 || { echo "STAGE0 FAILED $(date -Is)"; exit 1; }
echo "STAGE0 done $(date -Is)"
# THINGS reference tables do not depend on BabyView frame sampling: reuse the 2025.2 ones.
for s in valid129 valid85; do cp -p exemplar_set_embeddings/2025.2/$s/things_* "$BV_EXEMPLAR_EMBED_ROOT/$s/"; done
fi
nice -n 19 bash scripts/run_2025_2_analyses.sh > logs/analyses_2025.2_5s.log 2>&1
echo "STAGE1 exit $? $(date -Is)"
echo "ALL DONE"
