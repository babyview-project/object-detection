#!/usr/bin/env bash
# Run CLIP+DINOv3 manuscript analyses on 2025.2 packed shards.
# Writes to main_results_valid129s_2025.2/ and supplemental_results_valid85cats_2025.2/
# (does not overwrite the 868-hour 04302026 trees). Copies public-safe outputs to the capsule.
set -u
fail=0

PY="${PY:-/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python}"
MS="$(cd "$(dirname "$0")/.." && pwd)"
OD_ROOT="$(cd "$MS/../.." && pwd)"
CAP="${CAP:-$HOME/babyview-projects/vss2026/babyview-objects-manuscript-capsule}"
LOG="$MS/logs/analyses_2025.2.log"

export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/mpl-j7yang
export BV_EMBEDDINGS_BASE=/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full
export BV_EXEMPLAR_EMBED_ROOT="$MS/exemplar_set_embeddings/2025.2"
export BV_OUTPUT_TAG=2025.2
export BV_ANALYSIS_MODELS=clip,dinov3
export BV_EMBED_SOURCE=packed

cd "$MS"
mkdir -p logs "$MPLCONFIGDIR"

run_nb() {
  local nb="$1"
  local set="${2:-}"
  if [[ -n "$set" ]]; then
    export CATEGORY_SET="$set"
  fi
  echo "==== notebook $nb CATEGORY_SET=${CATEGORY_SET:-} ===="
  if ! "$PY" -uB scripts/execute_notebook.py "$nb"; then
    echo "FAILED notebook $nb CATEGORY_SET=${CATEGORY_SET:-}"
    fail=1
  fi
}

run_py() {
  echo "==== $* ===="
  if ! "$PY" -uB "$@"; then
    echo "FAILED $*"
    fail=1
  fi
}

echo "Starting 2025.2 analyses $(date -Is)"
echo "BV_EXEMPLAR_EMBED_ROOT=$BV_EXEMPLAR_EMBED_ROOT"
echo "BV_OUTPUT_TAG=$BV_OUTPUT_TAG"

run_nb 01_long_tailed_distribution.ipynb valid129
run_nb 01_long_tailed_distribution.ipynb valid85

run_nb 02_category-wise_cosine_sim.ipynb valid129
run_nb 02_category-wise_cosine_sim.ipynb valid85
run_nb 03_bv_things_rdm_comparison.ipynb valid129
run_nb 03_bv_things_rdm_comparison.ipynb valid85

export CATEGORY_SET=valid129
run_py scripts/compute_binary_template_correlations.py
export CATEGORY_SET=valid85
run_py scripts/compute_binary_template_correlations.py

run_py scripts/clip_threshold_sensitivity.py --category-set both

run_nb 05_within_between_cdi_cluster_correlation.ipynb valid129
run_nb 04_individual_rdms.ipynb valid129
run_nb 08_top8_kid_category_cosine_similarity.ipynb valid129

export TOP8_CATEGORY_SET=valid85
export BV_EMBED_MODEL=dinov3
run_py scripts/top8_within_between_vs_things.py
export BV_EMBED_MODEL=clip
# clip top-8 is invoked from notebook 05; rerun anyway so the 2025.2 folder is complete
run_py scripts/top8_within_between_vs_things.py

run_nb 09_top8_local_global_variability_valid85.ipynb valid85

if [[ -d "$CAP" ]]; then
  echo "==== copy public-safe 2025.2 outputs to capsule ===="
  mkdir -p "$CAP/analysis/manuscript-2026"
  rsync -a "$MS/exemplar_set_embeddings/2025.2/" \
    "$CAP/analysis/manuscript-2026/exemplar_set_embeddings/2025.2/"
  rsync -a \
    --exclude '*all_subjects*' \
    --exclude 'subject_*' \
    --exclude '*.npz' \
    --exclude '*/cache/' \
    "$MS/main_results_valid129s_2025.2/" \
    "$CAP/analysis/manuscript-2026/main_results_valid129s_2025.2/"
  rsync -a \
    --exclude '*all_subjects*' \
    --exclude 'subject_*' \
    --exclude '*.npz' \
    --exclude '*/cache/' \
    "$MS/supplemental_results_valid85cats_2025.2/" \
    "$CAP/analysis/manuscript-2026/supplemental_results_valid85cats_2025.2/"
  cp -a "$MS/manuscript_config.py" "$CAP/analysis/manuscript-2026/"
  for f in packed_crop_embeddings.py exemplar_set_zscore_embeddings.py \
           clip_threshold_sensitivity.py top8_within_between_vs_things.py \
           compute_binary_template_correlations.py execute_notebook.py \
           frame_prevalence.py run_long_tail_frame_prevalence.py; do
    cp -a "$MS/scripts/$f" "$CAP/analysis/manuscript-2026/scripts/"
  done
fi

echo "Finished 2025.2 analyses $(date -Is) fail=$fail"
echo "Results: $MS/main_results_valid129s_2025.2"
echo "         $MS/supplemental_results_valid85cats_2025.2"
exit "$fail"
