#!/usr/bin/env bash
# Stage 0 driver: build exemplar_set_embeddings/ (see 00_build_exemplar_embeddings.md).
#
# Usage:
#   ./run_exemplar_embedding_stage.sh              # valid129 then valid85 (BV script)
#   ./run_exemplar_embedding_stage.sh valid129     # valid129 only
#   ./run_exemplar_embedding_stage.sh all          # both category sets
#   SKIP_BV=1 ./run_exemplar_embedding_stage.sh valid129   # skip slow BV pass
#   RUN_THINGS_BD3_EMBED=1 ./run_exemplar_embedding_stage.sh valid129  # GPU THINGS embed
#
# Environment: optional paths.local.env in this directory (BV_*, THINGS_*).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/paths.local.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/paths.local.env"
  set +a
fi

CATEGORY_ARG="${1:-all}"
SKIP_BV="${SKIP_BV:-0}"
SKIP_THINGS_ZSCORE_BD3="${SKIP_THINGS_ZSCORE_BD3:-0}"
RUN_THINGS_BD3_EMBED="${RUN_THINGS_BD3_EMBED:-0}"
EXECUTE_NOTEBOOK_07="${EXECUTE_NOTEBOOK_07:-0}"

activate_conda() {
  for _conda_sh in \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/data/j7yang/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${_conda_sh}" ]]; then
      # shellcheck source=/dev/null
      source "${_conda_sh}"
      conda activate vislearnlabpy 2>/dev/null || true
      return 0
    fi
  done
}

run_bv_zscore() {
  local extra=()
  if [[ "${CATEGORY_ARG}" == "all" ]]; then
    extra=(--all-category-sets)
  else
    extra=(--category-set "${CATEGORY_ARG}")
  fi
  echo "=== Stage 0a: BabyView exemplar z-score (06 / exemplar_set_zscore_embeddings.py) ==="
  python -u scripts/exemplar_set_zscore_embeddings.py "${extra[@]}"
}

run_things_notebook_07() {
  echo "=== Stage 0b: THINGS CLIP + DINOv3 (07 notebook) ==="
  if [[ "${EXECUTE_NOTEBOOK_07}" == "1" ]]; then
  jupyter nbconvert --execute --to notebook --inplace \
    07_things_exemplar_set_zscore_embeddings.ipynb
  else
    echo "No headless script for THINGS CLIP/DINOv3 — run notebook 07 manually:"
    echo "  jupyter notebook 07_things_exemplar_set_zscore_embeddings.ipynb"
    echo "Or set EXECUTE_NOTEBOOK_07=1 to run via nbconvert (slow; needs jupyter)."
  fi
}

run_things_bd3() {
  if [[ "${RUN_THINGS_BD3_EMBED}" == "1" ]]; then
    echo "=== Stage 0c: THINGS BabyDINOv3 per-image embed (GPU) ==="
    python -u scripts/create_babydinov3_things_embeddings.py \
      --category-set valid129 --skip-existing
  else
    echo "=== Stage 0c: skipped (set RUN_THINGS_BD3_EMBED=1 to run create_babydinov3_things_embeddings.py) ==="
  fi

  if [[ "${SKIP_THINGS_ZSCORE_BD3}" == "1" ]]; then
    echo "=== Stage 0d: skipped (SKIP_THINGS_ZSCORE_BD3=1) ==="
    return 0
  fi

  echo "=== Stage 0d: THINGS BabyDINOv3 category z-score ==="
  local sets=()
  if [[ "${CATEGORY_ARG}" == "all" ]]; then
    sets=(valid129 valid85)
  else
    sets=("${CATEGORY_ARG}")
  fi
  for cs in "${sets[@]}"; do
    BV_CATEGORY_SET="${cs}" python -u scripts/things_exemplar_set_zscore_babydinov3.py
  done
}

check_stage() {
  echo "=== Stage 0e: check required CSVs ==="
  local sets=()
  if [[ "${CATEGORY_ARG}" == "all" ]]; then
    sets=(valid129 valid85)
  else
    sets=("${CATEGORY_ARG}")
  fi
  local failed=0
  for cs in "${sets[@]}"; do
    if ! python scripts/check_exemplar_stage.py --category-set "${cs}"; then
      failed=1
    fi
  done
  return "${failed}"
}

activate_conda

if [[ "${SKIP_BV}" != "1" ]]; then
  run_bv_zscore
else
  echo "=== Stage 0a: skipped (SKIP_BV=1) ==="
fi

run_things_notebook_07
run_things_bd3

if check_stage; then
  echo ""
  echo "Stage 0 complete. Continue with REPRODUCTION.md (01 parallel OK, then 02–05)."
else
  echo ""
  echo "Stage 0 incomplete — missing tables (often notebook 07 CLIP/DINOv3). See 00_build_exemplar_embeddings.md"
  exit 1
fi
