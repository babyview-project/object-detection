#!/usr/bin/env bash
# Stage 0a — BabyView exemplar z-score (notebook 06). See 00_build_exemplar_embeddings.md.
# Start in a detached tmux session (safe to disconnect SSH).
#
# Usage:
#   ./run_06_zscore_tmux.sh              # valid129 then valid85, BabyDINOv3 on
#   ./run_06_zscore_tmux.sh valid129     # valid129 only
#   tmux attach -t bv_zscore06           # watch progress
#   tail -f analysis/manuscript-2026/logs/exemplar_set_zscore_*.log

set -euo pipefail

SESSION_NAME="${BV_ZSCORE_TMUX_SESSION:-bv_zscore06}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/exemplar_set_zscore_${STAMP}.log"

CATEGORY_ARG="${1:-}"
EXTRA_PY_ARGS=()
if [[ -z "${CATEGORY_ARG}" ]]; then
  EXTRA_PY_ARGS=(--all-category-sets)
elif [[ "${CATEGORY_ARG}" == "all" ]]; then
  EXTRA_PY_ARGS=(--all-category-sets)
else
  EXTRA_PY_ARGS=(--category-set "${CATEGORY_ARG}")
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "  attach:  tmux attach -t ${SESSION_NAME}"
  echo "  or kill: tmux kill-session -t ${SESSION_NAME}"
  exit 1
fi

RUN_CMD="cd ${SCRIPT_DIR} && export PYTHONUNBUFFERED=1"
for _conda_sh in \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/data/j7yang/anaconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${_conda_sh}" ]]; then
    RUN_CMD+=" && source ${_conda_sh} && conda activate vislearnlabpy"
    break
  fi
done
RUN_CMD+=" && python -u scripts/exemplar_set_zscore_embeddings.py ${EXTRA_PY_ARGS[*]} 2>&1 | tee -a ${LOG_FILE}; echo EXIT_CODE=\$? | tee -a ${LOG_FILE}; exec bash"

tmux new-session -d -s "${SESSION_NAME}" bash -lc "${RUN_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "  tmux attach -t ${SESSION_NAME}"
echo "  tail -f ${LOG_FILE}"
echo "  tmux ls"
