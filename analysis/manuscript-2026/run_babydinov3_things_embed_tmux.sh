#!/usr/bin/env bash
# Embed THINGS images with BabyDINOv3 in a detached tmux session (~1.8k images for valid129).
#
# Usage:
#   ./run_babydinov3_things_embed_tmux.sh
#   tmux attach -t bv_things_bd3
#   tail -f analysis/manuscript-2026/logs/babydinov3_things_embed_*.log

set -euo pipefail

SESSION_NAME="${BV_THINGS_BD3_TMUX_SESSION:-bv_things_bd3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/babydinov3_things_embed_${STAMP}.log"

CHECKPOINT_STEP="${BV_BABYDINOV3_CHECKPOINT_STEP:-119999}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "  attach:  tmux attach -t ${SESSION_NAME}"
  echo "  or kill: tmux kill-session -t ${SESSION_NAME}"
  exit 1
fi

RUN_CMD="cd ${REPO_ROOT} && export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU}"
for _conda_sh in \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "/data/j7yang/anaconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh"; do
  if [[ -f "${_conda_sh}" ]]; then
    RUN_CMD+=" && source ${_conda_sh} && conda activate vislearnlabpy"
    break
  fi
done
RUN_CMD+=" && python -u analysis/manuscript-2026/scripts/create_babydinov3_things_embeddings.py"
RUN_CMD+=" --checkpoint-step ${CHECKPOINT_STEP} --category-set valid129 --skip-existing"
RUN_CMD+=" 2>&1 | tee -a ${LOG_FILE}; echo EXIT_CODE=\$? | tee -a ${LOG_FILE}; exec bash"

tmux new-session -d -s "${SESSION_NAME}" bash -lc "${RUN_CMD}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo "GPU: ${GPU}  checkpoint: ${CHECKPOINT_STEP}"
echo ""
echo "  tmux attach -t ${SESSION_NAME}"
echo "  tail -f ${LOG_FILE}"
