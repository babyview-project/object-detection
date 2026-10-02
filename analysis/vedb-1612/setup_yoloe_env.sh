#!/bin/bash
set -euo pipefail
# User pip.conf points at pypi.ngc.nvidia.com, which does not resolve here.
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
VENV=/data2/j7yang/venvs/yoloe
REPO=/home/j7yang/babyview-projects/vss2026/object-detection
echo "setup start $(date -Is)"
if [[ ! -x "${VENV}/bin/python" ]]; then
  /data/j7yang/anaconda3/bin/python -m venv "$VENV"
fi
"${VENV}/bin/pip" install -U pip
"${VENV}/bin/pip" install ultralytics supervision pandas tqdm pillow pyyaml huggingface_hub
mkdir -p "${REPO}/yoloe/pretrain"
"${VENV}/bin/python" - << PY
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="jameslahm/yoloe",
    filename="yoloe-v8l-seg.pt",
    local_dir="${REPO}/yoloe/pretrain",
)
print("weights", path)
PY
"${VENV}/bin/python" -c 'from ultralytics import YOLOE; print("YOLOE import ok")'
echo "setup end $(date -Is)"
