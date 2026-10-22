#!/usr/bin/env bash
# Qwen3.8-27B-FP8 pilot on the Gemini pilot's frame sets. Needs one free A40 (FP8 weights
# ~31 GB + KV cache); refuses to start on busy GPUs unless GPUS is given.
#
#   bash run_qwen_pilot.sh            # pick a free GPU, smoke test, then every set
#   GPUS=3 bash run_qwen_pilot.sh     # use GPU 3
#   STAGE=smoke bash run_qwen_pilot.sh
# Progress: tail -f logs/run.log ; outputs in ../gemini_pilot/out/private/qwen3.8-27b-fp8__*.jsonl
set -u
cd "$(dirname "$0")"
mkdir -p logs
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# FlashInfer would JIT-compile its sampler (needs nvcc, not on this node): use PyTorch sampling
export VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
STAGE=${STAGE:-all}

if [[ -z "${GPUS:-}" ]]; then
  # free = >= 40 GB unused (wait_for_gpu.sh also requires it on 3 checks in a row)
  GPUS=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
         awk -F', ' '$2 >= 40000 {print $1}' | head -1)
  if [[ -z "$GPUS" ]]; then
    echo "No free GPU (need >= 40 GB free). Current state:"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
    exit 1
  fi
fi
export CUDA_VISIBLE_DEVICES=$GPUS
echo "$(date -Is) model=$QWEN_MODEL GPUs=$CUDA_VISIBLE_DEVICES stage=$STAGE" | tee -a logs/run.log

run() { echo "==== $* $(date +%T)" | tee -a logs/run.log; nice -n 10 $PY -u run_qwen_pilot.py "$@" 2>&1 | tee -a logs/run.log | grep -E "frames/s|DONE|Error|error" ; }

run --task detect_open_format --frames depiction --limit 8          # smoke: loads, parses, timing
SMOKE=../gemini_pilot/out/private/qwen3.8-27b-fp8__detect_open_format__p1_depiction_frames_lim8.jsonl
if ! grep -q '"error": null' "$SMOKE" 2>/dev/null; then
  echo "$(date -Is) SMOKE FAILED: no parsed reply in $SMOKE; see logs/run.log" | tee -a logs/run.log
  exit 1
fi
echo "$(date -Is) smoke ok: $(grep -c '"error": null' "$SMOKE")/8 replies parsed" | tee -a logs/run.log
[[ "$STAGE" == smoke ]] && exit 0
run --task detect_open_format --frames depiction                    # 80 frames, human format tags
run --task detect_open_format --frames depiction --sampling qwen
run --task detect_open_format --frames fullrun500                   # vs the Gemini batch pilot: looping
run --task detect_open_format --frames fullrun500 --sampling qwen
run --task detect_open_format --frames gold_v2                      # 1,000 gold set v2 frames (RA truth later)
run --task detect_open_format --frames montage72
run --task detect --frames gold                                     # 5,919 frames, CDI list, vs Gemini detect
echo "$(date -Is) ALL DONE" | tee -a logs/run.log
