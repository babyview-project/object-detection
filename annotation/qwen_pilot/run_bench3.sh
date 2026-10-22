#!/usr/bin/env bash
# Wait for the bf16 download and two free GPUs, then run setting G (bf16, 2 GPUs, D's options).
set -u
cd "$(dirname "$0")"
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
until grep -q "DOWNLOAD DONE" /data/j7yang/models/download_qwen38_27b_bf16.log; do sleep 30; done
echo "$(date -Is) bf16 download done; waiting for 2 GPUs with >= 40 GB free and < 10% busy" | tee -a logs/bench.log
declare -A streak
while :; do
  free=()
  while IFS=', ' read -r idx f u; do
    if (( f >= 40000 && u < 10 )); then streak[$idx]=$(( ${streak[$idx]:-0} + 1 )); else streak[$idx]=0; fi
    (( ${streak[$idx]} >= 3 )) && free+=("$idx")
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)
  (( ${#free[@]} >= 2 )) && break
  sleep 60
done
G="${free[0]},${free[1]}"
echo "$(date -Is) bench round 3 start on GPUs $G" | tee -a logs/bench.log
CUDA_VISIBLE_DEVICES=$G nice -n 10 $PY -u bench_qwen.py --setting G_bf16_tp2_mtp > logs/bench_G_bf16_tp2_mtp.log 2>&1
$PY bench_qwen.py --report 2>&1 | grep -v -i warn | tee -a logs/bench.log
echo "$(date -Is) bench round 3 done" | tee -a logs/bench.log
