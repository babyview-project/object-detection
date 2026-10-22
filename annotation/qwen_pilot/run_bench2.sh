#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
echo "$(date -Is) bench round 2 start" | tee -a logs/bench.log
pids=(); i=0
for s in C2_compact E_noschema F_compact_mtp; do
  CUDA_VISIBLE_DEVICES=$i nice -n 10 $PY -u bench_qwen.py --setting $s > logs/bench_$s.log 2>&1 &
  pids+=($!); i=$((i+1))
done
for p in "${pids[@]}"; do wait "$p"; done
$PY bench_qwen.py --report 2>&1 | grep -v -i warn | tee -a logs/bench.log
echo "$(date -Is) bench round 2 done" | tee -a logs/bench.log
