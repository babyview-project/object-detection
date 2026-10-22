#!/usr/bin/env bash
# Split one Qwen pilot set across several GPUs (one shard per GPU), then merge.
#   GPUS="0 1 2 3" bash run_shards.sh --task detect --frames gold
set -u
cd "$(dirname "$0")"
mkdir -p logs
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
read -ra G <<< "${GPUS:?set GPUS, e.g. GPUS=\"0 1 2 3\"}"
N=${#G[@]}
echo "$(date -Is) shards on GPUs ${G[*]}: $*" | tee -a logs/run.log
pids=()
for k in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$k]} nice -n 10 $PY -u run_qwen_pilot.py "$@" --shard "$k/$N" > "logs/shard_${LOGTAG:-run}_$k.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
$PY run_qwen_pilot.py "$@" --merge-shards "$N" | tee -a logs/run.log
echo "$(date -Is) shards finished (fail=$fail)" | tee -a logs/run.log
