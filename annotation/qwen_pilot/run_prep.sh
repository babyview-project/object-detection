#!/usr/bin/env bash
# Prep before the full 2025.2 run (all 8 GPUs): prompt pilot (GPUs 0-1), hierarchy text pass (GPU 2),
# generic-label crop pass on S00320001 (GPUs 3-7). Each step logs to logs/run.log.
cd "$(dirname "$0")"
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
echo "$(date -Is) prep: prompt pilot GPUs 0-1, hierarchy GPU 2, generic crops GPUs 3-7" >> $L
R="--task detect_open_format --temperature 1.0 --mtp 3 --max-num-seqs 96 --batch 512"
( for a in "--frames fullrun500 --prompt-variant specific" "--frames montage72"; do
    CUDA_VISIBLE_DEVICES=0 nice -n 10 $PY -u run_qwen_pilot.py $R $a; done > logs/prep_pilot0.log 2>&1
  echo "$(date -Is) prompt pilot GPU 0 done" >> $L ) &
( for a in "--frames montage72 --prompt-variant specific" "--frames depiction --prompt-variant specific" "--frames depiction"; do
    CUDA_VISIBLE_DEVICES=1 nice -n 10 $PY -u run_qwen_pilot.py $R $a; done > logs/prep_pilot1.log 2>&1
  echo "$(date -Is) prompt pilot GPU 1 done" >> $L ) &
( CUDA_VISIBLE_DEVICES=2 nice -n 10 $PY -u qwen_hierarchy.py > logs/prep_hierarchy.log 2>&1
  echo "$(date -Is) hierarchy pass done (exit $?)" >> $L ) &
pids=()
for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((k + 3)) nice -n 10 $PY -u qwen_generic_crops.py --shard $k/5 > logs/prep_generic_$k.log 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
$PY qwen_generic_crops.py --merge-shards 5 >> $L 2>&1
echo "$(date -Is) generic crop pass done (fail=$fail)" >> $L
wait
echo "$(date -Is) prep all done" >> $L
