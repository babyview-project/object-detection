#!/usr/bin/env bash
# When the 7 original S00320001 shards finish, stop the slower restarted shard 5 and spread its
# remaining frames over all 8 GPUs (run_shards.sh recomputes what is left), then merge.
set -u
cd "$(dirname "$0")"
ARGS=(--task detect_open_format --frames subject:S00320001 --temperature 1.0)
until grep -q "shards finished" logs/subj_S00320001.out 2>/dev/null; do sleep 60; done
echo "$(date -Is) original shards done; stopping shard 5 to rebalance its remaining frames" | tee -a logs/run.log
for p in $(ps -eo pid,args | grep -- "--shard 5/8" | grep -v grep | awk '{print $1}'); do kill "$p"; done
sleep 30
GPU5=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 5)
for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$GPU5" | awk -F', ' '{print $2}'); do
  [ "$(ps -o user= -p "$p")" = "$USER" ] && kill "$p"          # our orphaned vLLM engine only
done
sleep 20
GPUS="0 1 2 3 4 5 6 7" LOGTAG=subj2 bash run_shards.sh "${ARGS[@]}" --mtp 3 --max-num-seqs 96 --batch 512 > logs/subj_S00320001_rebalance.out 2>&1
/data/j7yang/venvs/qwen/bin/python run_qwen_pilot.py "${ARGS[@]}" --merge-shards 8 2>&1 | grep merged | tee -a logs/run.log
/data/j7yang/venvs/qwen/bin/python run_qwen_pilot.py "${ARGS[@]}" --dry-run 2>&1 | head -1 | tee -a logs/run.log
echo "$(date -Is) S00320001 complete" | tee -a logs/run.log
