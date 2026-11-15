#!/usr/bin/env bash
# Full 2025.2 run (Qwen3.8-27B-FP8, setup B, temperature 1.0, every 5th 1 fps frame, S00320001 already done):
# 8 fixed shards, shard k on GPU k. Shards on GPUs in $LATER start once the generic-crop pass has finished.
# A shard that exits with an error (e.g. not enough free GPU memory) is retried every 2 minutes; it resumes.
cd "$(dirname "$0")"
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
LATER=${LATER:-"3 4 5 6 7"}
ARGS="--task detect_open_format --frames fullrun --temperature 1.0 --mtp 3 --max-num-seqs 96 --batch 512"
shard() {
  local k=$1 n=0
  case " $LATER " in *" $k "*) until grep -q "generic crop pass done" $L; do sleep 30; done ;; esac
  echo "$(date -Is) full run: shard $k/8 starting on GPU $k" >> $L
  until CUDA_VISIBLE_DEVICES=$k nice -n 10 $PY -u run_qwen_pilot.py $ARGS --shard $k/8 >> logs/full_shard_$k.log 2>&1; do
    n=$((n + 1)); echo "$(date -Is) full run: shard $k/8 exited with an error (try $n), retrying in 2 min" >> $L; sleep 120
  done
  echo "$(date -Is) full run: shard $k/8 done" >> $L
}
echo "$(date -Is) full run launched: $ARGS (shards on GPUs $LATER wait for the generic-crop pass)" >> $L
for k in 0 1 2 3 4 5 6 7; do shard $k & done
wait
$PY run_qwen_pilot.py $ARGS --merge-shards 8 >> $L 2>&1
echo "$(date -Is) full run complete" >> $L
