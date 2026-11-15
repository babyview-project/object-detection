#!/usr/bin/env bash
# Full VEDB 1612 run: same model and settings as the BabyView full run (setup B, temperature 1.0, MTP 3, every 5th 1 fps
# frame), frames resized to the BabyView pixel count (--pixels bv: 788 x 591, aspect kept). One fixed shard per GPU in
# $GPUS (default 0-6: GPU 7 lent to Tarun), each starting once that GPU's
# embedding worker (worker k ran on GPU k) has logged "done"; failed shards retry every 2 min; merge at the end.
cd "$(dirname "$0")"
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
read -ra G <<< "${GPUS:-0 1 2 3 4 5 6}"
N=${#G[@]}
ARGS="--task detect_open_format --frames vedb --pixels bv --temperature 1.0 --mtp 3 --max-num-seqs 96 --batch 512"
shard() { local k=$1 g=$2 n=0
  until grep -q "embeddings: worker $g/8 done" $L; do sleep 60; done
  echo "$(date -Is) VEDB run: shard $k/$N starting on GPU $g" >> $L
  until CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY -u run_qwen_pilot.py $ARGS --shard $k/$N >> logs/vedb_shard_$k.log 2>&1; do
    n=$((n + 1)); echo "$(date -Is) VEDB run: shard $k/$N exited with an error (try $n), retrying in 2 min" >> $L; sleep 120
  done
  echo "$(date -Is) VEDB run: shard $k/$N done" >> $L; }
echo "$(date -Is) VEDB run queued on GPUs ${G[*]} ($N shards): $ARGS" >> $L
for i in "${!G[@]}"; do shard $i ${G[$i]} & done
wait
$PY run_qwen_pilot.py $ARGS --merge-shards 6 >> $L 2>&1     # first batches of an earlier 6-shard launch (2026-10-27)
$PY run_qwen_pilot.py $ARGS --merge-shards $N >> $L 2>&1
echo "$(date -Is) VEDB run complete" >> $L
