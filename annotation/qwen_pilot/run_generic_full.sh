#!/usr/bin/env bash
# Second pass on generic labels (toy, animal, food, ...) for every generic box of the BabyView and VEDB full runs
# (qwen_generic_crops.py --tag fullrun_bv / fullrun_vedb; box lists from --prepare --store). One worker per GPU in $GPUS:
# BabyView shard k, then VEDB shard k; failed workers retry every 2 min; resumable. Then merge.
# Output: out/private/qwen_generic_crops_fullrun_{bv,vedb}.jsonl (new names; the original records and stores are unchanged).
cd "$(dirname "$0")"
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
read -ra G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
N=${#G[@]}
w() { local k=$1 g=$2
  for tag in fullrun_bv fullrun_vedb; do
    echo "$(date -Is) generic pass $tag: worker $k/$N starting on GPU $g" >> $L
    until CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY -u qwen_generic_crops.py --tag $tag --shard $k/$N \
          >> logs/generic_${tag}_$k.log 2>&1; do
      echo "$(date -Is) generic pass $tag: worker $k/$N failed, retrying in 2 min" >> $L; sleep 120; done
    echo "$(date -Is) generic pass $tag: worker $k/$N done" >> $L
  done; }
for i in "${!G[@]}"; do w $i ${G[$i]} & done
wait
for tag in fullrun_bv fullrun_vedb; do $PY qwen_generic_crops.py --tag $tag --merge-shards $N >> $L 2>&1; done
echo "$(date -Is) generic pass complete" >> $L
