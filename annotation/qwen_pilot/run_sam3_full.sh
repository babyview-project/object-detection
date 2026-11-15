#!/usr/bin/env bash
# SAM 3 check of every Qwen box (sam3_full_run.py): one worker per GPU in $GPUS, BabyView shard k then VEDB shard k;
# failed workers retry every 2 min; resumable (finished videos are skipped). Output: <store>/sam3/<video_id>.csv.gz
cd "$(dirname "$0")"
unset HF_HOME
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
read -ra G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
N=${#G[@]}
w() { local k=$1 g=$2
  for st in bv vedb; do
    echo "$(date -Is) SAM 3 $st: worker $k/$N starting on GPU $g" >> $L
    until CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nice -n 10 $PY -u -W ignore sam3_full_run.py --store $st --worker $k/$N \
          >> logs/sam3_${st}_$k.log 2>&1; do
      echo "$(date -Is) SAM 3 $st: worker $k/$N failed, retrying in 2 min" >> $L; sleep 120; done
    echo "$(date -Is) SAM 3 $st: worker $k/$N done" >> $L
  done; }
for i in "${!G[@]}"; do w $i ${G[$i]} & done
wait
echo "$(date -Is) SAM 3 full run complete" >> $L
