#!/usr/bin/env bash
# SAM 3 padded-crop pass on the BV / VEDB embedding sample (sam3_crop_sample.py): one worker per GPU in $GPUS,
# idle I/O priority (ionice -c3) and nice; failed workers retry every 2 min; resumable.
cd "$(dirname "$0")"
unset HF_HOME
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
read -ra G <<< "${GPUS:-0 1 2 3 4 5 6 7}"
N=${#G[@]}
w() { local k=$1 g=$2
  echo "$(date -Is) SAM 3 crop sample: worker $k/$N starting on GPU $g" >> $L
  until CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 ionice -c3 nice -n 10 $PY -u -W ignore sam3_crop_sample.py --worker $k/$N \
        >> logs/sam3_crop_$k.log 2>&1; do
    echo "$(date -Is) SAM 3 crop sample: worker $k/$N failed, retrying in 2 min" >> $L; sleep 120; done
  echo "$(date -Is) SAM 3 crop sample: worker $k/$N done" >> $L; }
for i in "${!G[@]}"; do w $i ${G[$i]} & done
wait
echo "$(date -Is) SAM 3 crop sample complete" >> $L
