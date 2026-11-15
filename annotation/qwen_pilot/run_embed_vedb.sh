#!/usr/bin/env bash
# CLIP + DINOv3 embeddings of every VEDB Qwen box (embed_full_run.py --dataset vedb), one worker per GPU in $GPUS
# (default: the GPUs free after the VEDB run, 0 2 3 5 6 7). Resumable: rerunning skips finished sessions.
# Then the box index. Store: /data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv
cd "$(dirname "$0")"
unset HF_HOME                       # the embed models live in the lab cache the script sets, not the Qwen cache
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
read -ra G <<< "${GPUS:-0 2 3 5 6 7}"
N=${#G[@]}
w() { local k=$1 g=$2
  echo "$(date -Is) VEDB embeddings: worker $k/$N starting on GPU $g" >> $L
  until CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 nice -n 10 $PY -u embed_full_run.py --dataset vedb --worker $k/$N \
        >> logs/embed_vedb_$k.log 2>&1; do
    echo "$(date -Is) VEDB embeddings: worker $k/$N failed, retrying in 2 min" >> $L; sleep 120; done
  echo "$(date -Is) VEDB embeddings: worker $k/$N done" >> $L; }
for i in "${!G[@]}"; do w $i ${G[$i]} & done
wait
nice -n 19 $PY embed_full_run.py --dataset vedb --index >> $L 2>&1
echo "$(date -Is) VEDB embeddings complete" >> $L
