#!/usr/bin/env bash
# CLIP + DINOv3 embeddings of every full-run Qwen box (embed_full_run.py), 8 workers (one per GPU):
# workers 4-7 start now on GPUs 4-7; workers 0-3 start on GPUs 0-3 once the VEDB pilot is done. Then the index.
cd "$(dirname "$0")"
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
w() { local k=$1
  case " 0 1 2 3 " in *" $k "*) until grep -q "VEDB pilot all done" $L; do sleep 60; done ;; esac
  echo "$(date -Is) embeddings: worker $k/8 starting on GPU $k" >> $L
  until CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=2 nice -n 10 $PY -u embed_full_run.py --worker $k/8 >> logs/embed_full_$k.log 2>&1; do
    echo "$(date -Is) embeddings: worker $k/8 failed, retrying in 2 min" >> $L; sleep 120; done
  echo "$(date -Is) embeddings: worker $k/8 done" >> $L; }
for k in 0 1 2 3 4 5 6 7; do w $k & done
wait
nice -n 19 $PY embed_full_run.py --index >> $L 2>&1
echo "$(date -Is) embeddings complete" >> $L
