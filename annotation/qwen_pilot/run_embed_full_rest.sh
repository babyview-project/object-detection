#!/usr/bin/env bash
# Replaces the waiting half of run_embed_full.sh: workers 0, 1, 3 start now; worker 2 once the native-size VEDB pilot
# on GPU 2 is done; the index once all 8 workers have logged "done".
cd "$(dirname "$0")"
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
w() { local k=$1
  [ "$k" = 2 ] && until grep -q "VEDB pilot native done" $L; do sleep 60; done
  echo "$(date -Is) embeddings: worker $k/8 starting on GPU $k" >> $L
  until CUDA_VISIBLE_DEVICES=$k OMP_NUM_THREADS=2 nice -n 10 $PY -u embed_full_run.py --worker $k/8 >> logs/embed_full_$k.log 2>&1; do
    echo "$(date -Is) embeddings: worker $k/8 failed, retrying in 2 min" >> $L; sleep 120; done
  echo "$(date -Is) embeddings: worker $k/8 done" >> $L; }
for k in 0 1 2 3; do w $k & done
until [ "$(grep -cE 'embeddings: worker [0-7]/8 done' $L)" -ge 8 ]; do sleep 60; done
nice -n 19 $PY embed_full_run.py --index >> $L 2>&1
echo "$(date -Is) embeddings complete" >> $L
