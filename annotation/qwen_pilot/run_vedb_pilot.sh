#!/usr/bin/env bash
# After the BabyView full run logs "full run complete": VEDB frame-size pilot on 500 frames, three settings in
# parallel (GPU 0: 910 px long side, GPU 1: BabyView pixel count, GPU 2: native 2048 x 1536), and on GPU 3 a retry
# of the BabyView frames whose reply did not parse. Same model and settings as the full run otherwise.
cd "$(dirname "$0")"
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python
L=logs/run.log
R="--task detect_open_format --temperature 1.0 --mtp 3 --max-num-seqs 96 --batch 512"
until grep -q "full run complete" $L; do sleep 60; done
echo "$(date -Is) VEDB pilot: 500 frames x 3 sizes on GPUs 0-2; BabyView parse-failure retry on GPU 3" >> $L
run() { local g=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY -u run_qwen_pilot.py $R "$@" > logs/vedb_pilot_$tag.log 2>&1
  echo "$(date -Is) VEDB pilot $tag done (exit $?): $(grep 'frames/s' logs/vedb_pilot_$tag.log | tail -1)" >> $L; }
run 0 px910 --frames vedb_pilot500 &
run 1 pixbv --frames vedb_pilot500 --pixels bv &
run 2 native --frames vedb_pilot500 --max-px 0 --max-model-len 16384 &
( CUDA_VISIBLE_DEVICES=3 nice -n 10 $PY -u run_qwen_pilot.py $R --frames fullrun > logs/fullrun_retry.log 2>&1
  echo "$(date -Is) BabyView retry done (exit $?): $(grep 'frames |' logs/fullrun_retry.log | head -1)" >> $L ) &
wait
echo "$(date -Is) VEDB pilot all done" >> $L
