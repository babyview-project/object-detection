#!/bin/bash
# Fine-grained 2D-format pass on every BabyView "drawing" / "photo" / "screen" box (qwen_picture_style.py --tag all2d),
# 8 shards, one per GPU. Shards resume; a shard that exits with an error is retried after 2 minutes.
# To stop cleanly: touch logs/picture_style.STOP (loops exit), then kill the python processes.
cd "$(dirname "$0")"
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
PY=/data/j7yang/venvs/qwen/bin/python; TAG=all2d; rm -f logs/picture_style.STOP
for k in 0 1 2 3 4 5 6 7; do
  ( until CUDA_VISIBLE_DEVICES=$k nice -n 10 $PY -u qwen_picture_style.py --tag $TAG --shard $k/8 --batch 1024 --threads 4 \
        >> logs/picture_style_${TAG}_$k.log 2>&1; do [ -e logs/picture_style.STOP ] && exit 1; sleep 120; done ) &
done
wait
[ -e logs/picture_style.STOP ] && { echo "stopped $(date)"; exit 1; }
/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python qwen_picture_style.py --tag $TAG --merge-shards 8
echo "PICTURE STYLE ALL DONE $(date)"
