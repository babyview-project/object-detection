#!/usr/bin/env bash
# S00320001 wrap-up on ONE idle GPU (util < 10% and >= 40 GB free on 3 checks a minute apart):
# 1) the frames still missing from the subject run, 2) box table, 3) CLIP + DINOv3 crop embeddings,
# 4) re-execute qwen_subject_analysis.ipynb into out/private/. Each step is logged in logs/run.log.
set -u
cd "$(dirname "$0")"
FROM=${FROM:-frames}                          # resume: FROM=boxes skips the subject frames, FROM=embed also the box table
LOG=logs/run.log
export HF_HOME=/data/j7yang/models/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export QWEN_MODEL=$(ls -d /data/j7yang/models/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
QPY=/data/j7yang/venvs/qwen/bin/python
KPY=/data/j7yang/anaconda3/envs/vislearnlabpy/bin/python
idle_gpu() {
  local G="" g k
  for k in 1 2 3; do
    g=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits |
        awk -F', ' -v need="$1" '$2 < 10 && $3 >= need {print $1; exit}')
    if [ -n "$g" ] && { [ -z "$G" ] || [ "$G" = "$g" ]; }; then G=$g; else return 1; fi
    [ $k -lt 3 ] && sleep 60
  done
  echo "$G"
}
if [ -n "${GPU:-}" ]; then G=$GPU          # GPU=7: use this GPU now (shares it with whatever is running there)
else until G=$(idle_gpu 40000); do sleep 120; done; fi
if [ "$FROM" = frames ]; then
  echo "$(date -Is) wrap-up on GPU $G: remaining subject frames" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$G nice -n 10 $QPY -u run_qwen_pilot.py --task detect_open_format --frames subject:S00320001 \
    --temperature 1.0 --mtp 3 --max-num-seqs 64 --batch 512 > logs/subj_finish.log 2>&1
  rc=$?
  echo "$(date -Is) remaining frames: exit $rc; $(wc -l < out/runs/qwen3.8-27b-fp8__detect_open_format__p1_subj_S00320001_t1_pp0.jsonl) records" >> "$LOG"
else
  echo "$(date -Is) wrap-up resumed on GPU $G from the box table" >> "$LOG"
fi
unset HF_HOME      # Qwen's cache above; CLIP text (domains) and DINOv3 live in the default / 2025.2 caches
if [ "$FROM" != embed ]; then
  OMP_NUM_THREADS=2 nice -n 19 $KPY embed_qwen_crops.py --prepare > logs/embed_prepare.log 2>&1 ||
    { echo "$(date -Is) box table FAILED (logs/embed_prepare.log)" >> "$LOG"; exit 1; }
fi
echo "$(date -Is) embedding Qwen crops on GPU $G" >> "$LOG"
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 $QPY embed_qwen_crops.py --device cuda:0 \
  > logs/embed_qwen_crops.log 2>&1 || { echo "$(date -Is) embedding FAILED (logs/embed_qwen_crops.log)" >> "$LOG"; exit 1; }
echo "$(date -Is) embedding done" >> "$LOG"
NB=../../analysis/qwen-open-vocab   # the subject notebook moved there (2026-10-27)
( cd $NB && python build_subject_nb.py >/dev/null &&
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 jupyter nbconvert --to notebook --execute qwen_subject_analysis.ipynb \
  --ExecutePreprocessor.kernel_name=vislearnlabpy --ExecutePreprocessor.timeout=7200 \
  --output-dir out/private --output qwen_subject_analysis_executed.ipynb ) > logs/subject_nb.log 2>&1 &&
  echo "$(date -Is) subject notebook executed" >> "$LOG" || echo "$(date -Is) subject notebook FAILED (logs/subject_nb.log)" >> "$LOG"
