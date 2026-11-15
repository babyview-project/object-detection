#!/usr/bin/env bash
# Gold pilot: two models x (name, verify on crops + retest pass; detect on whole frames).
# Run the 200-crop sanity pass first and look at it (see README), then:
#   setsid nohup bash run_all.sh > logs/run_all.log 2>&1 &
# Progress:  wc -l out/*.jsonl ; tail logs/run_all.log ; DONE marker at the end.
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE=${ENV_FILE:-$HOME/.secrets/hs-hs-langcog-gemini.env}
set -a; source "$ENV_FILE"; set +a
PY=${PY:-$HOME/.venvs/gemini/bin/python}
MODELS=${MODELS:-"gemini-2.5-flash-lite gemini-2.5-flash"}
mkdir -p logs

for m in $MODELS; do
  for t in name verify; do
    $PY run_gold_pilot.py --model "$m" --task "$t"
    $PY run_gold_pilot.py --model "$m" --task "$t" --pass 2 --limit-per-cat 8
  done
  $PY run_gold_pilot.py --model "$m" --task detect
done
$PY score_gold_pilot.py
echo "ALL DONE $(date)"
