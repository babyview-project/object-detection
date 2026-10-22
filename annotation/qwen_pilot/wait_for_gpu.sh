#!/usr/bin/env bash
# Wait until a GPU on this node has >= MIN_FREE_MIB free on CHECKS consecutive checks
# (INTERVAL s apart), then run the Qwen pilot on it. Gives up after MAX_HOURS.
#   bash wait_for_gpu.sh            # logs/wait.log
set -u
cd "$(dirname "$0")"
mkdir -p logs
MIN_FREE_MIB=${MIN_FREE_MIB:-40000}; CHECKS=${CHECKS:-3}; INTERVAL=${INTERVAL:-120}; MAX_HOURS=${MAX_HOURS:-24}
declare -A streak
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
echo "$(date -Is) waiting for a GPU with >= $MIN_FREE_MIB MiB free on $CHECKS checks ${INTERVAL}s apart" | tee -a logs/wait.log
while (( $(date +%s) < deadline )); do
  while IFS=', ' read -r idx free; do
    if (( free >= MIN_FREE_MIB )); then streak[$idx]=$(( ${streak[$idx]:-0} + 1 )); else streak[$idx]=0; fi
    if (( ${streak[$idx]} >= CHECKS )); then
      echo "$(date -Is) GPU $idx free ($free MiB) on $CHECKS checks: starting the pilot" | tee -a logs/wait.log
      GPUS=$idx bash run_qwen_pilot.sh
      echo "$(date -Is) pilot exited with $?" | tee -a logs/wait.log
      exit 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  sleep "$INTERVAL"
done
echo "$(date -Is) gave up after $MAX_HOURS h" | tee -a logs/wait.log
