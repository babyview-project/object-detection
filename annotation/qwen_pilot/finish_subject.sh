#!/usr/bin/env bash
# After the S00320001 shards (including the restarted shard 5) finish: final merge and a count.
set -u
cd "$(dirname "$0")"
until grep -q "shards finished" logs/subj_S00320001.out 2>/dev/null && ! pgrep -f "shard 5/8" >/dev/null; do sleep 120; done
/data/j7yang/venvs/qwen/bin/python run_qwen_pilot.py --task detect_open_format --frames subject:S00320001 --temperature 1.0 --merge-shards 8 2>&1 | grep merged | tee -a logs/run.log
/data/j7yang/venvs/qwen/bin/python run_qwen_pilot.py --task detect_open_format --frames subject:S00320001 --temperature 1.0 --dry-run 2>&1 | head -1 | tee -a logs/run.log
echo "$(date -Is) S00320001 complete" | tee -a logs/run.log
