#!/bin/bash
# Pull the remaining VEDB-1612 world-camera sessions from Oak (already fully
# downloaded there: 717 sessions, ~1.33 TiB, at
# oak-dtn:/oak/stanford/groups/mcfrank/bv_objects_related_data/vedb_1612_world/)
# in small batches, extract 1fps frames, then delete the staged raw mp4.
#
# /ccn2a is 98% full (~700 GB free, shared with the whole lab) and cannot hold
# the full raw set at once, so this never lets more than one batch of raw mp4s
# sit on disk: pull batch -> extract -> delete raw -> next batch.
#
# The 80-session subset already local at
#   /ccn2a/dataset/babyview/related_data/vedb_1612_world
# is untouched by this script (it is handled by run_extract_frames.sh); this
# script only pulls sessions not already present there.
#
# Usage:
#   ./pull_extract_cleanup_vedb_batch.sh              # one batch
#   ./pull_extract_cleanup_vedb_batch.sh --loop        # batches until nothing left
#   ./pull_extract_cleanup_vedb_batch.sh --dry-run     # show what would be pulled
#
# Run detached (ccn2 has no scheduler):
#   ssh ccn2-14 'cd REPO/analysis/vedb-1612 && setsid nohup bash pull_extract_cleanup_vedb_batch.sh --loop </dev/null >logs/pull_extract.log 2>&1 & echo ok'
set -u

REPO="/home/j7yang/babyview-projects/vss2026/object-detection"
REMOTE="oak-dtn:/oak/stanford/groups/mcfrank/bv_objects_related_data/vedb_1612_world"
PINNED_VIDEOS="/ccn2a/dataset/babyview/related_data/vedb_1612_world"   # 80-session subset, never touched
STAGING="/ccn2a/dataset/babyview/related_data/vedb_1612/raw_staging"  # transient, cleared after each session
FRAMES="/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps"
LOGDIR="${REPO}/analysis/vedb-1612/logs"

BATCH_SIZE="${BATCH_SIZE:-20}"                       # ~1.86 GB/session avg -> ~37 GB staged at a time
MIN_FREE_GB="${MIN_FREE_GB:-150}"                    # refuse to pull if /ccn2a free would drop below this
BWLIMIT="${BWLIMIT:-20M}"                            # cap rsync bandwidth, matches databrary/README.md convention
FFMPEG_WORKERS="${FFMPEG_WORKERS:-2}"                # matches run_extract_frames.sh default
PAUSE_BETWEEN_BATCHES="${PAUSE_BETWEEN_BATCHES:-60}" # seconds between batches when --loop

LOOP=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --loop) LOOP=1 ;;
    --dry-run) DRY_RUN=1 ;;
  esac
done

if [[ "$STAGING" == "$PINNED_VIDEOS" ]]; then
  echo "refusing to run: STAGING must not equal PINNED_VIDEOS" >&2
  exit 1
fi

mkdir -p "$STAGING" "$FRAMES" "$LOGDIR"

exec 200>"${LOGDIR}/.pull_extract.lock"
flock -n 200 || { echo "another instance is already running (lock held)"; exit 1; }

log() { echo "$(date -Is) $*"; }

free_gb() { df --output=avail -B1G /ccn2a | tail -1 | tr -d ' '; }

extract_one() {
  local video="$1" name out
  name="$(basename "$video" .mp4)"
  out="${FRAMES}/${name}"
  if [[ -f "${out}/.done" ]]; then return 0; fi
  rm -rf "$out"
  mkdir -p "$out"
  if ffmpeg -i "$video" -vf fps=1 -qscale:v 1 -video_track_timescale 1000 -frame_pts 1 \
      "${out}/%08d.jpg" -hide_banner -loglevel error; then
    touch "${out}/.done"
  else
    echo "FAIL ${name}" >&2
    return 1
  fi
}
export -f extract_one
export FRAMES

run_batch() {
  mapfile -t remote_names < <(rsync -n --list-only "${REMOTE}/" 2>/dev/null \
      | awk '{print $NF}' | grep -E '\.mp4$' | sed 's/\.mp4$//' | sort -u)
  mapfile -t pinned_names < <(find "$PINNED_VIDEOS" -maxdepth 1 -iname '*.mp4' -printf '%f\n' 2>/dev/null \
      | sed 's/\.mp4$//' | sort -u)
  mapfile -t done_names < <(find "$FRAMES" -mindepth 2 -maxdepth 2 -name '.done' -printf '%h\n' 2>/dev/null \
      | xargs -r -n1 basename | sort -u)
  mapfile -t exclude < <(printf '%s\n' "${pinned_names[@]:-}" "${done_names[@]:-}" | sort -u)
  mapfile -t todo < <(comm -23 <(printf '%s\n' "${remote_names[@]}") <(printf '%s\n' "${exclude[@]}"))

  if [[ ${#todo[@]} -eq 0 || ( ${#todo[@]} -eq 1 && -z "${todo[0]}" ) ]]; then
    log "nothing left to pull (${#remote_names[@]} on Oak, ${#pinned_names[@]} pinned locally, ${#done_names[@]} already extracted)"
    return 1
  fi

  local avail; avail=$(free_gb)
  if (( avail < MIN_FREE_GB )); then
    log "only ${avail}G free on /ccn2a (< ${MIN_FREE_GB}G floor) -- refusing to pull, stopping"
    return 1
  fi

  local batch=("${todo[@]:0:BATCH_SIZE}")
  log "batch of ${#batch[@]} sessions (${#todo[@]} remaining), ${avail}G free on /ccn2a"

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  would pull: %s\n' "${batch[@]}"
    return 1
  fi

  local filelist; filelist="$(mktemp)"
  printf '%s.mp4\n' "${batch[@]}" > "$filelist"

  ionice -c2 -n7 nice -n19 rsync -rlt --partial --info=progress2 \
    --bwlimit="$BWLIMIT" --files-from="$filelist" \
    "${REMOTE}/" "${STAGING}/"
  local rc=$?
  rm -f "$filelist"
  if [[ $rc -ne 0 ]]; then
    log "rsync exited ${rc}, stopping"
    return 1
  fi

  local staged=()
  for n in "${batch[@]}"; do
    [[ -f "${STAGING}/${n}.mp4" ]] && staged+=("${STAGING}/${n}.mp4")
  done
  log "extracting ${#staged[@]} sessions (ffmpeg workers=${FFMPEG_WORKERS})"
  printf '%s\n' "${staged[@]}" | xargs -n1 -P "$FFMPEG_WORKERS" bash -c 'extract_one "$1"' _

  local ok=0 fail=0
  for n in "${batch[@]}"; do
    if [[ -f "${FRAMES}/${n}/.done" ]]; then
      rm -f "${STAGING}/${n}.mp4"
      ok=$((ok + 1))
    else
      log "keeping staged mp4 for ${n} (extraction did not complete)"
      fail=$((fail + 1))
    fi
  done
  log "batch done: ${ok} extracted+cleaned, ${fail} failed/kept, $(free_gb)G free on /ccn2a"
  return 0
}

if [[ $LOOP -eq 1 ]]; then
  while run_batch; do
    log "pausing ${PAUSE_BETWEEN_BATCHES}s before next batch"
    sleep "$PAUSE_BETWEEN_BATCHES"
  done
  log "loop finished"
else
  run_batch
fi
