#!/bin/bash
# Pull packed imagenet-1k and lvis webdataset tar shards (+ LVIS annotations)
# from Oak to ccn2 node-local NVMe (/data2).
#
# Unlike the vedb pull, this lands on /data2 rather than /ccn2a: /ccn2a is
# shared NFS and already ~98% full lab-wide (only ~600GB free), while /data2
# is node-local with ~2.4TB free and faster for the sequential image reads
# detection/embedding work does. If this project later needs to share the
# raw images with other nodes/users, /ccn2a is the fallback -- but outputs
# (detection CSVs, embeddings) should go there regardless, not the raw data.
#
# Total size ~176GB (154.6GB imagenet tars + 20.2GB lvis tars + 1.3GB lvis
# annotations) -- small enough to pull in one shot, no stage/extract/cleanup
# batching needed like vedb.
#
# Usage:
#   ./pull_imagenet_lvis_from_oak.sh            # real pull
#   ./pull_imagenet_lvis_from_oak.sh --dry-run   # show what would transfer
set -u

DEST="/data2/j7yang"
REMOTE_BASE="oak-dtn:/oak/stanford/groups/mcfrank/bv_objects_related_data"
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"

BWLIMIT="${BWLIMIT:-40M}"        # validated safe on this cluster during the vedb pull
MIN_FREE_GB="${MIN_FREE_GB:-300}"

DRY=()
[[ "${1:-}" == "--dry-run" ]] && DRY=(-n)

mkdir -p "$LOGDIR" "${DEST}/imagenet-1k" "${DEST}/lvis/tars" "${DEST}/lvis/annotations"

log() { echo "$(date -Is) $*"; }
free_gb() { df --output=avail -B1G /data2 | tail -1 | tr -d ' '; }

avail=$(free_gb)
if (( avail < MIN_FREE_GB )); then
  log "only ${avail}G free on /data2 (< ${MIN_FREE_GB}G floor) -- aborting"
  exit 1
fi
log "starting pull, ${avail}G free on /data2"

pull() {
  local remote="$1" dest="$2" label="$3"
  ionice -c2 -n7 nice -n19 rsync -rlt --partial --info=progress2 "${DRY[@]}" \
    --bwlimit="$BWLIMIT" "$remote" "$dest"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "${label} rsync exited ${rc}"
    return "$rc"
  fi
  log "${label} done, $(free_gb)G free on /data2"
}

pull "${REMOTE_BASE}/imagenet-1k-tars/" "${DEST}/imagenet-1k/" "imagenet-1k-tars" || exit 1
pull "${REMOTE_BASE}/lvis-tars/" "${DEST}/lvis/tars/" "lvis-tars" || exit 1
pull "${REMOTE_BASE}/lvis/annotations/" "${DEST}/lvis/annotations/" "lvis-annotations" || exit 1

log "all done, $(free_gb)G free on /data2"
