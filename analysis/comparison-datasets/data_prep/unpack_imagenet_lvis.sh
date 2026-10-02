#!/bin/bash
# Unpack imagenet-1k and lvis tar shards (pulled from Oak to /data2) into flat
# image directories. create_image_embeddings.py walks a directory for
# .jpg/.png files -- it doesn't read tar archives -- so this has to run before
# the embedding step.
#
# imagenet-1k: each class tar (n01440764.tar, ...) holds ~1300 flat JPEGs
# whose filenames already encode the class twice (e.g.
# n01440764_3509_n01440764.JPEG), so all class tars for a split safely extract
# into one shared train/ or val/ directory -- no per-class subfolders needed.
# lvis: train2017.tar and val2017.tar hold flat COCO-id-named JPEGs
# (000000147328.jpg) -- each goes to its own split directory so train/val
# don't mix.
set -u

BASE="/data2/j7yang"
LOGDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
WORKERS="${WORKERS:-8}"   # local NVMe, CPU-bound tar extraction -- fine to parallelize more than a network pull
mkdir -p "$LOGDIR"

log() { echo "$(date -Is) $*"; }
free_gb() { df --output=avail -B1G /data2 | tail -1 | tr -d ' '; }

extract_shard() {
  local tar="$1" dest="$2"
  local marker="${dest}/.$(basename "$tar").done"
  [[ -f "$marker" ]] && return 0
  mkdir -p "$dest"
  if tar xf "$tar" -C "$dest"; then
    touch "$marker"
  else
    echo "FAIL ${tar}" >&2
    return 1
  fi
}
export -f extract_shard

unpack_dir_of_tars() {
  local tar_dir="$1" dest="$2" label="$3"
  mkdir -p "$dest"
  mapfile -t tars < <(find "$tar_dir" -maxdepth 1 -iname '*.tar' | sort)
  log "${label}: ${#tars[@]} shards -> ${dest}"
  printf '%s\n' "${tars[@]}" | xargs -I{} -P "$WORKERS" bash -c 'extract_shard "$1" "$2"' _ {} "$dest"
  log "${label} done, $(free_gb)G free on /data2"
}

log "starting unpack, $(free_gb)G free on /data2"

unpack_dir_of_tars "${BASE}/imagenet-1k/train" "${BASE}/imagenet-1k/extracted/train" "imagenet-1k train"
unpack_dir_of_tars "${BASE}/imagenet-1k/val"   "${BASE}/imagenet-1k/extracted/val"   "imagenet-1k val"
extract_shard "${BASE}/lvis/tars/train2017.tar" "${BASE}/lvis/extracted/train2017"
log "lvis train2017 done, $(free_gb)G free on /data2"
extract_shard "${BASE}/lvis/tars/val2017.tar" "${BASE}/lvis/extracted/val2017"
log "lvis val2017 done, $(free_gb)G free on /data2"

log "all unpacking done, $(free_gb)G free on /data2"
