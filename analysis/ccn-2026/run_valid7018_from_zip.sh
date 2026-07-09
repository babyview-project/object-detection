#!/usr/bin/env bash
# Regenerate valid7018 metrics + abstract figures from the committed public bundle.
# Manuscript / Overleaf sources stay private (gitignored).
#
# Usage (from repo root):
#   bash analysis/ccn-2026/run_valid7018_from_zip.sh
#   bash analysis/ccn-2026/run_valid7018_from_zip.sh --verify
#
# Requires: ccn-valid7018 conda env (see environment.yml) or vislearnlabpy.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1 ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (try --verify)" >&2
      exit 2
      ;;
  esac
done

PY="${PYTHON:-python}"

echo "==> Metrics (clone-safe --from-zip)"
"$PY" analysis/ccn-2026/scripts/compute_valid7018_local_global.py --from-zip

echo "==> Figures → valid7018/figures/ + abstract_figures/"
"$PY" analysis/ccn-2026/scripts/generate_valid7018_paper_figures.py --from-zip

if [[ "$VERIFY" -eq 1 ]]; then
  echo "==> Verify against committed reference (temp dir; no overwrite)"
  "$PY" analysis/ccn-2026/scripts/verify_valid7018_reproduction.py --with-figures
fi

echo "Done."
