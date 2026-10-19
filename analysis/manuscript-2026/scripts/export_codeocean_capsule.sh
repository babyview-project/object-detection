#!/usr/bin/env bash
# Build a slim CodeOcean / public-repo tree for the BabyView Objects manuscript.
#
# Default profile is manuscript-only: analysis notebooks 01–05/08–10, analysis
# scripts, anonymized shared tables, YOLOE↔VQA convergence summaries/figures,
# and z-scored category embeddings. No Stage 0 GPU/tmux tooling, no real
# subject-ID result trees, and no video-level VQA dumps (video_id rows).
#
# Usage (from monorepo root, or any cwd):
#   bash analysis/manuscript-2026/scripts/export_codeocean_capsule.sh
#   bash analysis/manuscript-2026/scripts/export_codeocean_capsule.sh --archive
#   bash analysis/manuscript-2026/scripts/export_codeocean_capsule.sh --dry-run
#
# Prefer writing under $HOME (not /tmp on shared clusters). Default OUT is
#   ../babyview-objects-manuscript-capsule   (sibling of the monorepo)
#
# See analysis/manuscript-2026/CODEOCEAN_EXPORT.md for include/exclude policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$MS_DIR/../.." && pwd)"

OUT=""
PROFILE="manuscript"   # manuscript | minimal
INCLUDE_FIGURES=0
INCLUDE_EXEMPLAR=1     # z-scored tables into exemplar_set_embeddings/
INCLUDE_LOCAL_RESULTS=0
INCLUDE_VQA_SUMMARY=1   # manuscript YOLOE↔VQA convergence (figures + summary CSVs)
INCLUDE_VQA_RAW=0       # full unconstrained_objects.csv (~19M, has video_id)
INCLUDE_STAGE0=0
INCLUDE_MANUSCRIPT_PDF=0
INCLUDE_AUTOFILL=0
CLEAR_NOTEBOOKS=1
INIT_GIT=0
ARCHIVE=0
DRY_RUN=0

usage() {
  sed -n '2,16p' "$0"
  cat <<'EOF'

Options:
  --out DIR                 Output directory (default: sibling of monorepo, or
                            $CODEOCEAN_EXPORT_OUT). Prefer a path under $HOME.
  --minimal                 Tier A shared tables + code only (no exemplar tables)
  --include-figures         Copy main/supplement figures/ trees (~118M)
  --include-local-results   Copy main/supplement results/ with privacy filters
                            (excludes subject_* folders, *all_subjects*, *.npz)
  --include-vqa-raw         Also ship video-level unconstrained_objects.csv (~19M)
  --no-vqa                  Skip YOLOE↔VQA summary tables/figures (not recommended)
  --include-stage0          Also ship Stage 0 notebooks 06–07 + GPU/tmux helpers
  --include-manuscript-pdf  Copy BV_Objects_Manuscript_*.pdf (~12M)
  --include-autofill        Copy results_preprint_numbers_*.{txt,csv}
  --keep-notebook-outputs   Do not strip Jupyter outputs (default: clear)
  --no-exemplar             Skip exemplar_set_embeddings/
  --init-git                git init + initial commit in OUT
  --archive                 Also write OUT.tar.gz
  --dry-run                 Print plan and estimated sizes; do not write
  -h, --help                Show this help

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --minimal) PROFILE="minimal"; INCLUDE_EXEMPLAR=0; shift ;;
    --include-figures) INCLUDE_FIGURES=1; shift ;;
    --include-local-results) INCLUDE_LOCAL_RESULTS=1; shift ;;
    --include-vqa-raw) INCLUDE_VQA_RAW=1; shift ;;
    --no-vqa) INCLUDE_VQA_SUMMARY=0; shift ;;
    --include-stage0) INCLUDE_STAGE0=1; shift ;;
    --include-manuscript-pdf) INCLUDE_MANUSCRIPT_PDF=1; shift ;;
    --include-autofill) INCLUDE_AUTOFILL=1; shift ;;
    --keep-notebook-outputs) CLEAR_NOTEBOOKS=0; shift ;;
    --no-exemplar) INCLUDE_EXEMPLAR=0; shift ;;
    --init-git) INIT_GIT=1; shift ;;
    --archive) ARCHIVE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUT" ]]; then
  OUT="${CODEOCEAN_EXPORT_OUT:-$(dirname "$REPO_ROOT")/babyview-objects-manuscript-capsule}"
fi
# Resolve OUT parent (may need write outside sandbox)
OUT_PARENT="$(dirname "$OUT")"
mkdir -p "$OUT_PARENT"
OUT="$(cd "$OUT_PARENT" && pwd)/$(basename "$OUT")"

# Soft warning if targeting world-readable temp areas on a shared machine
case "$OUT" in
  /tmp/*|/scratch/tmp/*|/var/tmp/*)
    echo "WARN: OUT is under a shared temp path ($OUT)." >&2
    echo "      Prefer a directory under \$HOME on shared clusters." >&2
    ;;
esac

need() {
  local p="$1"
  if [[ ! -e "$REPO_ROOT/$p" ]]; then
    echo "Missing required path: $p" >&2
    exit 1
  fi
}

need "analysis/manuscript-2026/REPRODUCTION.md"
need "data/shared_data_manuscript_2026/MANIFEST.json"

# Manuscript analysis notebooks (exclude Stage 0 builders 06–07 unless requested)
ANALYSIS_NOTEBOOKS=(
  01_long_tailed_distribution.ipynb
  02_category-wise_cosine_sim.ipynb
  03_bv_things_rdm_comparison.ipynb
  04_individual_rdms.ipynb
  05_within_between_cdi_cluster_correlation.ipynb
  08_top8_kid_category_cosine_similarity.ipynb
  09_top8_local_global_variability_valid85.ipynb
  10_animal_depiction_label_proportions.ipynb
)

# Scripts needed to rerun paper analyses from category tables (not Stage 0 / publish tooling)
ANALYSIS_SCRIPTS=(
  __init__.py
  _bootstrap.py
  bv_things_cdi_shuffle_inference.py
  compute_binary_template_correlations.py
  compare_dinov3_vs_babydinov3.py
  plot_things_dinov3_vs_babydinov3_umap.py
  top8_within_between_vs_things.py
  clip_threshold_sensitivity.py
  packed_crop_embeddings.py
  frame_prevalence.py
  run_long_tail_frame_prevalence.py
  README.md
)

# ----- plan -----
echo "==> CodeOcean capsule export"
echo "    repo:     $REPO_ROOT"
echo "    out:      $OUT"
echo "    profile:  $PROFILE"
echo "    figures:  $INCLUDE_FIGURES  exemplar: $INCLUDE_EXEMPLAR  local_results: $INCLUDE_LOCAL_RESULTS"
echo "    vqa_sum:  $INCLUDE_VQA_SUMMARY  vqa_raw: $INCLUDE_VQA_RAW  stage0: $INCLUDE_STAGE0  clear_nb: $CLEAR_NOTEBOOKS"

estimate_tree() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    echo "—"
  fi
}

echo
echo "Source sizes:"
printf "  %-55s %s\n" "shared_data_manuscript_2026" "$(estimate_tree "$REPO_ROOT/data/shared_data_manuscript_2026")"
printf "  %-55s %s\n" "shared .../vqa_detections" "$(estimate_tree "$REPO_ROOT/data/shared_data_manuscript_2026/vqa_detections")"
printf "  %-55s %s\n" "  unconstrained_objects.csv (raw; opt-in)" "$(estimate_tree "$REPO_ROOT/data/shared_data_manuscript_2026/vqa_detections/unconstrained_objects/unconstrained_objects.csv")"
printf "  %-55s %s\n" "shared .../embeddings" "$(estimate_tree "$REPO_ROOT/data/shared_data_manuscript_2026/embeddings")"
printf "  %-55s %s\n" "exemplar_set_embeddings (local)" "$(estimate_tree "$MS_DIR/exemplar_set_embeddings")"
printf "  %-55s %s\n" "main_results/.../results" "$(estimate_tree "$MS_DIR/main_results_valid129s_04302026/results")"
printf "  %-55s %s\n" "not_in_manuscript (EXCLUDED)" "$(estimate_tree "$MS_DIR/not_in_manuscript")"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "Dry run only — no files written."
  echo "Would include notebooks: ${ANALYSIS_NOTEBOOKS[*]}"
  [[ "$INCLUDE_STAGE0" -eq 1 ]] && echo "Would also include Stage 0: 06, 07 + GPU helpers"
  echo "Re-run without --dry-run to create: $OUT"
  exit 0
fi

# ----- clean destination -----
if [[ -e "$OUT" ]]; then
  echo
  echo "==> Removing existing $OUT"
  rm -rf "$OUT"
fi
mkdir -p "$OUT"

rsync_copy() {
  local src="$1" dest="$2"
  shift 2
  mkdir -p "$(dirname "$dest")"
  rsync -a --omit-dir-times "$@" "$src" "$dest"
}

# ----- data (Tier A) -----
echo
echo "==> data/shared_data_manuscript_2026/"
# Always exclude video-level VQA dump unless --include-vqa-raw (privacy + size).
SHARED_EXCLUDES=(
  --exclude '__pycache__/'
  --exclude 'vqa_detections/unconstrained_objects/unconstrained_objects.csv'
  --exclude 'vqa_detections/unconstrained_objects/all_objects_to_check.csv'
)
if [[ "$INCLUDE_VQA_SUMMARY" -eq 0 ]]; then
  SHARED_EXCLUDES+=(--exclude 'vqa_detections/')
fi
if [[ "$INCLUDE_VQA_RAW" -eq 1 ]]; then
  # Drop the unconstrained_objects.csv exclude (keep all_objects_to_check excluded).
  SHARED_EXCLUDES=(
    --exclude '__pycache__/'
    --exclude 'vqa_detections/unconstrained_objects/all_objects_to_check.csv'
  )
  if [[ "$INCLUDE_VQA_SUMMARY" -eq 0 ]]; then
    echo "    WARN: --include-vqa-raw without VQA summary; enabling summary tree" >&2
    INCLUDE_VQA_SUMMARY=1
  fi
fi
rsync_copy \
  "$REPO_ROOT/data/shared_data_manuscript_2026/" \
  "$OUT/data/shared_data_manuscript_2026/" \
  "${SHARED_EXCLUDES[@]}"

# Minimal root stubs still referenced via DATA_DIR / manuscript_config
DATA_STUBS=(
  included_categories_valid129.txt
  included_categories_valid85.txt
  cdi_words.csv
  MCDI_items_with_AoA.csv
  average_precision_for_included_categories.csv
  long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv
)
for f in "${DATA_STUBS[@]}"; do
  if [[ -f "$REPO_ROOT/data/$f" ]]; then
    rsync_copy "$REPO_ROOT/data/$f" "$OUT/data/$f"
  fi
done

# YOLOE↔VQA convergence stats (manuscript SI; currently live under analysis/ccn-2026/)
if [[ "$INCLUDE_VQA_SUMMARY" -eq 1 ]]; then
  echo "==> YOLOE↔VQA correlation tables"
  VQA_OUT="$OUT/data/shared_data_manuscript_2026/vqa_detections"
  mkdir -p "$VQA_OUT"
  for f in \
    "$REPO_ROOT/analysis/ccn-2026/valid7018/yoloe_vqa_correlation.json" \
    "$REPO_ROOT/analysis/ccn-2026/valid7018/yoloe_vqa_correlation_by_category.csv"
  do
    [[ -f "$f" ]] && rsync_copy "$f" "$VQA_OUT/$(basename "$f")"
  done
  # SI figure copies if present under manuscript tree
  if [[ -d "$MS_DIR/figures/si" ]]; then
    mkdir -p "$OUT/analysis/manuscript-2026/figures/si"
    shopt -s nullglob
    for f in "$MS_DIR"/figures/si/vqa_comparison.*; do
      rsync_copy "$f" "$OUT/analysis/manuscript-2026/figures/si/$(basename "$f")"
    done
    shopt -u nullglob
  fi
fi

# ----- manuscript analysis tree -----
echo "==> analysis/manuscript-2026/ (manuscript-only)"
MS_OUT="$OUT/analysis/manuscript-2026"
mkdir -p "$MS_OUT/scripts"

# Docs / config
for f in \
  README.md REPRODUCTION.md SCRIPTS.md DATA_AVAILABILITY.md \
  requirements-manuscript.txt manuscript_config.py
do
  [[ -e "$MS_DIR/$f" ]] && rsync_copy "$MS_DIR/$f" "$MS_OUT/$f"
done

if [[ "$INCLUDE_STAGE0" -eq 1 ]]; then
  for f in \
    00_build_exemplar_embeddings.md paths.example.env \
    run_exemplar_embedding_stage.sh run_06_zscore_tmux.sh run_babydinov3_things_embed_tmux.sh
  do
    [[ -e "$MS_DIR/$f" ]] && rsync_copy "$MS_DIR/$f" "$MS_OUT/$f"
  done
fi

if [[ "$INCLUDE_AUTOFILL" -eq 1 ]]; then
  for f in results_preprint_numbers_autofill.txt results_preprint_numbers_table.csv results_preprint_paragraphs_autofill.txt; do
    [[ -e "$MS_DIR/$f" ]] && rsync_copy "$MS_DIR/$f" "$MS_OUT/$f"
  done
fi

if [[ "$INCLUDE_MANUSCRIPT_PDF" -eq 1 ]]; then
  shopt -s nullglob
  for f in "$MS_DIR"/BV_Objects_Manuscript_*.pdf; do
    rsync_copy "$f" "$MS_OUT/$(basename "$f")"
  done
  shopt -u nullglob
fi

# Notebooks (cleared outputs by default)
echo "==> notebooks"
copy_notebook() {
  local src="$1"
  local dest="$MS_OUT/$(basename "$src")"
  if [[ "$CLEAR_NOTEBOOKS" -eq 1 ]]; then
    python3 - "$src" "$dest" <<'PY'
import json, sys
src, dest = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as f:
    nb = json.load(f)
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
nb["metadata"].pop("widgets", None)
with open(dest, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
PY
  else
    rsync_copy "$src" "$dest"
  fi
}

for nb in "${ANALYSIS_NOTEBOOKS[@]}"; do
  if [[ -f "$MS_DIR/$nb" ]]; then
    copy_notebook "$MS_DIR/$nb"
  else
    echo "    WARN: missing notebook $nb" >&2
  fi
done

if [[ "$INCLUDE_STAGE0" -eq 1 ]]; then
  for nb in 06_exemplar_set_zscore_embeddings.ipynb 07_things_exemplar_set_zscore_embeddings.ipynb; do
    [[ -f "$MS_DIR/$nb" ]] && copy_notebook "$MS_DIR/$nb"
  done
fi

# Whitelisted analysis scripts only
echo "==> scripts/ (analysis whitelist)"
for s in "${ANALYSIS_SCRIPTS[@]}"; do
  if [[ -e "$MS_DIR/scripts/$s" ]]; then
    rsync_copy "$MS_DIR/scripts/$s" "$MS_OUT/scripts/$s"
  fi
done

if [[ "$INCLUDE_STAGE0" -eq 1 ]]; then
  for s in \
    check_exemplar_stage.py \
    exemplar_set_zscore_embeddings.py \
    things_exemplar_set_zscore_babydinov3.py \
    create_babydinov3_things_embeddings.py \
    create_babydinov3_crop_embeddings.py
  do
    [[ -e "$MS_DIR/scripts/$s" ]] && rsync_copy "$MS_DIR/scripts/$s" "$MS_OUT/scripts/$s"
  done
fi

# Optional local results (privacy-filtered). Default: use shared Tier A tables instead.
if [[ "$INCLUDE_LOCAL_RESULTS" -eq 1 ]]; then
  echo "==> local results/ (privacy-filtered)"
  for pair in \
    "main_results_valid129s_04302026/results" \
    "supplemental_results_valid85cats_04302026/results"
  do
    if [[ -d "$MS_DIR/$pair" ]]; then
      rsync_copy "$MS_DIR/$pair/" "$MS_OUT/$pair/" \
        --exclude '__pycache__/' \
        --exclude 'subject_*/' \
        --exclude '*subject_*' \
        --exclude '*all_subjects*' \
        --exclude '*.npz' \
        --exclude 'top8_vs_things_within_between_*/'
    fi
  done
fi

if [[ "$INCLUDE_FIGURES" -eq 1 ]]; then
  echo "==> figures/"
  for pair in \
    "main_results_valid129s_04302026/figures" \
    "supplemental_results_valid85cats_04302026/figures"
  do
    if [[ -d "$MS_DIR/$pair" ]]; then
      rsync_copy "$MS_DIR/$pair/" "$MS_OUT/$pair/"
    fi
  done
fi

# Tier B: z-scored category tables at the path notebooks expect
if [[ "$INCLUDE_EXEMPLAR" -eq 1 ]]; then
  echo "==> exemplar_set_embeddings/ (z-score only, from shared public embeddings)"
  SHARED_EMB="$REPO_ROOT/data/shared_data_manuscript_2026/embeddings"
  if [[ -d "$SHARED_EMB" ]]; then
    mkdir -p "$MS_OUT/exemplar_set_embeddings"
    rsync_copy "$SHARED_EMB/" "$MS_OUT/exemplar_set_embeddings/" \
      --exclude '__pycache__/' \
      --exclude '*_raw_*'
  elif [[ -d "$MS_DIR/exemplar_set_embeddings" ]]; then
    echo "    WARN: shared embeddings missing; using local z-score tables only" >&2
    mkdir -p "$MS_OUT/exemplar_set_embeddings"
    rsync_copy "$MS_DIR/exemplar_set_embeddings/" "$MS_OUT/exemplar_set_embeddings/" \
      --exclude '__pycache__/' \
      --exclude '*_raw_*'
  else
    echo "    WARN: no embedding tables found; notebooks 02–05 will not regenerate" >&2
  fi
fi

# ----- capsule README -----
cat > "$OUT/README.md" <<EOF
# BabyView Objects — manuscript capsule

Slim, **manuscript-only** export for CodeOcean / public redistribution.
This is **not** the full internal monorepo.

Generated from: \`object-detection\` at $(date -u +%Y-%m-%dT%H:%MZ)  
Profile: **${PROFILE}** (figures=${INCLUDE_FIGURES}, exemplar=${INCLUDE_EXEMPLAR}, vqa_summary=${INCLUDE_VQA_SUMMARY}, stage0=${INCLUDE_STAGE0})

## Layout

\`\`\`text
analysis/manuscript-2026/          # notebooks 01–05, 08–10 + analysis scripts
analysis/manuscript-2026/exemplar_set_embeddings/   # z-scored category tables (Tier B)
data/shared_data_manuscript_2026/  # anonymized Tier A tables (MANIFEST.json)
data/shared_data_manuscript_2026/vqa_detections/    # YOLOE↔VQA SI summaries + figures
data/included_categories_*.txt, cdi_words.csv, …
\`\`\`

## Quick start

\`\`\`bash
pip install -r analysis/manuscript-2026/requirements-manuscript.txt
\`\`\`

1. Verify published aggregates under \`data/shared_data_manuscript_2026/results_valid129/\`.
2. YOLOE↔VQA frame-prevalence correlation: \`vqa_detections/yoloe_vqa_correlation.json\`
   and \`vqa_detections/figures/vqa_comparison_frame_prevalence.*\`.
3. Read \`analysis/manuscript-2026/REPRODUCTION.md\` for notebook order.
4. (Optional) Rerun notebooks **01–05 / 08–10** with cwd = \`analysis/manuscript-2026/\`.

## Intentionally not included

- Stage 0 GPU/tmux builders (notebooks 06–07, \`run_*_tmux.sh\`, crop embedding scripts)
- \`not_in_manuscript/\`, other conference pilots, YOLOE / raw video / per-crop \`.npy\`
- Local \`main_results/\` trees with real subject IDs (use anonymized \`shared_data_…/top8_valid85/\`)
- Video-level VQA dump \`unconstrained_objects.csv\` (has \`video_id\`; use \`--include-vqa-raw\` only if needed)

Full Stage 0 rebuild from crops requires BabyView data access (documented in the monorepo).

## CodeOcean

- Upload this directory as the capsule root (\`analysis/\` and \`data/\` as siblings).
- Point the reproducible run at Tier A verification and/or selected notebooks — not GPU re-embedding.

## License / data use

Only anonymized tables under \`data/shared_data_manuscript_2026/\` are intended for
public redistribution. Respect BabyView data-use agreements for any restricted assets.
EOF

# ----- hygiene -----
find "$OUT" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -type f -name '*.pyc' -delete 2>/dev/null || true
rm -f "$MS_OUT/paths.local.env" 2>/dev/null || true

# ----- privacy scrub (capsule only; does not modify monorepo) -----
echo "==> privacy scrub (subject IDs + absolute cluster/home paths)"
python3 - "$OUT" <<'PY'
"""Redact identifiable / internal-path leaks from the exported capsule."""
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

# Known excluded BabyView subject (documented in monorepo; do not ship raw ID).
EXCLUDED_REAL_ID = "00270001"
EXCLUDED_TOKEN = "<excluded-subject>"

TEXT_SUFFIXES = {
    ".py", ".ipynb", ".md", ".txt", ".csv", ".json", ".tsv", ".yml", ".yaml",
    ".env", ".sh", ".r", ".R", ".qmd", ".tex",
}

# Absolute path roots seen in this project (researcher home + cluster data roots).
PATH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Full home project prefixes → capsule-relative style
    (
        re.compile(
            r"/home/[^/\s\"']+/babyview-projects/vss2026/object-detection/"
            r"(?:analysis/(?:preprint|manuscript)-2026/)?"
        ),
        "",
    ),
    (re.compile(r"/home/[^/\s\"']+/babyview-projects/vss2026/object-detection"), "<REPO_ROOT>"),
    (re.compile(r"/home/[^/\s\"']+"), "<HOME>"),
    (
        re.compile(r"/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings"),
        "<BV_EMBEDDINGS_BASE>",
    ),
    (
        re.compile(r"/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class"),
        "<BV_CROPPED_DIR>",
    ),
    (re.compile(r"/data2/dataset/babyview/[^\s\"']*"), "<BV_DATA>/"),
    (re.compile(r"/data2/[^\s\"']*"), "<CLUSTER_DATA>/"),
    (re.compile(r"/ccn2/dataset/babyview/[^\s\"']*"), "<BV_DATA>/"),
    (re.compile(r"/ccn2/[^\s\"']*"), "<CLUSTER_DATA>/"),
]

# Real 8-digit BabyView subject IDs (00......). Keep participant_0N untouched.
SUBJECT_ID_RE = re.compile(r"\b00[0-9]{6}\b")
# Filter-list filename fragment that embeds the excluded ID.
FILTER_NAME_RE = re.compile(
    r"clip_image_embeddings_filtered-by-clip-([0-9.]+)_exclude-people_exclude-subject-"
    + re.escape(EXCLUDED_REAL_ID)
    + r"\.txt"
)

def scrub_text(text: str) -> str:
    # Filter-list filenames embed the excluded ID — rewrite before generic ID scrub.
    text = FILTER_NAME_RE.sub(
        r"clip_image_embeddings_filtered-by-clip-\1_exclude-people_exclude-subject-"
        + EXCLUDED_TOKEN
        + ".txt",
        text,
    )
    text = text.replace(EXCLUDED_REAL_ID, EXCLUDED_TOKEN)
    for pat, repl in PATH_PATTERNS:
        text = pat.sub(repl, text)
    # Any remaining 00xxxxxx subject-shaped IDs → opaque token
    text = SUBJECT_ID_RE.sub("<subject-id>", text)
    return text

n_files = 0
n_changed = 0
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
        "README", "MANIFEST.json", "requirements-manuscript.txt",
    }:
        # Still allow extensionless small text? skip binaries.
        continue
    if path.suffix.lower() in {".png", ".pdf", ".npz", ".npy", ".jpg", ".jpeg", ".gif"}:
        continue
    try:
        raw = path.read_bytes()
    except OSError:
        continue
    # Skip likely-binary
    if b"\0" in raw[:4096]:
        continue
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            continue
    scrubbed = scrub_text(text)
    n_files += 1
    if scrubbed != text:
        path.write_text(scrubbed, encoding="utf-8")
        n_changed += 1

print(f"    scanned {n_files} text files; scrubbed {n_changed}")
PY

chmod -R go-rwx "$OUT" 2>/dev/null || true

# Sanity: no obvious real subject_* result paths
if find "$OUT" -name 'subject_*' 2>/dev/null | grep -q .; then
  echo "ERROR: subject_* paths leaked into capsule:" >&2
  find "$OUT" -name 'subject_*' 2>/dev/null | head >&2
  exit 1
fi

# Sanity: no remaining real BabyView IDs or home/cluster absolute paths
LEAK_REPORT="$(
  python3 - "$OUT" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
pats = [
    (re.compile(r"\b00[0-9]{6}\b"), "subject-id"),
    (re.compile(r"/home/[A-Za-z0-9._-]+"), "home-path"),
    (re.compile(r"/data2/"), "data2-path"),
    (re.compile(r"/ccn2/"), "ccn2-path"),
]
# Allow false positives in floats? 00xxxxxx is specific enough.
hits = []
for path in root.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix.lower() in {".png", ".pdf", ".npz", ".npy"}:
        continue
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    for rx, label in pats:
        if rx.search(text):
            # show first match context
            m = rx.search(text)
            hits.append(f"{label}: {path.relative_to(root)} :: ...{text[max(0,m.start()-20):m.end()+40]!r}...")
            break
if hits:
    print("\n".join(hits[:30]))
PY
)"
if [[ -n "$LEAK_REPORT" ]]; then
  echo "ERROR: privacy scrub incomplete — remaining leaks:" >&2
  echo "$LEAK_REPORT" >&2
  exit 1
fi
echo "    privacy checks passed"
# ----- optional git -----
if [[ "$INIT_GIT" -eq 1 ]]; then
  echo "==> git init"
  (
    cd "$OUT"
    printf '%s\n' '__pycache__/' '*.pyc' 'paths.local.env' '.ipynb_checkpoints/' '.DS_Store' > .gitignore
    git init -q -b main
    git add -A
    git -c user.email="${GIT_AUTHOR_EMAIL:-capsule@localhost}" \
        -c user.name="${GIT_AUTHOR_NAME:-capsule-export}" \
        commit -q -m "Initial BabyView Objects manuscript capsule"
  )
fi

# ----- archive -----
if [[ "$ARCHIVE" -eq 1 ]]; then
  echo "==> archive"
  tar -C "$(dirname "$OUT")" -czf "${OUT}.tar.gz" "$(basename "$OUT")"
  chmod go-rwx "${OUT}.tar.gz" 2>/dev/null || true
  echo "    wrote ${OUT}.tar.gz ($(du -sh "${OUT}.tar.gz" | awk '{print $1}'))"
fi

echo
echo "==> Done"
du -sh "$OUT"
echo "Top-level:"
du -sh "$OUT"/* 2>/dev/null | sort -hr
echo
echo "Contents check (analysis notebooks + scripts):"
ls "$MS_OUT"/*.ipynb 2>/dev/null | xargs -n1 basename
ls "$MS_OUT/scripts"
echo
echo "Next: upload $OUT (or ${OUT}.tar.gz) to CodeOcean / GitHub"
