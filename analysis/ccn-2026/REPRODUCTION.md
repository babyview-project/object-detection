# CCN 2026 — reproducibility notes (exemplar variability)

This repo contains the **code** for the CCN 2026 poster analyses (Plots A–C + the 2×2 extremes t-SNE panel), plus a mixture of **committed** (versioned) and **local-only** (gitignored) generated outputs.

## The key reality right now

1. **End-to-end reruns on a fresh clone are not guaranteed** without access to the same cluster paths/data used by the notebooks (BV crops/embeddings and THINGS embedding stores).
2. **Much of the quantitative verification is possible right now** because the working tree already contains the *summary* CSVs needed for the reported aggregates (Plot B and Plot C summary tables).
3. A subset of files are **not safe to redistribute publicly** (notably per-exemplar tables containing `subject_id` and `stem`/filename-like fields, plus images/thumbnails/montages).

## Data tiers (CCN-specific)

### Tier A — Shareable / verifiable without raw data
These files are intended to be safe to redistribute because they are **category-level aggregates** only (no `subject_id`, no per-exemplar `stem`):

- Plot B (BV→BV centroid distance) *summary* tables:
  - `bv-to-bv-centroid_distance_clip_summary.csv`
  - `bv-to-bv-centroid_distance_dinov3_summary.csv`
  - `bv-to-bv-centroid_distance_clip_vs_dinov3_comparison.csv`
- Plot C (within-category kNN diversity) summary tables:
  - `bv_within_category_knn_clip_k5_summary.csv`
  - `bv_within_category_knn_dinov3_k5_summary.csv`
  - `bv_within_category_knn_clip_vs_dinov3_k5_comparison.csv`
- CCN local/global extremes lists (category names only):
  - `ccn2026_local_global_extreme_categories_clip_dino.csv`
  - `ccn2026_local_global_extreme_categories_things_clip_dino.csv`
  - `ccn2026_local_global_overlap_categories_used_for_tsne_2x2.csv`
- THINGS-vs-BV overlap counts (category-level):
  - `things_image_vs_embedding_counts.csv`
- THINGS/BV metrics CSVs under:
  - `plotC_knn_diversity_outputs/new_things_embeddings_20260428/` (metric tables, not per-exemplar)

Tier A is what `scripts/build_shared_public_data_ccn.py` will export into `data/shared_data_ccn_2026/`.

### Tier B — Full rerun of notebooks (requires private cluster paths)
Plot A, Plot B, Plot C, and the 2×2 extremes t-SNE panel all compute embeddings/metrics from:
- BV crops and BV embeddings (configured via env vars in notebook parameter cells)
- THINGS embeddings (loaded via `analysis/ccn-2026/load_things_embeddings.py`, env-driven paths)

This tier can work **only for users who have access to those source directories**.

### Tier C — Rebuild from raw video / raw detections
This is not covered as a fully automated “Stage 0” for CCN in this folder (unlike the manuscript pipeline). Rebuilding from raw sources requires running parts of the detection/cropping/embedding pipeline documented elsewhere in the repo (see top-level `README.md` and `yoloe/` docs).

## What you can do today

### 1. Generate the public-shareable CCN data bundle (recommended)
This exports only the safe Tier A tables into `data/shared_data_ccn_2026/`.

From the repo root (`object-detection/`):
```bash
python analysis/ccn-2026/scripts/build_shared_public_data_ccn.py
```

After this run, verify that the output directory contains at least:
- `MANIFEST.json`
- category-level Plot B summary CSVs
- Plot C summary CSVs

### 2. Verify headline results from Tier A (no rerun required)
Compare the published aggregate numbers against the Tier A CSVs in `data/shared_data_ccn_2026/` (or the existing local output CSVs under `analysis/ccn-2026/plotB_tsne_distance_to_centroid_outputs_*` and `analysis/ccn-2026/plotC_knn_diversity_outputs/`).

### 3. If you need to rerun notebooks end-to-end
You must set the BV/THINGS paths used by the notebooks to locations you can access.

See `analysis/ccn-2026/paths.example.env` for a template of the defaults.

## Privacy / redistribution guardrails

Do **not** publish or commit:
- Plot B per-exemplar CSVs containing `subject_id` and `stem` (examples exist as `*_per_exemplar.csv`)
- images, montages, and thumbnails (Plot A montages, Plot B t-SNE PNG/PDFs, Plot E montages, interactive `tsne_datapage/` thumbnails)
- any raw crop/video paths or “participant identifier” fields (even if anonymized elsewhere)

## File mapping (notebooks → expected outputs)

- `02_plotB_tsne_distance_to_centroid.ipynb`
  - produces Plot B summary and per-exemplar CSVs (per-exemplar is Tier B/C only)
- `03_plotC_knn_diversity.ipynb`
  - produces Plot C summary CSVs and THINGS/BV comparison tables

The output folders are intentionally gitignored; the intended reproducibility path is to export Tier A tables via the bundle script above.

## Notebooks (public GitHub)

CCN notebooks are committed **without executed outputs** (no embedded figures, stdout, or machine-specific paths from prior runs). Before sharing results, re-run locally with `paths.local.env` configured (copy from [`paths.example.env`](paths.example.env)).

