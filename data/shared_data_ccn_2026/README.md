# CCN 2026 — shared data (valid7018 cohort)

Primary cohort (Methods-aligned):
- `valid7018/` — global + local metrics on the same 7,018 rater-validated crops
  (`valid7018_paper_stats.json` for headline ρ). **Must match**
  `analysis/ccn-2026/valid7018/` (cohort-internal feature-wise z-score).
- `embeddings/valid7018_bv_embeddings.zip` — paired CLIP+DINO `.npy` (~20 MB;
  feature-wise globally normalized; run `build_valid7018_embeddings_zip.py`)
- `embeddings/valid7018_embedding_norm_stats.json` — cohort z-score μ/σ (fit on 7,018 crops)
- `crops/valid7018_crops.zip` — all 7,018 crop JPEGs aligned 1:1 with the embedding zip
  (run `build_valid7018_crops_zip.py`; **internal** — includes body parts / glasses)

## Models

| Model | Checkpoint / weights | Dim |
|-------|----------------------|-----|
| CLIP | OpenAI **`ViT-B-32-quickgelu`** (not LAION) | 512 |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 |

Embeddings are feature-wise z-scored with μ/σ fit on all 7,018 crops pooled across
valid85 (`normalization`: `featurewise_zscore_within_valid7018_cohort`).

## Public release (privacy-filtered)

- `public/` — crops + embeddings with body-part categories and glasses removed,
  stems anonymized. Rebuild with `build_valid7018_public_release.py`.

Also included:
- Category sets, per-class validation precision, frequency tables, CDI semantic map (`inputs/`)
- `montages/valid7018_montage_crops.zip` — JPEG thumbnails for abstract Figure 1A (subset)

## Regenerating this bundle

After recomputing metrics under `analysis/ccn-2026/valid7018/`, re-export so the
shared CSVs stay consistent with the embeddings:

```bash
python analysis/ccn-2026/scripts/build_shared_public_data_ccn.py
```

(Or copy the metric CSVs/JSON from `analysis/ccn-2026/valid7018/` into
`data/shared_data_ccn_2026/valid7018/`.)
