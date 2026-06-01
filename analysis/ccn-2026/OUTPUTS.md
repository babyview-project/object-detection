# CCN 2026 — outputs reference

Generated artifacts are grouped by plot letter or script. Prefer **dated** Plot B folder `plotB_tsne_distance_to_centroid_outputs_20260402/` over `*_20260401/` unless you are reproducing an older run.

## Plot A — [`plotA_category_montages_low_to_high/`](plotA_category_montages_low_to_high/)

| File pattern | Description |
|--------------|-------------|
| `exemplar_montage_*_{category}_dist=*.png` / `.pdf` | Single-category montage (25 exemplars) |
| `exemplar_montage_combined_*_horizontal.svg` | Multi-category strip for poster |
| `plotA_selected_categories_low_to_high_variability.csv` | Selected categories, ranks, mean distances |

## Plot B — `plotB_tsne_distance_to_centroid_outputs_20260402/`

| File pattern | Description |
|--------------|-------------|
| `bv-to-bv-centroid_distance_{clip,dinov3}_summary.csv` | Per-category `mean_bv_to_bv_centroid`, `std`, `n_bv` |
| `bv-to-bv-centroid_distance_{clip,dinov3}_per_exemplar.csv` | Per crop: `dist_to_bv_centroid`, ids, stem |
| `bv-to-bv-centroid_distance_{clip}_tsne_by_category/` | Per-category t-SNE PNG/PDF |
| `*_clip_vs_dinov3_tsne_joint_side_by_side/` | Joint multi-category t-SNE panels |

Legacy joint BV–THINGS runs: `bv_to_things_centroid_{clip,dinov3}_tsne_by_category/` (superseded for CCN poster by BV-only notebook 02).

## Plot C — [`plotC_knn_diversity_outputs/`](plotC_knn_diversity_outputs/)

| Path / file | Description |
|-------------|-------------|
| `bv_within_category_knn_{clip,dinov3}_k5_summary.csv` | Category-level mean kNN distance |
| `bv_within_category_knn_clip_vs_dinov3_k5_comparison.csv` | CLIP vs DINO per category |
| `ccn2026_variability_2x2_panel.png` | Main 2×2 variability figure |
| `ccn2026_local_global_extreme_categories_clip_dino.csv` | Extreme category lists (CLIP & DINO) |
| `ccn2026_local_global_extremes_tsne_overlap_2x2.png` | t-SNE panel for overlap extremes |
| `new_things_embeddings_20260428/` | THINGS metrics **valid129**, used by `scripts/generate_things_plotabc_and_bv_comparisons.py` |
| `new_things_embeddings_valid85_20260428/` | Same for **valid85** |
| `original_embeddings_rerun/` | Metrics from raw on-disk embeddings |
| `zscore_rerun/` | Metrics from z-scored exemplar CSVs |

Human-readable extreme lists (legacy): [`old_plots/plotC_knn_diversity_outputs/ccn2026_local_global_extreme_categories_clip_dino.md`](old_plots/plotC_knn_diversity_outputs/ccn2026_local_global_extreme_categories_clip_dino.md)

## THINGS comparisons — `plot_things_and_bv_comparisons_outputs/`

Produced by [`scripts/generate_things_plotabc_and_bv_comparisons.py`](scripts/generate_things_plotabc_and_bv_comparisons.py). Includes Plot A/B/C analogs for THINGS, BV–THINGS scatter panels, correlation 2×2 figures, and `bv_vs_things_paired_stats_summary.csv`.

Variant: `plot_things_and_bv_comparisons_outputs_valid85/` (valid85 category set).

## Plot E — [`plotE_invalid_exemplar_montages_per_category/`](plotE_invalid_exemplar_montages_per_category/)

| File | Description |
|------|-------------|
| `invalid_exemplar_counts_by_category.csv` | Counts of failed per-file precision crops |
| `invalid_exemplar_montage_manifest.csv` | Paths to per-category montage PNG/PDF |
| `invalid_montage_{category}.png` | Montage of invalid exemplars |

## Interactive — [`tsne_datapage/`](tsne_datapage/)

| File | Description |
|------|-------------|
| `index.html` | Browser UI |
| `manifest.json` | Category coordinates + exemplar thumbnails |
| `thumbs/` | Cached crop thumbnails (large) |

## Archive & legacy — do not use for new runs

| Location | Notes |
|----------|--------|
| [`archive/`](archive/) | Large one-off exports (executed notebook 03, combined poster PNG) |
| [`old_plots/`](old_plots/) | ~600 MB of superseded Plot A/B/C outputs and label sidecars |

To reclaim disk space after verifying newer outputs, you may delete `old_plots/` and duplicate `plotB_*_20260401/` trees locally (not required for rerunning notebooks).
