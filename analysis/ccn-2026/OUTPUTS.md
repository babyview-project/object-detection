# CCN 2026 — outputs reference

## Metrics — `valid7018/` (in git)

From `scripts/compute_valid7018_local_global.py`:

| File | Description |
|------|-------------|
| `bv_valid7018_{clip,dinov3}_local_global_k5.csv` | Per-category global dispersion + mean kNN |
| `bv_valid7018_clip_vs_dinov3_local_global_k5.csv` | CLIP vs DINO per category |
| `bv_valid7018_correlations_k5.csv` | Headline Spearman ρ |
| `valid7018_paper_stats.json` | Category picks + autofilled statistics |
| `valid7018_run.json` | Run metadata |

## Scratch figures — `valid7018/figures/` (gitignored)

All regenerated PNG/PDF variants from `generate_valid7018_paper_figures.py`, including exploratory frequency panels.

## Abstract panels — `abstract_figures/` (in git)

Subset copied by `publish_abstract_figures()`:

| File | Panel |
|------|-------|
| `fig1A_valid7018_montages_low_to_high_global.*` | Montages (clock → book) |
| `fig1B_valid7018_tsne_dinov3.*` | DINOv3 t-SNE aligned with montage categories |
| `fig1B_valid7018_tsne_dinov3_semantic_diverse.*` | Alternative t-SNE (one per CDI semantic group) |
| `fig1C_valid7018_cross_model_k5.*` | Cross-model global/local scatter |
| `fig_explore_frequency_vs_global_robustness_2x2.*` | Frequency vs dispersion |
| `valid7018_figure_category_selection.csv` | Category picks for panels A/B |

## Public bundle — `data/shared_data_ccn_2026/`

Mirror of `valid7018/` metrics plus `embeddings/`, `montages/`, and `inputs/`.
