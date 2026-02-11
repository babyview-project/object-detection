# Exemplar Variability Analyses

This folder contains Jupyter notebooks (and equivalent Python scripts) for analyzing within-category exemplar variability in BabyView (BV) and THINGS embeddings (CLIP and DINOv3). The notebooks are numbered in a suggested run order; dependencies between them are described below.

**Recommended:** Run notebooks with the kernel working directory set to this folder (`exemplar_variability_analyses`) so that relative paths and module imports resolve correctly.

---

## Notebook index

| # | Notebook | Purpose |
|---|----------|--------|
| **01** | [01_within_category_variability.ipynb](01_within_category_variability.ipynb) | Compute within-category variability (distance to centroid, mean pairwise distance, per-exemplar variance, cosine similarity to centroid) for BV and THINGS. Produces `*_within_category_variability.csv` and t-SNE plots per dataset. |
| **02** | [02_load_things_embeddings.ipynb](02_load_things_embeddings.ipynb) | **Utility:** Load THINGS embeddings from directory (DinoV3/CLIP) or from .docs. Used by notebooks 03 and 06 when working with THINGS. |
| **03** | [03_bv_to_things_centroid_distances.ipynb](03_bv_to_things_centroid_distances.ipynb) | For each category, compute BV exemplar distances to the THINGS category centroid (and to the BV centroid). Outputs summary CSV, scatter (spread vs distance to THINGS), optional per-exemplar CSV, t-SNE per category, and montages of farthest BV exemplars from THINGS centroid. |
| **04** | [04_bv_within_category_knn_diversity.ipynb](04_bv_within_category_knn_diversity.ipynb) | Within-category **kNN diversity** for BV: for each exemplar, mean distance to its k nearest neighbors (within category). Low mean kNN distance ⇒ more micro-structure (local clusters); high ⇒ more uniform spread. Produces per-category summary CSV and optional per-exemplar CSV; supports multiple k (e.g. 5 and 10) for a combined multi-k summary. |
| **05** | [05_visualize_bv_knn_diversity.ipynb](05_visualize_bv_knn_diversity.ipynb) | **Visualization:** Reads outputs from 03 and 04. Produces (1) bar plot of categories ranked by mean kNN distance, (2) k=5 vs k=10 scatter, (3) kNN diversity vs centroid spread scatter, (4) optional violin plots of per-exemplar kNN distance for selected categories. |
| **06** | [06_visualize_exemplars_montage.ipynb](06_visualize_exemplars_montage.ipynb) | Montages of cropped exemplars for **top and bottom variable categories** (by distance-to-centroid from the variability CSV). Supports `bv_clip`, `bv_dinov3`, `things_clip`, `things_dinov3`. Exemplars are ordered by distance to centroid so spread is visible. |

---

## Workflow and dependencies

```
01_within_category_variability  →  *_within_category_variability.csv
        ↓
02_load_things_embeddings       (utility; used by 03 and THINGS parts of 06)
        ↓
03_bv_to_things_centroid_distances  →  bv_to_things_centroid_*_summary.csv, scatter, t-SNE, montages
        ↓
04_bv_within_category_knn_diversity →  bv_within_category_knn_*_summary.csv (and optional per-exemplar)
        ↓
05_visualize_bv_knn_diversity   (reads 03 + 04 outputs; produces rank bars, k5 vs k10, kNN vs spread, violins)

06_visualize_exemplars_montage  (reads *_within_category_variability.csv from 01; uses embeddings per config)
```

- **01** should be run first if you need variability CSVs or the montages in **06**.
- **02** defines loaders used by **03** (and by THINGS configs in **06**); run or import as needed.
- **03** and **04** can be run in parallel; **05** needs both of their outputs.
- **06** needs the variability CSVs from **01** (and, for THINGS configs, THINGS embeddings as in **02**).

---

## Main outputs (by notebook)

| Notebook | Example outputs |
|---------|------------------|
| 01 | `bv_clip_within_category_variability.csv`, `things_clip_within_category_variability.csv`, t-SNE plots by category, violin plots |
| 02 | (No file outputs; defines loaders) |
| 03 | `bv_to_things_centroid_{clip,dinov3}_summary.csv`, `*_spread_vs_things_centroid.png`, `*_tsne_by_category/`, `*_montages_farthest_from_things_centroid/` |
| 04 | `bv_within_category_knn_{clip,dinov3}_k{k}_summary.csv`, optional `*_per_exemplar.csv`, `*_multi_k_summary.csv` |
| 05 | `*_k{k}_rank_bars.png`, `*_k5_vs_k10_scatter.png`, `*_k{k}_vs_centroid_spread.png`, optional `*_violins_selected.png` |
| 06 | `exemplar_montages/` or `exemplar_montages_{config}/`: `exemplar_montage_{rank}_{category}.png` and `*_distances.txt` |

---

## Python scripts

Each notebook has a corresponding script for command-line use (same names without the numeric prefix and with `.py` extension): `load_things_embeddings.py`, `bv_to_things_centroid_distances.py`, `bv_within_category_knn_diversity.py`, `visualize_bv_knn_diversity.py`, `visualize_exemplars_montage.py`. The scripts accept `argparse` options; the notebooks use a parameters cell at the top instead.

---

## Data and paths

- **BV grouped embeddings:** e.g. `.../yoloe_cdi_embeddings/clip_embeddings_grouped_by_age-mo_normalized/{category}/*.npy`
- **THINGS embeddings:** e.g. `.../clip_image_embeddings_npy_by_category/{category}/*.npy`
- **Category list:** `../../../data/things_bv_overlap_categories_exclude_zero_precisions.txt`
- **Metadata/crops (BV):** configurable in each notebook (e.g. `merged_frame_detections_with_metadata.csv`, cropped images directory).

Path constants are set in the first code cells of each notebook; adjust them for your environment.
