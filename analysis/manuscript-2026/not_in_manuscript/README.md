# Analyses not in the submitted manuscript

Code and outputs here were **not** part of the BabyView Objects submission
(June 2026). They include CCN-style exemplar variability (poster pipeline: [`../../ccn-2026/README.md`](../../ccn-2026/README.md)), early t-SNE/UMAP
explorations, longitudinal embedding-drift pilots, clutter proxies, VEST exports,
and pooled top-8 supplemental variants that were superseded by notebooks **01–10**
in the parent directory.

## Run from this folder

Set your Jupyter or shell working directory to
`analysis/manuscript-2026/not_in_manuscript/` unless a script says otherwise.

Shared path helpers:

```python
from _paths import MANUSCRIPT_DIR, PROJECT_ROOT, NOT_IN_MANUSCRIPT_DIR
```

Manuscript pipeline outputs (embeddings, `main_results_*`, `supplemental_*`) live
in `MANUSCRIPT_DIR` (parent folder).

## Contents

| Item | Description |
|------|-------------|
| `10_fill_results_preprint_numbers.ipynb` | Autofill `results_preprint.tex` from pipeline CSVs (drafting aid) |
| `12_clutter_proxy_objects_per_frame.ipynb` | Objects-per-frame vs age (clutter proxy) |
| `13_embedding_visual_drift_pilot.ipynb` | Longitudinal embedding drift pilot |
| `embedding_drift_exploration/` | Drift pilot results & figures |
| `embedding_visual_drift_pilot.py`, `embedding_drift_extensions.py` | Drift scripts |
| `per_category_context_analysis.py` | Location/activity stratification |
| `INTEGRATED_ANALYSIS_PLAN.md` | CCN → longitudinal planning notes |
| `exemplar_variability_analyses/` | Within-category variability (CCN extended abstract) |
| `bv_things_results_*`, `tsne_cdi_results_*`, `umap_cdi_results_*` | Early 2D embedding plots |
| `visualize_tsne_umap_*.py`, `bv_things_semantic_category_analysis.ipynb` | t-SNE/UMAP utilities |
| `sample_bv_things_exemplars_by_clip_similarity.ipynb` | Exemplar sampling by CLIP distance |
| `clip_vs_dinov3_babyview_comparison.py` | Early CLIP vs DINOv3 comparison |
| `pooled_bv_vs_things_cdi_domain_valid85.py` | Pooled (non–top-8) CDI domain bars |
| `top8_subject_category_centroid_comparison.py` | Top-8 centroid comparison (extra figures) |
| `build_bv_valid129_vest_export.py`, `vest_export_bv_valid129/` | VEST visualization export |
| `online_annotation/` | Online annotation supplement (Quarto) |
| `plots/` | Misc plots from early CDI visualizations |

## Related manuscript code

See [`../README.md`](../README.md) and [`../REPRODUCTION.md`](../REPRODUCTION.md) for
notebooks **01–10**, `data/shared_data_manuscript_2026/`, and submission figures.
