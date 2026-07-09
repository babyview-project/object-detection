# Python scripts in `manuscript-2026/scripts/`

Numbered notebooks (**01–10**, plus Stage 0 **06–07**) are the main manuscript analyses. **Unnumbered `.py` files**
live under [`scripts/`](scripts/) — Stage 0 headless runners, statistics imported by notebooks,
supplement figures, and publication packaging. Exploratory scripts live under
[`not_in_manuscript/`](not_in_manuscript/) (see [`not_in_manuscript/README.md`](not_in_manuscript/README.md)).

Run scripts from `analysis/manuscript-2026/` (e.g. `python scripts/check_exemplar_stage.py`).
Shared paths: [`manuscript_config.py`](manuscript_config.py) at the manuscript root.

## Quick map

| Script | Stage | Run when | Notebook / output |
|--------|-------|----------|-------------------|
| [`manuscript_config.py`](manuscript_config.py) | — | import only | Shared paths (`MANUSCRIPT_DIR`, `exemplar_embed_dir`, …) |
| [`scripts/check_exemplar_stage.py`](scripts/check_exemplar_stage.py) | 0 | After 06–07 | Gate before **02–05** |
| [`scripts/exemplar_set_zscore_embeddings.py`](scripts/exemplar_set_zscore_embeddings.py) | 0a | Build BV tables | **06** → `exemplar_set_embeddings/` |
| [`scripts/things_exemplar_set_zscore_babydinov3.py`](scripts/things_exemplar_set_zscore_babydinov3.py) | 0d | After THINGS BabyDINOv3 `.npy` | **07** (BabyDINOv3 block) |
| [`scripts/create_babydinov3_things_embeddings.py`](scripts/create_babydinov3_things_embeddings.py) | 0c | GPU; once per checkpoint | Per-image THINGS `.npy` |
| [`scripts/create_babydinov3_crop_embeddings.py`](scripts/create_babydinov3_crop_embeddings.py) | 0* | GPU; upstream of **06** | Per-crop BV BabyDINOv3 `.npy` on disk |
| [`scripts/bv_things_cdi_shuffle_inference.py`](scripts/bv_things_cdi_shuffle_inference.py) | — | import only | **05**, `top8_within_between_vs_things.py` |
| [`scripts/compute_binary_template_correlations.py`](scripts/compute_binary_template_correlations.py) | 1 | After **03** (orders/embeddings) | Binary CDI template RDM control |
| [`scripts/compare_dinov3_vs_babydinov3.py`](scripts/compare_dinov3_vs_babydinov3.py) | 1 | After Stage 0 | DINOv3 vs BabyDINOv3 supplement stats |
| [`scripts/plot_things_dinov3_vs_babydinov3_umap.py`](scripts/plot_things_dinov3_vs_babydinov3_umap.py) | 1 | After compare script | UMAP figure (THINGS backbones) |
| [`scripts/clip_threshold_sensitivity.py`](scripts/clip_threshold_sensitivity.py) | 1 | Optional supplement | CLIP threshold sensitivity SI |
| [`scripts/top8_within_between_vs_things.py`](scripts/top8_within_between_vs_things.py) | 1 | After Stage 0; often from **05** | Top-8 CDI vs THINGS (valid85/129) |
| [`scripts/build_shared_public_data.py`](scripts/build_shared_public_data.py) | publish | After local results exist | [`data/shared_data_manuscript_2026/`](../../data/shared_data_manuscript_2026/) |

\* `create_babydinov3_crop_embeddings.py` is **Tier C** infrastructure (millions of `.npy`).
Only needed if BabyDINOv3 crop embeddings are not already on your cluster path.

Shell wrappers: [`run_exemplar_embedding_stage.sh`](run_exemplar_embedding_stage.sh),
[`run_06_zscore_tmux.sh`](run_06_zscore_tmux.sh),
[`run_babydinov3_things_embed_tmux.sh`](run_babydinov3_things_embed_tmux.sh).

---

## Infrastructure (import, do not run as main)

### `manuscript_config.py`

Single source for `MANUSCRIPT_DIR`, `PROJECT_ROOT`, `DATA_DIR`, `EXEMPLAR_EMBED_ROOT`,
`CATEGORY_SET_FILES`, and helpers `bv_embedding_csv()` / `things_embedding_csv()`.

```python
from manuscript_config import exemplar_embed_dir, bv_embedding_csv
```

### `scripts/bv_things_cdi_shuffle_inference.py`

Shared logic for CDI-domain shuffle nulls, cluster Δ tables, and bar plots. Used by:

- Notebook **05** (`05_within_between_cdi_cluster_correlation.ipynb`)
- [`scripts/top8_within_between_vs_things.py`](scripts/top8_within_between_vs_things.py)

Notebook **05** adds `scripts/` to `sys.path` when importing this module.

---

## Stage 0 — Embedding tables (see [`00_build_exemplar_embeddings.md`](00_build_exemplar_embeddings.md))

| Script | Purpose |
|--------|---------|
| `scripts/exemplar_set_zscore_embeddings.py` | Headless **06**: scan CLIP filter list → average BV `.npy` → z-score CSVs. `./run_06_zscore_tmux.sh` |
| `scripts/things_exemplar_set_zscore_babydinov3.py` | Z-score THINGS BabyDINOv3 category trees → `things_babydinov3_exemplar_avg_*`. `BV_CATEGORY_SET=valid129` |
| `scripts/create_babydinov3_things_embeddings.py` | Embed ~1.8k THINGS images (GPU). `./run_babydinov3_things_embed_tmux.sh` |
| `scripts/create_babydinov3_crop_embeddings.py` | Embed BV crops with BabyDINOv3 checkpoint (GPU, long). Feeds **06** when `BV_BABYDINOV3_EMBEDDINGS_DIR` is populated |
| `scripts/check_exemplar_stage.py` | `python scripts/check_exemplar_stage.py --category-set valid129` — exit 1 if any of 6 core CSVs missing |

Notebook **07** (THINGS CLIP + DINOv3) has **no** separate headless script yet; run the notebook or set `EXECUTE_NOTEBOOK_07=1` in `run_exemplar_embedding_stage.sh`.

---

## Stage 1 — Manuscript analyses (after Stage 0 + embeddings exist)

| Script | Typical command | Outputs |
|--------|-----------------|--------|
| `scripts/compute_binary_template_correlations.py` | `python scripts/compute_binary_template_correlations.py` | `main_results_*/results/binary_template_*` |
| `scripts/compare_dinov3_vs_babydinov3.py` | `CATEGORY_SET=valid129 python scripts/compare_dinov3_vs_babydinov3.py` | `dinov3_vs_babydinov3_*`, scatter PNG |
| `scripts/plot_things_dinov3_vs_babydinov3_umap.py` | `python scripts/plot_things_dinov3_vs_babydinov3_umap.py` | UMAP coords + figure under `main_results_*/figures/` |
| `scripts/clip_threshold_sensitivity.py` | `python scripts/clip_threshold_sensitivity.py` | SI figures, `clip_threshold_sensitivity_valid*.csv` |
| `scripts/top8_within_between_vs_things.py` | `python scripts/top8_within_between_vs_things.py` | `supplemental_results_*/results/top8_vs_things_*` |

Notebook **05** can invoke `scripts/top8_within_between_vs_things.py` in a cell; you can also run it standalone with env vars `BV_EMBED_MODEL`, `TOP8_CATEGORY_SET`.

---

## Publication

| Script | Purpose |
|--------|---------|
| `scripts/build_shared_public_data.py` | Copy/anonymize tables → [`data/shared_data_manuscript_2026/`](../../data/shared_data_manuscript_2026/). Run after local pipeline completes. |

---

## Not in the manuscript (`not_in_manuscript/`)

Includes `pooled_bv_vs_things_cdi_domain_valid85.py`, drift pilots, early `visualize_tsne_umap_*.py`,
CCN `exemplar_variability_analyses/`, VEST export, etc. **Do not run for submission reproduction**
unless you are extending the project.

---

## Why keep scripts separate from notebooks?

| Reason | Example |
|--------|---------|
| **Long / tmux jobs** | `exemplar_set_zscore_embeddings.py`, `create_babydinov3_*` |
| **Reusable library** | `bv_things_cdi_shuffle_inference.py` |
| **Thin CLIs** | `compare_dinov3_vs_babydinov3.py`, `check_exemplar_stage.py` |
| **Called from notebooks** | **05** → `scripts/top8_within_between_vs_things.py` |

Merging everything into numbered notebooks would duplicate logic and make headless/cluster runs harder. Use this file + notebook numbers together: **notebooks for exploration**, **scripts for repeatability**.
