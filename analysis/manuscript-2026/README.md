# BabyView Objects — manuscript analyses (2026)

Analysis code for the BabyView Objects manuscript (June 2026 submission).

**New to the pipeline?** Start with **[Stage 0: build embeddings](00_build_exemplar_embeddings.md)**
(notebooks **06–07**), then **[REPRODUCTION.md](REPRODUCTION.md)** for the full workflow.

> **Note:** `analysis/preprint-2026/` redirects here. Exploratory work that was
> **not** in the submission lives in **[`not_in_manuscript/`](not_in_manuscript/)**.

## Quick links

| Resource | Purpose |
|----------|---------|
| [**00_build_exemplar_embeddings.md**](00_build_exemplar_embeddings.md) | **Stage 0** — run before 02–05 (notebooks 06–07) |
| [**SCRIPTS.md**](SCRIPTS.md) | Catalog of [`scripts/`](scripts/) helpers — when to run, vs notebooks |
| [REPRODUCTION.md](REPRODUCTION.md) | Step-by-step pipeline, dependencies, manuscript ↔ outputs |
| [scripts/check_exemplar_stage.py](scripts/check_exemplar_stage.py) | Verify `exemplar_set_embeddings/` before downstream notebooks |
| [run_exemplar_embedding_stage.sh](run_exemplar_embedding_stage.sh) | Headless Stage 0 driver (BV script + checks) |
| [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) | Paste-ready code/data availability text |
| [`data/shared_data_manuscript_2026/`](../../data/shared_data_manuscript_2026/) | Anonymized CSV/JSON tables safe to share with the paper |
| [not_in_manuscript/](not_in_manuscript/) | CCN variability, drift pilots, early t-SNE/UMAP (not in paper) |
| [../ccn-2026/README.md](../ccn-2026/README.md) | **CCN 2026 poster** — Plots A–C, kNN/local–global, THINGS comparisons |
| [paths.example.env](paths.example.env) | Local paths for per-image embeddings |
| [requirements-manuscript.txt](requirements-manuscript.txt) | Python packages |
| [manuscript_config.py](manuscript_config.py) | `MANUSCRIPT_DIR` / `PROJECT_ROOT` helpers |

```bash
python analysis/manuscript-2026/scripts/build_shared_public_data.py
```

## Pipeline stages

Run with cwd = `analysis/manuscript-2026/`.

### Stage 0 — Category embedding tables (run first for 02–05)

| Step | Notebook / script | Summary |
|------|-------------------|---------|
| 0a | `06_exemplar_set_zscore_embeddings.ipynb` | BabyView category embeddings |
| 0b | `07_things_exemplar_set_zscore_embeddings.ipynb` | THINGS CLIP + DINOv3 |
| 0c–0d | `scripts/create_babydinov3_things_embeddings.py`, `scripts/things_exemplar_set_zscore_babydinov3.py` | THINGS BabyDINOv3 |

See **[00_build_exemplar_embeddings.md](00_build_exemplar_embeddings.md)**. Quick check: `python scripts/check_exemplar_stage.py`.

### Stage 1+ — Manuscript analyses (notebooks 01–10)

| # | Notebook | Summary | Needs Stage 0? |
|---|----------|---------|----------------|
| 01 | `01_long_tailed_distribution.ipynb` | Detection frequencies & power-law | No |
| 02 | `02_category-wise_cosine_sim.ipynb` | BV vs THINGS category cosine | **Yes** |
| 03 | `03_bv_things_rdm_comparison.ipynb` | RDM comparison & figures | **Yes** |
| 04 | `04_individual_rdms.ipynb` | Per-child RDMs (per-image `.npy`) | Per-child data |
| 05 | `05_within_between_cdi_cluster_correlation.ipynb` | CDI cluster geometry | **Yes** |
| 06–07 | *(Stage 0)* | — | — |
| 08–09 | Top-8 supplement notebooks | Inter-child / variability | Mixed |
| 10 | `10_animal_depiction_label_proportions.ipynb` | Animal depiction supplement | Prior outputs / annotation CSV |

**Recommended order:** **Stage 0 (06→07)** → **01** (parallel OK) → **02 → 03 → 05** → scripts → **04, 08–09, 10**.

LaTeX autofill (optional, not in submission): [`not_in_manuscript/10_fill_results_preprint_numbers.ipynb`](not_in_manuscript/10_fill_results_preprint_numbers.ipynb).

## Key outputs & scripts (in submission)

| Path / script | Role |
|---------------|------|
| `main_results_valid129s_04302026/` | Main-text `results/` and `figures/` |
| `supplemental_results_valid85cats_04302026/` | Supplement & top-8 valid85 |
| `exemplar_set_embeddings/` | Category-level z-scored tables |

Helper `.py` files under **[`scripts/`](scripts/)** (Stage 0 runners, shuffle library, supplement CLIs, public data export) are cataloged in **[SCRIPTS.md](SCRIPTS.md)**.

## Conventions

- `CATEGORY_SET`: `valid129` (main) or `valid85` (supplement).
- Notebooks may use `PREPRINT_DIR`; it points at this folder (`manuscript_config.PREPRINT_DIR`).
