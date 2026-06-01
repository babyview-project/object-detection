# Reproducing BabyView Objects (manuscript 2026)

This document describes how to rerun the analyses reported in *BabyView Objects*
(manuscript submission, June 2026) using this repository. The numbered notebooks
under `analysis/manuscript-2026/` are the primary analysis code (merged from the
former `preprint-2026` and `manuscript-2026` folders). Public anonymized tables
live in [`data/shared_data_manuscript_2026/`](../../data/shared_data_manuscript_2026/) for verifying most aggregate
statistics without access to raw BabyView recordings.

## Repository layout

```text
analysis/manuscript-2026/
├── 00_build_exemplar_embeddings.md            # Stage 0 checklist (START HERE for 02–05)
├── scripts/                                   # Helper .py (Stage 0, stats, publish)
├── scripts/check_exemplar_stage.py            # Verify exemplar_set_embeddings/
├── run_exemplar_embedding_stage.sh            # Headless Stage 0 driver
├── SCRIPTS.md                                 # Catalog of scripts/
├── 06_exemplar_set_zscore_embeddings.ipynb    # Stage 0a — BabyView category embeddings
├── 07_things_exemplar_set_zscore_embeddings.ipynb   # Stage 0b — THINGS CLIP + DINOv3
├── 01_long_tailed_distribution.ipynb          # Category frequency & power-law fits
├── 02_category-wise_cosine_sim.ipynb          # BV vs THINGS per-category cosine
├── 03_bv_things_rdm_comparison.ipynb          # RDM structure (CLIP / DINOv3 / BabyDINOv3)
├── 04_individual_rdms.ipynb                   # Per-child RDMs & top-8 agreement
├── 05_within_between_cdi_cluster_correlation.ipynb  # CDI cluster geometry + shuffle tests
├── 08_top8_kid_category_cosine_similarity.ipynb
├── 09_top8_local_global_variability_valid85.ipynb
├── 10_animal_depiction_label_proportions.ipynb
├── exemplar_set_embeddings/{valid129,valid85}/
├── main_results_valid129s_04302026/            # Main-text figures & tables
├── supplemental_results_valid85cats_04302026/  # Supplement / top-8 valid85
├── not_in_manuscript/                          # Exploratory / CCN / drift (not in submission)
├── scripts/build_shared_public_data.py
├── paths.example.env
└── requirements-manuscript.txt
```

**Not in the submitted manuscript** — see [`not_in_manuscript/README.md`](not_in_manuscript/README.md):

- `not_in_manuscript/10_fill_results_preprint_numbers.ipynb` — LaTeX number autofill (optional)
- Notebooks `12`–`13`, longitudinal drift, clutter proxy, CCN exemplar variability
- Early t-SNE/UMAP, VEST export, pooled CDI-domain exploratory scripts

## Environment

1. Python 3.10+ with scientific stack (see `requirements-manuscript.txt`).
2. Jupyter for notebooks; run with cwd = `analysis/manuscript-2026/` unless noted.
3. For full replication from crops, set paths from `paths.example.env` (copy to
   `paths.local.env` and `source` it, or export `BV_*` variables in the shell).

```bash
pip install -r analysis/manuscript-2026/requirements-manuscript.txt
```

## Data tiers

| Tier | What | Who needs it |
|------|------|----------------|
| **A — `data/shared_data_manuscript_2026/`** | Category embeddings (z-scored), detection prevalences, main result CSVs, anonymized top-8 tables | Anyone verifying paper numbers |
| **B — `exemplar_set_embeddings/`** | Same tables at full precision (in repo after local run) | Regenerating figures from notebooks 02–05 without re-embedding |
| **C — Per-image `.npy` + CLIP filter lists** | Millions of crop embeddings under `BV_EMBEDDINGS_BASE` | Rebuilding embeddings (notebooks 06–07) |
| **D — Raw detections / video** | YOLOE outputs, BabyView access | Notebook 01 from scratch, individual RDMs (04) |

Tier **A** is committed under `data/shared_data_manuscript_2026/` and listed in `data/shared_data_manuscript_2026/MANIFEST.json`.
Regenerate after local runs with:

```bash
python analysis/manuscript-2026/scripts/build_shared_public_data.py
```

## Recommended workflow

### Stage 0 — Build `exemplar_set_embeddings/` (required for 02–05)

Follow **[00_build_exemplar_embeddings.md](00_build_exemplar_embeddings.md)**:

1. **06** / `scripts/exemplar_set_zscore_embeddings.py` — BabyView category tables  
2. **07** — THINGS CLIP + DINOv3  
3. `scripts/things_exemplar_set_zscore_babydinov3.py` (+ optional GPU embed script)  
4. `python scripts/check_exemplar_stage.py --category-set valid129`

Or run `./run_exemplar_embedding_stage.sh valid129` (see that script for `SKIP_BV`, `RUN_THINGS_BD3_EMBED`, etc.).

Notebook **01** does not use these tables and can run in parallel with Stage 0.

### Verify manuscript statistics (Tier A only)

Aggregate result tables are in `data/shared_data_manuscript_2026/results_valid129/` (see `MANIFEST.json`).
Compare them to the manuscript or recompute from Tier B notebooks **01–05**, **08–09**.

Optional LaTeX autofill: run
[`not_in_manuscript/10_fill_results_preprint_numbers.ipynb`](not_in_manuscript/10_fill_results_preprint_numbers.ipynb)
(cwd = `not_in_manuscript/`). It auto-discovers `main_results_valid129*/results/` under
the manuscript folder (or symlink/copy from `data/shared_data_manuscript_2026/results_valid129/`) and writes
`results_preprint_numbers_autofill.txt` in `analysis/manuscript-2026/`.

### Regenerate main figures (Tier B)

| Step | Notebook / script | Produces |
|------|-------------------|----------|
| **0** | [Stage 0](00_build_exemplar_embeddings.md): `06` → `07` → BabyDINOv3 scripts | `exemplar_set_embeddings/` |
| 1 | `01` (may run parallel with Stage 0) | Long-tail CSVs → `main_results_.../results/` |
| 2 | `02` | Category-wise cosine tables & montages |
| 3 | `03` | RDM comparisons, category orders |
| 4 | `05` | CDI within/between, cluster shuffle, delta figures |
| 5 | `scripts/compute_binary_template_correlations.py` | Binary CDI template vs real RDM |
| 6 | `scripts/compare_dinov3_vs_babydinov3.py` | DINOv3 vs BabyDINOv3 supplement |
| 7 | `04` | Individual & top-8 RDM panels (needs Tier C) |
| 8 | `08`, `09`, `scripts/top8_within_between_vs_things.py` | Top-8 supplement |
| 9 | `10` | Animal depiction proportions (supplement) |

Set `CATEGORY_SET` to `valid129` (main text) or `valid85` (supplement) at the top of
each notebook. Outputs use threshold token `filtered-0.27` in filenames.

### Full rebuild from embeddings (Tier C)

1. Obtain BabyView object-detection embeddings (YOLOE + CLIP/DINOv3/BabyDINOv3) per
   `yoloe/README.md` and project data agreements.
2. Configure `BV_EMBEDDINGS_BASE`, `BV_CLIP_FILTER_LIST`, and model dirs (see
   `paths.example.env`).
3. Complete [Stage 0](00_build_exemplar_embeddings.md) (`06`, `07`, BabyDINOv3 scripts).
4. Continue with steps 1–9 in the table above.

Headless helpers: `./run_exemplar_embedding_stage.sh`, `./run_06_zscore_tmux.sh`, `./run_babydinov3_things_embed_tmux.sh`.

## Figure ↔ artifact mapping (valid129 main text)

| Manuscript topic | Primary outputs | Notebook / script |
|------------------|-----------------|-------------------|
| Long-tailed category frequencies | `long_tailed_*_valid129.csv`, power-law fits | `01` |
| BV–THINGS category cosine | `category_wise_cosine_similarity_*_valid129.csv` | `02` |
| RDM structure (CLIP / DINOv3) | `bv_things_rdm_comparison_*`, clustered PDFs in `figures/` | `03` |
| CDI cluster within/between | `cluster_within_between_*`, `bv_vs_things_cluster_strength_*` | `05` |
| Binary semantic template control | `binary_template_vs_real_rdm_correlations_valid129.csv` | `compute_binary_template_correlations.py` |
| Top-8 RDM agreement | `individual_rdm_pairwise_*_top8_densest_*` (anonymized in `data/shared_data_manuscript_2026/`) | `04` |
| DINOv3 vs BabyDINOv3 | `dinov3_vs_babydinov3_*`, UMAP coords | `compare_dinov3_vs_babydinov3.py`, `plot_things_dinov3_vs_babydinov3_umap.py` |
| Animal depiction supplement | `animal_depiction_label_proportions_by_category.csv` | `10` (aggregate also in `data/shared_data_manuscript_2026/inputs/`) |

Supplement (valid85): repeat `02`–`03` with `CATEGORY_SET=valid85`; top-8 analyses in
`supplemental_results_valid85cats_04302026/` and `data/shared_data_manuscript_2026/top8_valid85/`.

## Category sets

- **valid129** — 129 CDI categories passing detection precision ≥ 0.6 and CLIP
  filter 0.27 (`data/included_categories_valid129.txt`).
- **valid85** — Subset used for several supplement panels
  (`data/included_categories_valid85.txt`).

## Privacy

- One subject (`00270001`) is excluded from all embedding analyses (see notebook `04`).
- Public release uses `participant_01`–`participant_08` for the eight densest children;
  ranks are in `data/shared_data_manuscript_2026/metadata/participant_registry_top8.csv`.
- Do not commit raw annotation filenames, video paths, or `paths.local.env`.

## Citation / code availability

When submitting the manuscript, link to this repository and note that
`data/shared_data_manuscript_2026/` contains anonymized intermediates, with full
per-image replication requiring BabyView data access as described above.
