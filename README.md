# BabyView object detection

Code to detect, recognize, and label objects across BabyView videos. Models run on
frames extracted from video at **1 fps**. Open-vocabulary detection uses class names
from the child vocabulary MCDI survey (word lists in [`yoloe/tools/`](yoloe/tools/)).

For YOLOE setup, frame extraction, and batch prediction, see **[`yoloe/README.md`](yoloe/README.md)**.

---

## BabyView Objects manuscript (2026)

Primary analyses, figures, and **anonymized shareable tables** for the *BabyView Objects*
manuscript (June 2026 submission) live under **[`analysis/manuscript-2026/`](analysis/manuscript-2026/)**.
The former `preprint-2026` tree was merged here; **[`analysis/preprint-2026/README.md`](analysis/preprint-2026/README.md)** redirects.

### Start here

| Document | Use when |
|----------|----------|
| **[`analysis/manuscript-2026/README.md`](analysis/manuscript-2026/README.md)** | Overview, notebook order, key output folders |
| **[`analysis/manuscript-2026/00_build_exemplar_embeddings.md`](analysis/manuscript-2026/00_build_exemplar_embeddings.md)** | **Stage 0** — run notebooks **06–07** before **02–05** |
| **[`analysis/manuscript-2026/REPRODUCTION.md`](analysis/manuscript-2026/REPRODUCTION.md)** | Full pipeline, data tiers, figure ↔ notebook mapping |
| **[`analysis/manuscript-2026/SCRIPTS.md`](analysis/manuscript-2026/SCRIPTS.md)** | Catalog of [`analysis/manuscript-2026/scripts/`](analysis/manuscript-2026/scripts/) helpers |
| **[`analysis/manuscript-2026/DATA_AVAILABILITY.md`](analysis/manuscript-2026/DATA_AVAILABILITY.md)** | Paste-ready code/data availability text |

```bash
pip install -r analysis/manuscript-2026/requirements-manuscript.txt
# cwd = analysis/manuscript-2026/
python scripts/check_exemplar_stage.py --category-set valid129
```

### Manuscript notebooks (`analysis/manuscript-2026/`)

| # | Notebook | Role |
|---|----------|------|
| **06–07** | Stage 0 | Category-level z-scored embedding tables → `exemplar_set_embeddings/` |
| **01** | `01_long_tailed_distribution.ipynb` | Detection frequencies & power-law (no Stage 0) |
| **02–03** | Category cosine, BV–THINGS RDMs | Needs Stage 0 |
| **04** | `04_individual_rdms.ipynb` | Per-child RDMs (needs per-image embeddings) |
| **05** | CDI cluster geometry | Needs Stage 0 |
| **08–09** | Top-8 supplement | Inter-child / local–global variability |
| **10** | Animal depiction supplement | Uses annotation aggregates |

Shell helpers: `run_exemplar_embedding_stage.sh`, `run_06_zscore_tmux.sh`, `run_babydinov3_things_embed_tmux.sh`. Path template: [`paths.example.env`](analysis/manuscript-2026/paths.example.env). Config: [`manuscript_config.py`](analysis/manuscript-2026/manuscript_config.py).

### Manuscript outputs (local, after running the pipeline)

| Path | Contents |
|------|----------|
| `analysis/manuscript-2026/exemplar_set_embeddings/` | Category-level z-scored tables (`valid129`, `valid85`) |
| `analysis/manuscript-2026/main_results_valid129s_04302026/` | Main-text `results/` and `figures/` |
| `analysis/manuscript-2026/supplemental_results_valid85cats_04302026/` | Supplement & top-8 valid85 |

### Shareable data (no raw video or participant IDs)

**[`data/shared_data_manuscript_2026/`](data/shared_data_manuscript_2026/)** — anonymized CSV/JSON for verifying paper statistics.

| Resource | Purpose |
|----------|---------|
| [`data/shared_data_manuscript_2026/README.md`](data/shared_data_manuscript_2026/README.md) | Layout (`embeddings/`, `results_valid129/`, `top8_valid85/`, …) |
| `data/shared_data_manuscript_2026/MANIFEST.json` | File list and generation timestamp |

Regenerate after local runs:

```bash
python analysis/manuscript-2026/scripts/build_shared_public_data.py
```

Category lists used in the paper: [`data/included_categories_valid129.txt`](data/included_categories_valid129.txt), [`data/included_categories_valid85.txt`](data/included_categories_valid85.txt).

### Exploratory work (not in the submission)

**[`analysis/manuscript-2026/not_in_manuscript/`](analysis/manuscript-2026/not_in_manuscript/)** — drift pilots, clutter proxy, early t-SNE/UMAP, LaTeX autofill notebook, and related scripts. See [`not_in_manuscript/README.md`](analysis/manuscript-2026/not_in_manuscript/README.md).

---

## Related conference analyses

| Folder | Paper / venue | Entry point |
|--------|---------------|-------------|
| **[`analysis/ccn-2026/`](analysis/ccn-2026/)** | CCN 2026 poster — exemplar variability (Plots A–C, kNN, THINGS) | [`README.md`](analysis/ccn-2026/README.md), [`SCRIPTS.md`](analysis/ccn-2026/SCRIPTS.md), [`OUTPUTS.md`](analysis/ccn-2026/OUTPUTS.md) |
| **[`analysis/vss-2026/`](analysis/vss-2026/)** | VSS 2026 — group RDM / embedding pipeline (CLIP, DINOv3) | [`README.md`](analysis/vss-2026/README.md) |
| **[`analysis/individual_analyses/`](analysis/individual_analyses/)** | Per-child RDMs & developmental trajectories | [`README_notebooks_05_06_07.md`](analysis/individual_analyses/README_notebooks_05_06_07.md), [`README_pca_subspace_stability.md`](analysis/individual_analyses/README_pca_subspace_stability.md), [`clip_dino_rdm_correlations/README.md`](analysis/individual_analyses/clip_dino_rdm_correlations/README.md) |
| **[`analysis/developmental_trend_analysis_R/`](analysis/developmental_trend_analysis_R/)** | R / Quarto developmental stats | [`data/README_data_csvs.md`](analysis/developmental_trend_analysis_R/data/README_data_csvs.md) |

CCN variability code also appears under [`analysis/manuscript-2026/not_in_manuscript/exemplar_variability_analyses/`](analysis/manuscript-2026/not_in_manuscript/exemplar_variability_analyses/).

---

## Detection & preprocessing pipeline

End-to-end flow: **frames → YOLOE detections → crops → embeddings → manuscript notebooks**.

| Directory | Role |
|-----------|------|
| **[`yoloe/`](yoloe/)** | YOLOE install, `extract_frames.py`, `predict_frames.py`, examples | [`yoloe/README.md`](yoloe/README.md) |
| **`yoloe/tools/`** | MCDI / open-vocab class lists (`ram_tag_list*.txt`, `MCDI_items_with_AoA.csv`) | |
| **[`preprocessing/`](preprocessing/)** | Crop from bboxes, blur/size filters, sampling (`sample_object_crops_variability.py`), RSA helpers, BV validated sorting assets | |
| **[`image-embedding/`](image-embedding/)** | `create_image_embeddings.py`, `analyze_embeddings.ipynb` | |
| **[`frame_data/`](frame_data/)** | Merged frame-level detection CSVs (CLIP thresholds 0.26–0.28), sampled frames | |
| **[`depth/`](depth/)** | Depth extraction and object-depth CSV builders | |
| **[`imu/`](imu/)** | IMU alignment for short clips | |
| **[`video-qa/`](video-qa/)** | Video QA utilities (e.g. unconstrained objects) | |
| **[`variability_exp/`](variability_exp/)** | Variability experiment assets | |

---

## Human annotation & validation

| Path | Role |
|------|------|
| **[`annotation/README.md`](annotation/README.md)** | Crop annotation GUI (`annotate_crops.py`) |
| `annotation/per_class_validation_data.csv` | Per-category precision (manuscript & CCN filters) |
| `annotation/per_file_precision_data.csv` | Per-crop rater validation |
| `annotation/sampled_object_crops_*` | Exemplar sampling manifests (100 ex / 8 subjects) |

Notebook **10** and shared inputs use animal-depiction aggregates; many analyses depend on precision ≥ 0.6 from `per_class_validation_data.csv`.

---

## Other `data/` assets

| Path | Role |
|------|------|
| [`data/shared_data_manuscript_2026/`](data/shared_data_manuscript_2026/) | Public manuscript intermediates (see above) |
| `data/embeddings/`, `data/coef_data/`, `data/figures/` | Project embedding and coefficient artifacts |
| `data/things_bv_overlap_categories*.txt` | THINGS ↔ BabyView category overlap lists |
| `data/annotation_excluded_categories.txt` | Categories excluded from annotation |

---

## Repository map

```text
object-detection/
├── README.md                          ← this file
├── yoloe/                             ← detection (see yoloe/README.md)
├── preprocessing/                     ← crops, sampling, filters
├── image-embedding/                   ← CLIP/DINO-style crop embeddings
├── frame_data/                        ← frame-level detection tables
├── annotation/                        ← rater GUI + validation CSVs
├── data/
│   ├── included_categories_valid*.txt
│   └── shared_data_manuscript_2026/   ← anonymized paper tables
└── analysis/
    ├── manuscript-2026/               ← BabyView Objects (main)
    ├── preprint-2026/                 ← redirect → manuscript-2026
    ├── ccn-2026/                      ← CCN poster variability
    ├── vss-2026/                      ← group RDM pipeline
    ├── individual_analyses/           ← per-child / developmental
    └── developmental_trend_analysis_R/
```

---

## Privacy & replication tiers

| Tier | Location | Who needs it |
|------|----------|--------------|
| **A** | `data/shared_data_manuscript_2026/` | Verify aggregate manuscript statistics |
| **B** | `analysis/manuscript-2026/exemplar_set_embeddings/`, `main_results_*` | Regenerate figures from notebooks without re-embedding |
| **C** | Per-image `.npy` under cluster paths (`BV_EMBEDDINGS_BASE`, see `paths.example.env`) | Rebuild embeddings (notebooks 06–07) |
| **D** | Raw YOLOE outputs / BabyView video | Notebook 01 from scratch, individual RDMs (04) |

Do not commit `paths.local.env`, raw video paths, or identifiable participant IDs. One subject is excluded from embedding analyses (see [`REPRODUCTION.md`](analysis/manuscript-2026/REPRODUCTION.md)).
