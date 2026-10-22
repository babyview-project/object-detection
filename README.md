# BabyView object detection

Code to detect, recognize, and label objects across BabyView videos. Models run on
frames extracted from video at **1 fps**. Open-vocabulary detection uses class names
from the child vocabulary MCDI survey (word lists in [`yoloe/tools/`](yoloe/tools/)).

For YOLOE setup, frame extraction, and batch prediction, see **[`yoloe/README.md`](yoloe/README.md)**.
Folder-level indexes: **[`analysis/README.md`](analysis/README.md)** (which analysis folder is
active, legacy or archived, and which data release / output tag each uses) and
**[`annotation/README.md`](annotation/README.md)** (human annotation and Gemini labelling).

---

## BabyView Objects manuscript (2026)

Primary analyses, figures, and **anonymized shareable tables** for the *BabyView Objects*
manuscript (June 2026 submission) live under **[`analysis/manuscript-2026/`](analysis/manuscript-2026/)**
(the former `preprint-2026` tree was merged into it).

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
| `…/main_results_valid129s_2025.2/`, `…/supplemental_results_valid85cats_2025.2/` | Same analyses on the 2025.2 release (1 fps): `scripts/run_2025_2_analyses.sh` |
| `…/main_results_valid129s_2025.2_5s/`, `…/supplemental_results_valid85cats_2025.2_5s/` | 2025.2 at one frame per 5 s (`BV_FRAME_STRIDE=5`, `logs/run_5s.sh`); `results/comparison_vs_1s/` compares it with 1 fps |

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
| **[`analysis/vedb-1612/`](analysis/vedb-1612/)** | VEDB 80-session world subset — same frame / YOLOE / CLIP / DINOv3 pipeline | [`README.md`](analysis/vedb-1612/README.md) |
| **[`analysis/comparison-datasets/`](analysis/comparison-datasets/)** | ImageNet-1k and LVIS v1 as comparison datasets (WordNet mapping onto CDI words, BV vs ImageNet / LVIS / THINGS / VEDB) | [`README.md`](analysis/comparison-datasets/README.md) |
| **[`analysis/developmental_trend_analysis_R/`](analysis/developmental_trend_analysis_R/)** | R / Quarto developmental stats | [`README.md`](analysis/developmental_trend_analysis_R/README.md) |
| **[`analysis/archive/`](analysis/archive/)** | Finished work (CCN 2025) | [`README.md`](analysis/archive/README.md) |

CCN variability code also appears under [`analysis/manuscript-2026/not_in_manuscript/exemplar_variability_analyses/`](analysis/manuscript-2026/not_in_manuscript/exemplar_variability_analyses/).

---

## Detection & preprocessing pipeline

End-to-end flow: **frames → YOLOE detections → crops → embeddings → manuscript notebooks**.

| Directory | Role |
|-----------|------|
| **[`yoloe/`](yoloe/)** | YOLOE install, `extract_frames.py`, `predict_frames.py`, examples | [`yoloe/README.md`](yoloe/README.md) |
| **`yoloe/tools/`** | MCDI / open-vocab class lists (`ram_tag_list*.txt`, `MCDI_items_with_AoA.csv`) | |
| **[`preprocessing/`](preprocessing/)** | Crop from bboxes, blur/size filters, sampling (`sample_object_crops_variability.py`), RSA helpers, BV validated sorting assets | |
| **[`image-embedding/`](image-embedding/)** | 2025.2 crop + CLIP / DINOv3 pipeline (`run_cdi_crop_pipeline.py`, packed shards), older `create_image_embeddings.py` | [`README_embeddings.md`](image-embedding/README_embeddings.md) |
| **[`frame_data/`](frame_data/)** | 868-hour frame tables; the merged CLIP-filtered tables were `/data2` links, now gone | [`README.md`](frame_data/README.md) |
| **[`depth/`](depth/)** | Depth extraction and object-depth CSV builders | [`README.md`](depth/README.md) |
| **[`imu/`](imu/)** | IMU alignment for short clips | [`README.md`](imu/README.md) |
| **[`video-qa/`](video-qa/)** | Video QA utilities (e.g. unconstrained objects) | [`README.md`](video-qa/README.md) |
| **[`variability_exp/`](variability_exp/)** | Perceptual free-sort experiment assets (untracked; crop links removed) | [`README.md`](variability_exp/README.md) |
| **[`databrary/`](databrary/)** | Databrary API client for downloading BabyView data | [`README.md`](databrary/README.md) |

---

## Human annotation & validation

| Path | Role |
|------|------|
| **[`annotation/README.md`](annotation/README.md)** | Index of annotation studies, shared label files, crop GUI |
| [`annotation/precision_868h/`](annotation/precision_868h/) | Prolific crop validation: `per_class_validation_data.csv` (per-category precision, manuscript & CCN filters), `per_file_precision_data.csv`, exemplar sampling manifests |
| [`annotation/cvat_3000_frames/`](annotation/cvat_3000_frames/) | Mira / Dora CVAT boxes and format tags on 3,000 sampled frames; animal-depiction labels |
| [`annotation/gemini_pilot/`](annotation/gemini_pilot/), [`annotation/gemini_full_run/`](annotation/gemini_full_run/) | Gemini (Vertex) model pilot and the full 2025.2 run |
| [`annotation/clip_band_annotation/`](annotation/clip_band_annotation/), [`annotation/gold_set/`](annotation/gold_set/) | CLIP-threshold RA sample (object-check web task) and gold set v2 |

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
├── frame_data/                        ← 868-hour frame-level detection tables
├── annotation/                        ← human annotation + Gemini labelling (see its README)
│   ├── precision_868h/                ← Prolific crop validation (precision CSVs)
│   ├── cvat_3000_frames/              ← CVAT boxes + format tags
│   ├── gemini_pilot/  gemini_full_run/
│   └── clip_band_annotation/  gold_set/
├── data/
│   ├── included_categories_valid*.txt
│   └── shared_data_manuscript_2026/   ← anonymized paper tables
├── depth/  imu/  video-qa/  databrary/  variability_exp/
└── analysis/                          ← see analysis/README.md
    ├── manuscript-2026/               ← BabyView Objects (main)
    ├── ccn-2026/                      ← CCN poster variability
    ├── comparison-datasets/           ← ImageNet-1k + LVIS v1 vs BabyView
    ├── vedb-1612/                     ← VEDB world subset (80 sessions)
    ├── individual_analyses/           ← per-child / developmental (legacy, still read)
    ├── vss-2026/                      ← group RDM pipeline (legacy, still read)
    ├── developmental_trend_analysis_R/
    └── archive/                       ← ccn-2025
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

Frames and crops show children's homes and family members: they never go into git, Slack or
shared notebooks. Annotation folders keep images, subject / video ids and per-frame outputs
in git-ignored `out/private/` folders. BabyView images go to Gemini only through Vertex AI on
the lab grant project (never AI Studio or API keys). The old `/data2` disk is gone; links and
paths into it have been removed or are documented where they still appear in legacy code.
