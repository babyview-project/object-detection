# BabyView object detection

Code to detect, recognize, and label objects across BabyView videos, and the analyses of the *BabyView Objects*
manuscript. Frames are extracted at 1 fps.

Folder-level indexes: **[`analysis/README.md`](analysis/README.md)** (which analysis folder is active or
archived, and which data release / output tag each uses) and **[`annotation/README.md`](annotation/README.md)**
(human annotation, Qwen and Gemini labelling).

---

## Revision (November 2026): Qwen open-vocabulary pipeline

The revised manuscript's main text uses Qwen3.8-27B open-vocabulary detections (one frame every 5 s, renamed labels,
828 Tier 1 and 182 core categories), compared with VEDB, THINGS, LVIS and ImageNet.

| Folder | What |
|--------|------|
| **[`annotation/qwen_pilot/`](annotation/qwen_pilot/)** | Qwen runs (BabyView, VEDB), crop embeddings, SAM 3 masks; raw outputs via `out/runs` |
| **[`analysis/qwen-open-vocab/`](analysis/qwen-open-vocab/README.md)** | Every main-text analysis and figure, the response-letter figures, controls, THINGS-27 domain variant |
| [`analysis/comparison-datasets/`](analysis/comparison-datasets/README.md), [`analysis/vedb-1612/`](analysis/vedb-1612/README.md) | LVIS / ImageNet category tables; VEDB frames, stores and adult-session list |
| `analysis/manuscript-2026/revision_private/` | Manuscript and response-letter figures (git-ignored; holds BabyView frames) |

---

## Original pipeline (June 2026 submission; SI 1 of the revision)

The YOLOE + CLIP pipeline on 129 CDI categories, analyses, and **anonymized shareable tables** live under
**[`analysis/manuscript-2026/`](analysis/manuscript-2026/)**. The revision keeps its results on the 2025.2 release
(1 fps and one frame per 5 s). For YOLOE setup, frame extraction and batch prediction, see
**[`yoloe/README.md`](yoloe/README.md)**.

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
| `…/main_results_valid129s_2025.2/`, `…/supplemental_results_valid85cats_2025.2/` | YOLOE analyses on the 2025.2 release (1 fps; SI 1): `scripts/run_2025_2_analyses.sh`. The 868-hour `*_04302026` trees of the submitted version were removed (git history) |
| `…/main_results_valid129s_2025.2_5s/`, `…/supplemental_results_valid85cats_2025.2_5s/` | The same at one frame per 5 s (SI 1 sampling-rate check): `scripts/run_5s_analyses.sh` |

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

---

## Other analysis folders

| Folder | Paper / venue | Entry point |
|--------|---------------|-------------|
| **[`analysis/vedb-1612/`](analysis/vedb-1612/)** | VEDB volume 1612 (709 adult sessions): download, frames, YOLOE / CLIP / DINOv3 stores, adult-session list | [`README.md`](analysis/vedb-1612/README.md) |
| **[`analysis/comparison-datasets/`](analysis/comparison-datasets/)** | ImageNet-1k and LVIS v1: data prep, category tables, LVIS crop-size / CLIP-filter controls (notebook 16) | [`README.md`](analysis/comparison-datasets/README.md) |
| **[`analysis/talk-figures/unr-2026/`](analysis/talk-figures/unr-2026/)** | Figure helpers and fonts the manuscript figure scripts import | [`README.md`](analysis/talk-figures/unr-2026/README.md) |
| **[`analysis/archive/`](analysis/archive/)** | Finished or superseded: CCN 2025 paper, CCN 2026 poster, 129-category YOLOE comparisons with ImageNet / LVIS / VEDB | [`README.md`](analysis/archive/README.md) |

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
| **[`databrary/`](databrary/)** | Databrary API client for downloading BabyView data | [`README.md`](databrary/README.md) |

---

## Human annotation & validation

| Path | Role |
|------|------|
| **[`annotation/README.md`](annotation/README.md)** | Index of annotation studies, shared label files, crop GUI |
| [`annotation/precision_868h/`](annotation/precision_868h/) | Prolific crop validation: `per_class_validation_data.csv` (per-category precision, manuscript & CCN filters), `per_file_precision_data.csv`, exemplar sampling manifests |
| [`annotation/qwen_pilot/`](annotation/qwen_pilot/) | Qwen open-vocabulary detection runs (raw outputs via `out/runs`), crop embeddings, SAM 3 checks |
| [`annotation/gemini_full_run/`](annotation/gemini_full_run/), [`annotation/gold_set/`](annotation/gold_set/) | Full 2025.2 Gemini run; gold set v2 |
| [`annotation/archive/`](annotation/archive/) | Finished: CVAT boxes and format tags on 3,000 frames (`cvat_3000_frames/`, animal-depiction labels), Gemini model pilot (`gemini_pilot/`) |

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
│   ├── qwen_pilot/  gemini_full_run/  gold_set/
│   └── archive/                       ← cvat_3000_frames, gemini_pilot
├── data/
│   ├── included_categories_valid*.txt
│   └── shared_data_manuscript_2026/   ← anonymized paper tables
├── depth/  imu/  databrary/
└── analysis/                          ← see analysis/README.md
    ├── manuscript-2026/               ← BabyView Objects (main)
    ├── qwen-open-vocab/               ← revision main analyses (Qwen open vocabulary)
    ├── comparison-datasets/           ← ImageNet-1k + LVIS v1 vs BabyView
    ├── vedb-1612/                     ← VEDB world subset
    ├── talk-figures/unr-2026/         ← figure helpers the manuscript scripts import
    └── archive/                       ← ccn-2025, ccn-2026
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
