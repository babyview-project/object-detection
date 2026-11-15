# VEDB 1612 — world-camera subset

> **2026-11-15:** the YOLOE + CLIP BabyView–VEDB analyses (`analyses/`, `figures/`, `run_analyses.py`,
> `bv_vedb_comparison.py`, `plot_*.py`) moved to `analysis/archive/vedb-1612/`. Kept here: download and frame
> extraction, the VEDB YOLOE / embedding scripts and `exemplar_set_embeddings/` (read by comparison-datasets
> notebook 16), the adult-session list and adults-only store builder used by `qwen-open-vocab/`.

Preprocess the VEDB world-camera videos (volume 1612; first 80 sessions, then all 717, of which the 709 adult
sessions are analyzed) the same way the manuscript processes BabyView: 1 fps frames, YOLOE with the CDI noun list,
CLIP filter at 0.27, then DINOv3 on the boxes that pass. The revision's Qwen detections of VEDB are run from
`annotation/qwen_pilot/` (`run_vedb.sh`) and analyzed in `analysis/qwen-open-vocab/`.

This folder is the analysis home (tables and figures). The mp4s and the heavy
outputs stay on `/ccn2a`.

## Data

| What | Path |
|------|------|
| World videos (80 mp4s, 178 GB) | `/ccn2a/dataset/babyview/related_data/vedb_1612_world` |
| Session manifest | `.../vedb_1612_world/world_subset_manifest.json` |
| 1 fps frames | `/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps` |
| YOLOE CDI detections | `.../vedb_1612/outputs/object_detections/cdi` |
| CLIP + DINOv3 shards | `.../vedb_1612/outputs/image_embeddings/vedb_full` |
| CLIP-passer frame tars | `.../vedb_1612/extracted_frames_1fps_annotated` |

Paths are also in [`vedb_config.py`](vedb_config.py). The subset is one
`worldPrivate.mp4` per session, sampled across settings (kitchen, grocery,
campus, and so on). Eye videos are not included.

VEDB session folders are named `session_<id>_worldPrivate`. They are not
BabyView subject IDs, so the manuscript subject exclusion (`00270001`) does
not apply. Pass `--exclude-subjects` with an empty list.

> **Scope update.** The pipeline was later run on all **717** sessions of
> volume 1612 (the 80-session manifest above was the first batch), so the
> stores hold 717 sessions from 56 participants.

## Adults only (November 2026)

Our analyses use **adult** VEDB observers only: 18 years or older at recording.
That is 709 of 717 sessions and 50 of 56 participants, aged 18.7–46.4. The 8
excluded sessions come from 6 participants aged 7.7–12.5. No one is between
12.5 and 18, so any cutoff from 13 to 18 removes the same sessions.

- Ages come from the participant record of each Databrary session.
  `build_adult_stores.py --fetch` pulls them (after `python -m databrary login`)
  into `/ccn2a/dataset/babyview/related_data/vedb_1612/participant_ages.csv`.
  That file stays out of git, because the repo is public and ages are
  authorized-user data.
- [`vedb_adult_sessions.txt`](vedb_adult_sessions.txt) lists the included
  session ids.
- Every analysis reads an **`_adults` view** of each store: the same files,
  with shards and session folders symlinked for adult sessions and index
  tables filtered to them. The views are `vedb_full_adults`, `cdi_adults`,
  `extracted_frames_1fps_adults`, and, on `/data2`,
  `qwen3.8-27b_setupB_t1_pixbv_adults`. The original stores are unchanged; the
  preprocessing scripts still write to them. To rebuild the views, run
  `python build_adult_stores.py`.
- The BabyView/VEDB embedding sample and its SAM 3 outputs are referenced by
  row number, so they are filtered rather than redrawn
  (`../qwen-open-vocab/filter_bv_vedb_sample_adults.py`; SAM 3 →
  `sam3_crop_sample_bv_vedb_adults`). Do not rerun
  `annotation/qwen_pilot/sam3_crop_sample.py` or `embed_masked_sample.py`
  against the filtered sample, because their `idx` refers to the original rows.
- [`rerun_adults_only.sh`](rerun_adults_only.sh) reruns every VEDB-dependent
  analysis and figure: YOLOE (SI), Qwen (main text, CDI and THINGS-27
  domains), the talk figures and the manuscript figures. The pre-exclusion
  outputs are in
  `/data2/j7yang/object-detection-offload/backups/pre_vedb_adults_20261115/`.

## Pipeline

Same stages as BabyView 2025.2. Run from the repo root, in tmux.

### 1. Frames at 1 fps

```bash
cd yoloe
python extract_frames.py \
  --videos_dir /ccn2a/dataset/babyview/related_data/vedb_1612_world \
  --output_dir /ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps \
  --fps 1 --num_processes 8
```

About 27 hours of video, so on the order of 100k JPEGs. That is small next to
the BabyView 2025.2 frame tree.

### 2. YOLOE detections (CDI nouns)

```bash
cd yoloe
python predict_frames.py \
  --input_dir /ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps \
  --output_dir /ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi \
  --text_prompts cdi \
  --confidence 0.1 \
  --device_ids "[0]" \
  --no-tmux
```

Each video should end as `<video_id>/bounding_box_predictions.csv`, which is
what the crop embedder reads.

### 3. CLIP filter, then DINOv3

Same models as the manuscript: CLIP ViT-B/32, DINOv3 ViT-B/16, cosine ≥ 0.27
against `a photo of a {label}`. Crops stay in memory; one `.npz` shard per video.

```bash
cd image-embedding
python run_cdi_crop_pipeline.py --stage scan \
  --frames-root /ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps \
  --cdi-root /ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi \
  --emb-root /ccn2a/dataset/babyview/related_data/vedb_1612/outputs/image_embeddings/vedb_full \
  --annotated-root /ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps_annotated \
  --exclude-subjects

python run_cdi_crop_pipeline.py --stage embed \
  --frames-root /ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps \
  --cdi-root /ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi \
  --emb-root /ccn2a/dataset/babyview/related_data/vedb_1612/outputs/image_embeddings/vedb_full \
  --annotated-root /ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps_annotated \
  --exclude-subjects

python run_cdi_crop_pipeline.py --stage index \
  --frames-root /ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps \
  --cdi-root /ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi \
  --emb-root /ccn2a/dataset/babyview/related_data/vedb_1612/outputs/image_embeddings/vedb_full \
  --annotated-root /ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps_annotated \
  --exclude-subjects
```

### 4. Analyses (archived 2026-11-15: `../archive/vedb-1612/`)

The YOLOE + CLIP BabyView–VEDB analyses below now live in `../archive/vedb-1612/`; the revision's BabyView–VEDB
comparisons are in `../qwen-open-vocab/` (`bv_vedb_analysis.ipynb`). The manuscript notebooks that transferred were
the pooled ones:

| Manuscript | VEDB analogue |
|------------|----------------|
| 01 long-tailed detection frequencies | category counts from the CDI CSVs |
| 02 category cosine vs THINGS | category means from `vedb_full` shards |
| 03 RDM comparison | VEDB RDM vs the existing THINGS RDM |
| 05 within- vs between-category geometry | pooled crops, not per child |

Per-child notebooks (04, 08–09) do not apply: these are adult headcam sessions,
not BabyView children.

### 5. BV vs VEDB additions (October 2026; archived)

- `plot_bv_vedb_rdms.py` also writes RDMs clustered within each CDI domain on the BV or the VEDB RDM:
  `analyses/{valid129,valid85}/figures/bv_vedb_rdm_comparison_<model>_clustered_{bv,vedb}_*.png`,
  with the category orders in `analyses/*/results/bv_vedb_rdm_order_clustered_*.csv`.
- `plot_bv_vedb_category_montage.py` also writes a montage of every CDI animal category:
  `analyses/{valid129,valid85}/figures/bv_vedb_category_wise_cosine_montage_animals_*`.
- Three-way BV / VEDB / ImageNet analyses and the cross-dataset RDM correlation matrices (with THINGS and
  LVIS) live in [`../comparison-datasets/`](../comparison-datasets/README.md).
- `plot_bv_vedb_cluster_delta_bars.py` tests VEDB vs BV cluster strength against a label-shuffle null on raw
  cosine-distance Δ. See the review notes in `../comparison-datasets/README.md`: that null removes structure
  from both datasets, and RDMs differ in spread, so the scaled Δ with a category bootstrap there is the
  better test.
