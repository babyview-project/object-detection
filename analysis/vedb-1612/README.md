# VEDB 1612 — world-camera subset

Preprocess and analyze the **80-session** VEDB world-camera subset the same way
the manuscript processes BabyView: 1 fps frames, YOLOE with the CDI noun list,
CLIP filter at 0.27, then DINOv3 on the boxes that pass.

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

### 4. Analyses

Write notebooks and tables under this folder (`results/`, `figures/`). The
manuscript notebooks that transfer are the pooled ones:

| Manuscript | VEDB analogue |
|------------|----------------|
| 01 long-tailed detection frequencies | category counts from the CDI CSVs |
| 02 category cosine vs THINGS | category means from `vedb_full` shards |
| 03 RDM comparison | VEDB RDM vs the existing THINGS RDM |
| 05 within- vs between-category geometry | pooled crops, not per child |

Per-child notebooks (04, 08–09) do not apply: these are adult headcam sessions,
not BabyView children.

### 5. BV vs VEDB additions (October 2026)

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
