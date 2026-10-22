# frame_data

Frame-level detection tables for the **868-hour** release.

| File | Status |
|---|---|
| `preprocessed_object_detections.csv`, `sampled_100k_frames.csv` | tracked here |
| `merged_frame_detections_with_metadata_filtered-0.{26,27,28}.csv` | **gone**: these were symlinks to `/data2/dataset/babyview/868_hours/outputs/j7yang/frame_data/`, and `/data2` no longer exists (links removed 2026-10-22) |

The merged tables (one row per CLIP-passing YOLOE box, at CLIP cut-offs 0.26 / 0.27 /
0.28) are still named by 868-hour code paths, e.g. `analysis/manuscript-2026/scripts/frame_prevalence.py`
(868-hour branch), `analysis/ccn-2025/`, `analysis/individual_analyses/12_*` and several
`analysis/manuscript-2026/not_in_manuscript/` notebooks. Those paths fail until the tables
are restored from a backup.

For **2025.2**, the equivalent is the packed crop index
`/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full/crops_index.csv.gz`
(`frame_prevalence.load_detections_from_packed_index`), selected automatically when
`BV_EMBEDDINGS_BASE` points at that store.
