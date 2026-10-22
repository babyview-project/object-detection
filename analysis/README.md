# analysis/

One folder per paper, poster or analysis line. **Start with `manuscript-2026/`**; the rest
either feed it or are earlier / side projects.

| Folder | Status | What it is | Entry point |
|---|---|---|---|
| [`manuscript-2026/`](manuscript-2026/) | **active** | *BabyView Objects* manuscript: long tail, category geometry, BV vs THINGS, per-child RDMs, top-8 children. Runs on the 868-hour release (`*_04302026`), 2025.2 at 1 fps (`*_2025.2`) and 2025.2 at one frame per 5 s (`*_2025.2_5s`). | [`README.md`](manuscript-2026/README.md), [`REPRODUCTION.md`](manuscript-2026/REPRODUCTION.md), [`SCRIPTS.md`](manuscript-2026/SCRIPTS.md) |
| [`ccn-2026/`](ccn-2026/) | **active** | CCN 2026 poster: exemplar variability (valid7018 crops), camera-ready figures | [`README.md`](ccn-2026/README.md) |
| [`comparison-datasets/`](comparison-datasets/) | **active** | ImageNet-1k, LVIS v1, THINGS, VEDB vs BabyView; context and crop-quality controls (notebooks 11–16) | [`README.md`](comparison-datasets/README.md) |
| [`vedb-1612/`](vedb-1612/) | **active** | VEDB 80-session world subset through the same frame / YOLOE / CLIP / DINOv3 pipeline | [`README.md`](vedb-1612/README.md) |
| [`individual_analyses/`](individual_analyses/) | legacy, still read | Per-child RDMs and developmental trajectories (868-hour). Its per-child RDM tables are read by manuscript notebooks 04, 08, 09 and the top-8 scripts; the `*_intersection_threshold_20..31` sweeps were built from 868-hour embeddings on `/data2`, which is gone, so they are kept in git (they cannot be regenerated). | [`README_notebooks_05_06_07.md`](individual_analyses/README_notebooks_05_06_07.md) |
| [`vss-2026/`](vss-2026/) | legacy, still read | VSS 2026: group RDM / embedding pipeline (CLIP, DINOv3) at CLIP 0.26–0.28. Its `bv_things_comp_*` outputs are read by manuscript notebook 10 and several `not_in_manuscript/` scripts, so it stays in place. | [`README.md`](vss-2026/README.md) |
| [`developmental_trend_analysis_R/`](developmental_trend_analysis_R/) | legacy | R / Quarto developmental statistics | [`README.md`](developmental_trend_analysis_R/README.md) |
| [`archive/`](archive/) | archived | Finished projects nothing else reads: `ccn-2025/` | [`README.md`](archive/README.md) |
| `talk-figures/` | local only (untracked) | Slides and figures for talks (UNR 2026, CCN). Not in git. | – |

## Data releases and output tags

| Release | Frames | Embedding store | Output folders |
|---|---|---|---|
| 868-hour | 1 fps, `/ccn2/dataset/babyview/outputs_20250312/` (per-crop `.npy` trees were on `/data2`, now gone) | legacy `.npy` | `*_04302026` |
| 2025.2 | 1 fps, `/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/` | packed shards + `crops_index.csv.gz` under `/ccn2a/.../outputs/image_embeddings/babyview_full/` | `*_2025.2` |
| 2025.2, one frame per 5 s | the same store, filtered at load time (`BV_FRAME_STRIDE=5`, `frame_number % 5 == 0`) | same | `*_2025.2_5s` |

The 5-s run is launched with `manuscript-2026/logs/run_5s.sh`; `manuscript-2026/scripts/compare_output_tags.py`
compares any two tags (results in `main_results_valid129s_2025.2_5s/results/comparison_vs_1s/`).
