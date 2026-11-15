# analysis/

One folder per analysis line. **For the revision, start with `qwen-open-vocab/`** (main text);
`manuscript-2026/` is the original YOLOE + CLIP pipeline, kept for SI 1.

| Folder | Status | What it is | Entry point |
|---|---|---|---|
| [`qwen-open-vocab/`](qwen-open-vocab/) | **active (main text)** | The revision's analyses on the full 2025.2 Qwen3.8-27B open-vocabulary run (renamed labels; 183 → 182 core categories after VEDB minors were excluded): long tail, formats, BabyView vs VEDB / THINGS / LVIS / ImageNet, domain clustering (CDI or THINGS-27 domains, `QOV_DOMAINS`), per-child and within-child analyses, controls, and the manuscript and response-letter figures | [`README.md`](qwen-open-vocab/README.md) |
| [`manuscript-2026/`](manuscript-2026/) | **active (SI 1)** | Original YOLOE + CLIP pipeline on 129 CDI categories, on the 2025.2 release at 1 fps (`*_2025.2`) and one frame per 5 s (`*_2025.2_5s`). The git-ignored `revision_private/` holds the manuscript and response-letter figures | [`README.md`](manuscript-2026/README.md), [`REPRODUCTION.md`](manuscript-2026/REPRODUCTION.md), [`SCRIPTS.md`](manuscript-2026/SCRIPTS.md) |
| [`comparison-datasets/`](comparison-datasets/) | **active (inputs)** | ImageNet-1k and LVIS v1 data prep and category tables (read by `qwen-open-vocab/`), and notebook 16 (crop size / CLIP-filter controls on LVIS, Response Figure 1) | [`README.md`](comparison-datasets/README.md) |
| [`vedb-1612/`](vedb-1612/) | **active (inputs)** | VEDB download, frames, YOLOE / CLIP / DINOv3 stores, the adult-session list and adults-only store builder | [`README.md`](vedb-1612/README.md) |
| [`talk-figures/unr-2026/`](talk-figures/unr-2026/) | **active (helpers)** | What remains of the UNR talk folder: figure code the manuscript scripts import, Lato fonts, and the montages copied into the manuscript | [`README.md`](talk-figures/unr-2026/README.md) |
| [`archive/`](archive/) | archived | Finished or superseded: `ccn-2025/`, `ccn-2026/`, the 129-category YOLOE comparisons (`comparison-datasets/`) and BabyView–VEDB analyses (`vedb-1612/`) | [`README.md`](archive/README.md) |

Removed on 2026-11-15 (still in git history): `individual_analyses/` (its top-8 trajectory table is now
`data/top8_trajectory_correlations_clip.csv`), `vss-2026/`, `developmental_trend_analysis_R/`,
`manuscript-2026/not_in_manuscript/` and `online_annotation/`, and the talk decks and slide figures.

## Data releases and output tags (`manuscript-2026/`)

| Release | Frames | Embedding store | Output folders |
|---|---|---|---|
| 868-hour (original submission) | 1 fps, `/ccn2/dataset/babyview/outputs_20250312/` (per-crop `.npy` trees were on `/data2`, now gone) | legacy `.npy` | `*_04302026` (removed; result tables in git history) |
| 2025.2 | 1 fps, `/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/` | packed shards + `crops_index.csv.gz` under `/ccn2a/.../outputs/image_embeddings/babyview_full/` | `*_2025.2` |
| 2025.2, one frame per 5 s | the same store, filtered at load time (`BV_FRAME_STRIDE=5`, `frame_number % 5 == 0`) | same | `*_2025.2_5s` |

`manuscript_config.DEFAULT_OUTPUT_TAG` is `2025.2`. The 5-s run is launched with
`manuscript-2026/scripts/run_5s_analyses.sh`; `manuscript-2026/scripts/compare_output_tags.py` compares any two tags.
The Qwen pipeline in `qwen-open-vocab/` samples one frame every 5 s throughout.
