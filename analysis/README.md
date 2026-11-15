# analysis/

One folder per paper, poster or analysis line. **Start with `manuscript-2026/`**; the rest
either feed it or are earlier / side projects.

| Folder | Status | What it is | Entry point |
|---|---|---|---|
| [`manuscript-2026/`](manuscript-2026/) | **active** | *BabyView Objects* manuscript: long tail, category geometry, BV vs THINGS, per-child RDMs, top-8 children. The revision keeps the YOLOE analyses on 2025.2 at 1 fps (`*_2025.2`, SI); the 868-hour `*_04302026` and 5-s `*_2025.2_5s` trees were removed (git history). | [`README.md`](manuscript-2026/README.md), [`REPRODUCTION.md`](manuscript-2026/REPRODUCTION.md), [`SCRIPTS.md`](manuscript-2026/SCRIPTS.md) |
| [`comparison-datasets/`](comparison-datasets/) | **active** | ImageNet-1k, LVIS v1, THINGS, VEDB vs BabyView; context and crop-quality controls (notebooks 11–16) | [`README.md`](comparison-datasets/README.md) |
| [`qwen-open-vocab/`](qwen-open-vocab/) | **active** | Manuscript analyses on the full 2025.2 Qwen3.8-27B run with Qwen's open categories (844, beyond the 129): long tail, BV RDMs and domain clustering, per-child RDMs, BV vs THINGS on 515 shared concepts; plus the 129-category Qwen vs YOLOE + CLIP replication | [`README.md`](qwen-open-vocab/README.md) |
| [`vedb-1612/`](vedb-1612/) | **active** | VEDB 80-session world subset through the same frame / YOLOE / CLIP / DINOv3 pipeline | [`README.md`](vedb-1612/README.md) |
| [`archive/`](archive/) | archived | Finished projects: `ccn-2025/`, `ccn-2026/` (the Code Ocean export still copies `ccn-2026/valid7018/yoloe_vqa_correlation*`) | [`README.md`](archive/README.md) |
| `talk-figures/unr-2026/` | local only (untracked) | Figure helpers the manuscript scripts import (`fig_qwen_talk.py`, `talk_style.py`, `make_talk_montages.py`, `fonts/`) | – |

Removed on 2026-11-15 (still in git history): `individual_analyses/` (its top-8 trajectory table is now `data/top8_trajectory_correlations_clip.csv`), `vss-2026/`, `developmental_trend_analysis_R/`, `manuscript-2026/not_in_manuscript/`.

## Data releases and output tags

| Release | Frames | Embedding store | Output folders |
|---|---|---|---|
| 868-hour | 1 fps, `/ccn2/dataset/babyview/outputs_20250312/` (per-crop `.npy` trees were on `/data2`, now gone) | legacy `.npy` | `*_04302026` (removed; git history) |
| 2025.2 | 1 fps, `/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/` | packed shards + `crops_index.csv.gz` under `/ccn2a/.../outputs/image_embeddings/babyview_full/` | `*_2025.2` |

`manuscript_config.DEFAULT_OUTPUT_TAG` is `2025.2`. Frame subsampling (`BV_FRAME_STRIDE`) and
`manuscript-2026/scripts/compare_output_tags.py` still work, but the 5-s outputs were removed.
