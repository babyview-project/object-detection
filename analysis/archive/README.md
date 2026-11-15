# analysis/archive/

Finished analyses kept for the record. Nothing in the active pipeline depends on them,
except where noted.

| Folder | What | Notes |
|---|---|---|
| [`ccn-2025/`](ccn-2025/) | CCN 2025 paper: 163-category long-tailed distribution, AoA predictions, THINGS–BV correlations (868-hour, CLIP 0.27) | Several scripts read `frame_data/merged_frame_detections_*` and `/data2` paths that no longer exist. |
| [`ccn-2026/`](ccn-2026/) | CCN 2026 poster: exemplar variability (valid7018 crops), camera-ready figures; moved here 2026-11-15 | `valid7018/yoloe_vqa_correlation*` is copied by `manuscript-2026/scripts/export_codeocean_capsule.sh`. |
| [`comparison-datasets/`](comparison-datasets/) | Notebooks 04 and 06–15 and their outputs: the 129-category YOLOE + CLIP comparisons of BabyView with ImageNet, LVIS, THINGS and VEDB (cluster strength, RDMs, context, room / activity, co-occurrence, geometry, background, prevalence), redone on the Qwen categories in `qwen-open-vocab/`; moved 2026-11-15 | Paths inside the notebooks assume `analysis/comparison-datasets/notebooks/`. |
| [`vedb-1612/`](vedb-1612/) | YOLOE + CLIP BabyView–VEDB analyses (`analyses/`, `figures/`, `run_analyses.py`, `bv_vedb_comparison.py`, `plot_*.py`), redone in `qwen-open-vocab/`; moved 2026-11-15 | |

The former `analysis/preprint-2026/` (a redirect to `manuscript-2026/`) was removed on 2026-10-22.
