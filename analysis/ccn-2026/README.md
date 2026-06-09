# CCN 2026 — exemplar variability (valid7018 cohort)

Analysis for the **CCN 2026** extended abstract: global and local within-category dispersion on **7,018** rater-validated BabyView object crops (85 CDI noun categories), using paired CLIP and DINOv3 embeddings.

## Output layout

| Path | Contents | In git? |
|------|----------|---------|
| [`valid7018/`](valid7018/) | Category-level CSVs + `valid7018_paper_stats.json` | yes |
| [`valid7018/figures/`](valid7018/figures/) | Scratch figure regen (all panel variants) | no (gitignored) |
| [`abstract_figures/`](abstract_figures/) | Selected panels for manual abstract layout | yes |
| [`data/shared_data_ccn_2026/`](../../data/shared_data_ccn_2026/) | Public bundle (embeddings, montage JPEGs, frequency tables) | yes |

The old `plotC_knn_diversity_outputs/` name was legacy from an earlier notebook pipeline and has been removed.

## Quick start (clone-safe)

```bash
conda env create -f analysis/ccn-2026/environment.yml   # once; or use vislearnlabpy
conda activate ccn-valid7018
bash analysis/ccn-2026/run_valid7018_from_zip.sh
```

Verify without overwriting committed tables:

```bash
python analysis/ccn-2026/scripts/verify_valid7018_reproduction.py --with-figures
```

Manuscript / Overleaf sources (`447_Variability_in_Young_Child*`) stay **gitignored** — only analysis code, metrics, and abstract panels are public in this repo.

See [`SCRIPTS.md`](SCRIPTS.md), [`OUTPUTS.md`](OUTPUTS.md), and [`REPRODUCTION.md`](REPRODUCTION.md).
