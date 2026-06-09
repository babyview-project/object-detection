# CCN 2026 — reproducibility (valid7018)

## Environment

```bash
conda env create -f analysis/ccn-2026/environment.yml
conda activate ccn-valid7018
```

Or use the lab `vislearnlabpy` env. Pip-only: `pip install -r analysis/ccn-2026/requirements.txt`.

## Clone-safe (one command)

```bash
bash analysis/ccn-2026/run_valid7018_from_zip.sh
```

Writes metrics to `valid7018/`, scratch figures to `valid7018/figures/`, and copies selected panels to `abstract_figures/`.

## Verify (CI uses this; does not overwrite `valid7018/`)

```bash
python analysis/ccn-2026/scripts/verify_valid7018_reproduction.py
python analysis/ccn-2026/scripts/verify_valid7018_reproduction.py --with-figures
```

Checks Spearman ρ, category counts (7,018 exemplars / 85 categories), and per-category metrics against committed CSVs. GitHub Actions: `.github/workflows/ccn-valid7018.yml`.

**Private:** `447_Variability_in_Young_Child*` (Overleaf export) is gitignored.

## Maintainer (cluster paths)

Embeddings live under ``/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings``
(override with ``BV_EMBEDDINGS_BASE``; see ``paths.example.env``). Per-crop vectors
from ``clip_embeddings_new`` and ``facebook_dinov3-vitb16-pretrain-lvd1689m`` are
feature-wise globally normalized using μ/σ fit from grouped age-month dirs
(notebook 05; ``valid7018_embedding_normalize.py``).

```bash
export BV_EMBEDDINGS_BASE=/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings
python analysis/ccn-2026/scripts/build_valid7018_embeddings_zip.py
python analysis/ccn-2026/scripts/build_valid7018_montage_crops_zip.py
python analysis/ccn-2026/scripts/compute_valid7018_local_global.py
python analysis/ccn-2026/scripts/generate_valid7018_paper_figures.py
python analysis/ccn-2026/scripts/build_shared_public_data_ccn.py
```
