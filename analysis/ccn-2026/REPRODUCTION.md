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

```bash
python analysis/ccn-2026/scripts/build_valid7018_embeddings_zip.py
python analysis/ccn-2026/scripts/build_valid7018_montage_crops_zip.py
python analysis/ccn-2026/scripts/build_shared_public_data_ccn.py
```
