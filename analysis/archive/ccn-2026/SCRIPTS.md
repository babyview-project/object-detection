# CCN 2026 — script reference

## Pipeline

```bash
conda activate ccn-valid7018   # see environment.yml
bash analysis/ccn-2026/run_valid7018_from_zip.sh
```

| Script / entry | Role |
|----------------|------|
| [`run_valid7018_from_zip.sh`](run_valid7018_from_zip.sh) | One-command regen (metrics + figures) |
| [`verify_valid7018_reproduction.py`](scripts/verify_valid7018_reproduction.py) | Smoke-test vs committed CSVs (CI) |
| [`compute_valid7018_local_global.py`](scripts/compute_valid7018_local_global.py) | Global + local metrics on 7,018 crops |
| [`generate_valid7018_paper_figures.py`](scripts/generate_valid7018_paper_figures.py) | Montage + frequency + scatter panels → `abstract_figures/` |
| [`valid7018_category_metrics.py`](scripts/valid7018_category_metrics.py) | Shared metric computation (imported) |
| [`build_shared_public_data_ccn.py`](scripts/build_shared_public_data_ccn.py) | Export `data/shared_data_ccn_2026/` |

## Maintainer-only (cluster paths)

| Script | Role |
|--------|------|
| [`build_valid7018_embeddings_zip.py`](scripts/build_valid7018_embeddings_zip.py) | Package CLIP+DINO `.npy` for git |
| [`build_valid7018_montage_crops_zip.py`](scripts/build_valid7018_montage_crops_zip.py) | Package montage JPEGs for git |

## Shared modules

| Module | Role |
|--------|------|
| [`load_valid7018_embeddings.py`](load_valid7018_embeddings.py) | Load embedding zip |
| [`load_valid7018_montage_crops.py`](load_valid7018_montage_crops.py) | Load montage JPEG zip |
