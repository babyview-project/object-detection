# CCN 2026 scripts (valid7018)

Run from repo root (`object-detection/`) with **`ccn-valid7018`** (`../environment.yml`) or **`vislearnlabpy`**.

## Clone-safe

```bash
conda activate ccn-valid7018
bash analysis/ccn-2026/run_valid7018_from_zip.sh
```

| Script | Output |
|--------|--------|
| `run_valid7018_from_zip.sh` | Metrics + abstract panels (one command) |
| `verify_valid7018_reproduction.py` | Pass/fail vs committed reference (no overwrite) |
| `compute_valid7018_local_global.py` | `valid7018/` (metrics) |
| `generate_valid7018_paper_figures.py` | `valid7018/figures/` (scratch) → `abstract_figures/` (committed panels) |
| `build_shared_public_data_ccn.py` | `data/shared_data_ccn_2026/` |

## Maintainer (cluster crop/embedding paths)

```bash
python analysis/ccn-2026/scripts/build_valid7018_embeddings_zip.py
python analysis/ccn-2026/scripts/build_valid7018_montage_crops_zip.py
python analysis/ccn-2026/scripts/build_shared_public_data_ccn.py
```

See [`../SCRIPTS.md`](../SCRIPTS.md) and [`../REPRODUCTION.md`](../REPRODUCTION.md).
