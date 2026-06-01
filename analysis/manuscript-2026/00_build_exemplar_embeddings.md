# Stage 0 — Build category embedding tables

**Run this before notebooks 02–05** (and before 03/05-style scripts that read
`exemplar_set_embeddings/`). Notebook **01** (detection frequencies) does **not**
depend on this stage and can run in parallel.

Outputs land in `exemplar_set_embeddings/{valid129,valid85}/` — the same tables
copied to [`data/shared_data_manuscript_2026/embeddings/`](../../data/shared_data_manuscript_2026/embeddings/) for
public release.

## Pipeline overview

```text
Stage 0 (this doc)          Stage 1+ (manuscript analyses)
──────────────────          ──────────────────────────────
06  BabyView aggregates  →  02  category cosine
07  THINGS CLIP+DINOv3   →  03  RDM comparison
    (+ scripts below)    →  05  CDI cluster geometry
                           →  01  long-tail (parallel OK)
                           →  04, 08–09  per-child (needs per-image .npy)
```

| Step | What | Notebook / script | Typical runtime |
|------|------|-------------------|-----------------|
| **0a** | BabyView category means (z-scored) | [`06_exemplar_set_zscore_embeddings.ipynb`](06_exemplar_set_zscore_embeddings.ipynb) or [`scripts/exemplar_set_zscore_embeddings.py`](scripts/exemplar_set_zscore_embeddings.py) | **Long** for valid129 (CLIP filter list scan) |
| **0b** | THINGS CLIP + DINOv3 category means | [`07_things_exemplar_set_zscore_embeddings.ipynb`](07_things_exemplar_set_zscore_embeddings.ipynb) | Moderate |
| **0c** | THINGS BabyDINOv3 per-image `.npy` (GPU) | [`scripts/create_babydinov3_things_embeddings.py`](scripts/create_babydinov3_things_embeddings.py) | GPU; run once per checkpoint |
| **0d** | THINGS BabyDINOv3 category means | [`scripts/things_exemplar_set_zscore_babydinov3.py`](scripts/things_exemplar_set_zscore_babydinov3.py) | Fast |
| **0e** | Sanity check | [`scripts/check_exemplar_stage.py`](scripts/check_exemplar_stage.py) | Seconds |

Keep **06 and 07 as separate notebooks** — different data roots, exemplar selection
rules, and runtimes. Use this file (or [`run_exemplar_embedding_stage.sh`](run_exemplar_embedding_stage.sh)) as the single checklist.

For other helpers (binary template, UMAP, top-8 script, etc.), see **[SCRIPTS.md](SCRIPTS.md)** (`scripts/`).

## Environment

```bash
cd analysis/manuscript-2026
pip install -r requirements-manuscript.txt
# Optional: cp paths.example.env paths.local.env && set -a && source paths.local.env && set +a
```

Key variables: see [`paths.example.env`](paths.example.env) (`BV_EMBEDDINGS_BASE`,
`BV_CLIP_FILTER_LIST`, `THINGS_CLIP_NPY_DIR`, `THINGS_DINOV3_EMBEDDINGS_DIR`, …).

## Quick start (valid129, main text)

### Option A — Shell driver (headless where possible)

```bash
cd analysis/manuscript-2026
./run_exemplar_embedding_stage.sh valid129
```

This runs BabyView z-scoring via Python, prints instructions for notebook **07**,
optionally runs BabyDINOv3 THINGS steps, then runs `scripts/check_exemplar_stage.py`.

### Option B — Manual

```bash
cd analysis/manuscript-2026

# 0a — BabyView (tmux-friendly)
./run_06_zscore_tmux.sh valid129
# or: BV_CATEGORY_SET=valid129 python scripts/exemplar_set_zscore_embeddings.py

# 0b — THINGS CLIP + DINOv3 (interactive notebook; set CATEGORY_SETS = ["valid129"])
jupyter notebook 07_things_exemplar_set_zscore_embeddings.ipynb

# 0c–0d — THINGS BabyDINOv3 (if not already on disk)
./run_babydinov3_things_embed_tmux.sh   # GPU embed ~1.8k images
BV_CATEGORY_SET=valid129 python scripts/things_exemplar_set_zscore_babydinov3.py

# 0e — verify
python scripts/check_exemplar_stage.py --category-set valid129
```

### Supplement (valid85)

Repeat **0a** with `valid85` and **0b** with `CATEGORY_SETS = ["valid85"]` (or both sets in **07**), then:

```bash
BV_CATEGORY_SET=valid85 python scripts/things_exemplar_set_zscore_babydinov3.py
python scripts/check_exemplar_stage.py --category-set valid85
```

Or: `./run_exemplar_embedding_stage.sh all` for both category sets (BV via `--all-category-sets`).

## Expected files (valid129, notebooks 02–05)

Under `exemplar_set_embeddings/valid129/`:

- `bv_{clip,dinov3,babydinov3}_exemplar_avg_zscore_within_valid129.csv`
- `things_{clip,dinov3,babydinov3}_exemplar_avg_zscore_within_valid129.csv`
- `exemplar_embedding_run.json`, `things_exemplar_embedding_run.json`

If tables already exist from a prior run or from [`data/shared_data_manuscript_2026/`](../../data/shared_data_manuscript_2026/), you can **copy** them into `exemplar_set_embeddings/` and skip Stage 0.

## Using shared config in notebooks

```python
from manuscript_config import (
    MANUSCRIPT_DIR,
    EXEMPLAR_EMBED_ROOT,
    exemplar_embed_dir,
    bv_embedding_csv,
    things_embedding_csv,
)

CATEGORY_SET = "valid129"
embed_dir = exemplar_embed_dir(CATEGORY_SET)
bv_clip = bv_embedding_csv("clip", CATEGORY_SET)
```

## Next steps

After `scripts/check_exemplar_stage.py` succeeds:

1. **01** — long-tail / prevalence (optional order; independent of embeddings)
2. **02 → 03 → 05** — main BV vs THINGS analyses
3. See [`REPRODUCTION.md`](REPRODUCTION.md) for the full manuscript workflow
