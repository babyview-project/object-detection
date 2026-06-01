# CCN 2026 — Python entry points

## Poster pipeline

| Location | Role |
|----------|------|
| Notebooks **01–06** | Main CCN figures (see [`README.md`](README.md)) |
| [`load_things_embeddings.py`](load_things_embeddings.py) | **Shared module** — THINGS `.npy` loaders; imported by notebook **01** and `scripts/rerun_local_global_original_embeddings.py` |

You do **not** need anything under [`scripts/`](scripts/) for Plots A–C.

## Optional scripts (`scripts/`)

Maintenance, THINGS comparison, and side-project utilities are in **[`scripts/README.md`](scripts/README.md)**:

| Script | Summary |
|--------|---------|
| `rerun_local_global_original_embeddings.py` | Local/global metrics from raw embeddings |
| `rerun_local_global_zscore.py` | Local/global metrics from z-scored CSVs |
| `generate_things_plotabc_and_bv_comparisons.py` | THINGS analog figures + BV comparisons |
| `make_local_global_extremes_tsne_panel.py` | CLI twin of notebook **06** |
| `regenerate_plotc_vector_pdfs.py` | Vector PDFs for Plot C bar charts |
| `build_tsne_datapage.py` | Interactive HTML datapage |

## CLI reference

### `scripts/rerun_local_global_original_embeddings.py`

```bash
python analysis/ccn-2026/scripts/rerun_local_global_original_embeddings.py \
  --category-set valid129 \
  --k 5 \
  --out-dir analysis/ccn-2026/plotC_knn_diversity_outputs/original_embeddings_rerun
```

### `scripts/rerun_local_global_zscore.py`

```bash
python analysis/ccn-2026/scripts/rerun_local_global_zscore.py \
  --valid-set valid129 \
  --k 5 \
  --out-dir analysis/ccn-2026/plotC_knn_diversity_outputs/zscore_rerun
```

Inputs: `analysis/manuscript-2026/exemplar_set_embeddings/{valid_set}/` (local after Stage 0; gitignored).

### `scripts/generate_things_plotabc_and_bv_comparisons.py`

```bash
python analysis/ccn-2026/scripts/generate_things_plotabc_and_bv_comparisons.py \
  --input-dir analysis/ccn-2026/plotC_knn_diversity_outputs/new_things_embeddings_20260428 \
  --out-dir analysis/ccn-2026/plot_things_and_bv_comparisons_outputs \
  --valid-set valid129 \
  --k 5
```

### `scripts/build_tsne_datapage.py`

```bash
python analysis/ccn-2026/scripts/build_tsne_datapage.py \
  --tsne-csv analysis/manuscript-2026/tsne_cdi_results_clip/tsne_cdi_coordinates.csv \
  --per-exemplar-csv analysis/ccn-2026/old_plots/plotB_tsne_distance_to_centroid_outputs_20260401/bv_to_things_centroid_clip_per_exemplar.csv \
  --out-dir analysis/ccn-2026/tsne_datapage
```

### `scripts/regenerate_plotc_vector_pdfs.py`

```bash
python analysis/ccn-2026/scripts/regenerate_plotc_vector_pdfs.py
```

Reads Plot C summary CSVs in `plotC_knn_diversity_outputs/` and writes matching PDFs.
