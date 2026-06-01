# CCN 2026 — optional scripts

These CLIs are **not** required for the CCN poster pipeline (notebooks **01–03**, optional **04–06**). They support THINGS comparisons, figure export tweaks, and interactive exploration.

**Shared module (imported, stays in parent folder):** [`../load_things_embeddings.py`](../load_things_embeddings.py)

## When you need what

| Script | Role | Run if… |
|--------|------|---------|
| [`rerun_local_global_original_embeddings.py`](rerun_local_global_original_embeddings.py) | Recompute local/global metrics from **on-disk** `.npy` embeddings | You need fresh `things_vs_bv_*` CSVs under `plotC_knn_diversity_outputs/` |
| [`rerun_local_global_zscore.py`](rerun_local_global_zscore.py) | Same metrics from **z-scored** exemplar tables (`exemplar_set_embeddings/`) | You prefer manuscript-aligned z-scored inputs |
| [`generate_things_plotabc_and_bv_comparisons.py`](generate_things_plotabc_and_bv_comparisons.py) | THINGS Plot A/B/C analogs + BV–THINGS comparison figures | You have metric CSVs from a `rerun_*` folder and want comparison panels |
| [`make_local_global_extremes_tsne_panel.py`](make_local_global_extremes_tsne_panel.py) | 2×2 local/global extremes on category t-SNE | You want the script instead of [`06_make_local_global_extremes_tsne_panel.ipynb`](../06_make_local_global_extremes_tsne_panel.ipynb) |
| [`regenerate_plotc_vector_pdfs.py`](regenerate_plotc_vector_pdfs.py) | Publication PDFs for Plot C rank/extreme bar charts | Notebook **03** already saved PNGs but you need editable PDFs (`pdf.fonttype=42`) |
| [`build_tsne_datapage.py`](build_tsne_datapage.py) | Static HTML + thumbnails for browsing exemplars | Exploratory only; defaults target **legacy** joint BV–THINGS per-exemplar CSV columns |

## Poster pipeline (no scripts required)

```
01 Plot A → 02 Plot B → 03 Plot C → 06 t-SNE panel (notebook)
```

Optional notebooks **04** (inclusion stats) and **05** (invalid montages) do not use this folder.

## Example commands

From repo root (`object-detection/`):

```bash
python analysis/ccn-2026/scripts/rerun_local_global_zscore.py --valid-set valid129
python analysis/ccn-2026/scripts/generate_things_plotabc_and_bv_comparisons.py
python analysis/ccn-2026/scripts/regenerate_plotc_vector_pdfs.py
```

See [`../SCRIPTS.md`](../SCRIPTS.md) for full argument lists.

## Notes

- **`rerun_local_global_*`:** Pick **one** backend (raw embeddings vs z-score); outputs go under `plotC_knn_diversity_outputs/{original_embeddings_rerun,zscore_rerun,...}`.
- **`make_local_global_extremes_tsne_panel.py`:** Duplicates notebook **06**; prefer the notebook for reproducibility.
- **`build_tsne_datapage.py`:** Expects `dist_to_things_centroid` in the per-exemplar CSV. Current Plot B outputs (`bv-to-bv-centroid_distance_*_per_exemplar.csv`) use `dist_to_bv_centroid` — pass `--per-exemplar-csv` pointing at a joint-run CSV or update the script before using with BV-only outputs.
