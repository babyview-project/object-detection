# Shared public data (BabyView Objects manuscript)

Anonymized intermediate tables for the BabyView Objects manuscript (June 2026).
Safe to redistribute with the repository: no raw video, crop paths, or real
participant identifiers.

## Regenerating

From the repository root (after running the manuscript pipeline locally):

```bash
python analysis/manuscript-2026/scripts/build_shared_public_data.py
```

Reads from `analysis/manuscript-2026/main_results_*`, `exemplar_set_embeddings/`,
and `data/`, then writes this directory. Real subject IDs are mapped to
`participant_01`–`participant_08` in memory only.

## Layout

| Path | Contents |
|------|----------|
| `MANIFEST.json` | File list and generation timestamp |
| `metadata/participant_registry_top8.csv` | Pseudonymous top-8 ranks and valid85 coverage |
| `category_lists/` | `included_categories_valid{129,85}.txt` |
| `inputs/` | Detection prevalence, precision, animal-depiction proportions |
| `embeddings/` | Category-level z-scored exemplar means (BV + THINGS) |
| `results_valid129/` | Main-text statistics tables |
| `top8_valid85/{clip,dinov3}/` | Anonymized top-8 CDI vs THINGS tables |

## Analysis code

Notebooks and scripts: [`analysis/manuscript-2026/`](../../analysis/manuscript-2026/).
See [`REPRODUCTION.md`](../../analysis/manuscript-2026/REPRODUCTION.md).
