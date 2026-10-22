# data/

Small, shareable inputs and published intermediates. No frames, crops or subject-level
tables live here.

| Path | What |
|---|---|
| `included_categories_valid129.txt`, `included_categories_valid85.txt` | the category sets used in the paper (valid129 = main, valid85 = supplement) |
| `cdi_words.csv`, `MCDI_items_with_AoA.csv`, `vocabulary_lists/` | CDI words, semantic categories, age of acquisition |
| `things_bv_overlap_categories*.txt`, `annotation_excluded_categories.txt` | THINGS / BabyView overlap lists and exclusions |
| `things_{clip,dinov3}_embeddings.{csv,npz}` | THINGS category embeddings |
| `long_tailed_dist_prop_included_categories*.csv`, `overall_category_distribution.csv` | detection-frequency tables (868-hour) |
| `average_precision_for_included_categories.csv`, `combined_precision_metrics_with_prevalence.csv` | per-category precision summaries |
| `babyview-videos-metadata-2025.1.csv`, `all_contexts_public.csv`, `vedi_averages.csv` | video metadata and context tables |
| `embeddings/`, `coef_data/`, `figures/` | older embedding and model-coefficient artefacts |
| [`shared_data_manuscript_2026/`](shared_data_manuscript_2026/) | anonymised tables behind the manuscript statistics (see its README; regenerate with `analysis/manuscript-2026/scripts/build_shared_public_data.py`) |
| `shared_data_ccn_2026/` | the same for the CCN 2026 poster (`valid5921_public` crops and embeddings) |
