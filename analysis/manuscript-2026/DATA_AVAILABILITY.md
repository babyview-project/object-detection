# Data and code availability (manuscript boilerplate)

Use or adapt the following in the manuscript **Data/Code availability** section.

---

**Code.** Analysis code is available at [repository URL], directory
`analysis/manuscript-2026/`. Build category embedding tables first (Stage 0:
`00_build_exemplar_embeddings.md`, notebooks 06–07), then follow
`analysis/manuscript-2026/REPRODUCTION.md`.

**Shareable intermediates.** Anonymized category-level embedding tables, detection
prevalence summaries, and statistical result tables are in
`data/shared_data_manuscript_2026/` (see `MANIFEST.json`). Eight longitudinal
participants are labeled `participant_01`–`participant_08` (densest-first); no real
family identifiers are included.

**Restricted data.** Per-image crop embeddings and raw egocentric video require
BabyView data access under the project’s data-use agreement. Environment variables for
local embedding paths are documented in `analysis/manuscript-2026/paths.example.env`.

**Reproducing main-text statistics.** Compare tables in `data/shared_data_manuscript_2026/results_valid129/`
to the manuscript, or rerun notebooks **01–05** and **08–09** per `REPRODUCTION.md`.
Optional LaTeX autofill:
`analysis/manuscript-2026/not_in_manuscript/10_fill_results_preprint_numbers.ipynb`.

---
