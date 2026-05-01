# Preprint 2026 Analysis

This folder contains the notebooks, manuscript source files, and generated results used for the 2026 preprint analyses.

## What Is Here

- `01_long_tailed_distribution.ipynb` to `11_animal_depiction_label_proportions.ipynb`: analysis notebooks used to produce main and supplemental result tables.
- `supplemental_preprint.tex`: supplemental text.
- `main_results_valid129s_04302026/results`: primary outputs used in the main Results text.
- `supplemental_results_valid85cats_04302026/results`: supplemental outputs used in supplement tables/text.

## Typical Workflow

1. Run/update notebooks (`01`-`11`) as needed for analysis changes.
2. Verify outputs in `main_results_valid129s_04302026/results` and `supplemental_results_valid85cats_04302026/results`.
3. Run `10_fill_results_preprint_numbers.ipynb` to refresh preprint number summaries and backup text.
4. Copy/confirm updated values in manuscript and supplemental text files.
5. Manually fill fields that are intentionally not auto-derived.

## Notes

- Output folders include date-stamped snapshots; keep existing folders for reproducibility.
- The autofill step is designed to reduce transcription errors, but manuscript text still needs a final manual pass.
