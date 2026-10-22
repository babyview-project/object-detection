# precision_868h/: Prolific crop-validation study (868-hour release)

Raters on Prolific (objects-gallerize, github.com/babyview-project/objects-gallerize) saw
pages of YOLOE crops for one category and marked the invalid ones (other category,
empty, unrecognisable). Crops were sampled at up to 100 exemplars per category, keeping
categories with exemplars from at least 8 children, with OOD attention-check tiles.

## Outputs used elsewhere (untracked: they list crop file names with subject ids)

| File | Used by |
|---|---|
| `per_class_validation_data.csv` | per-category precision; the precision >= 0.6 filter in the manuscript, CCN and preprocessing (`manuscript_config.ANNOTATION_DIR`) |
| `per_file_precision_data.csv` | per-crop rater verdicts |
| `sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_babyview_only.csv` | the exemplar sample (Stage 0 exemplar sets, top-8 analyses) |
| `sampled_object_crops_100_bucket_assignments_100ex_8subj*.csv`, `*_report.txt` | earlier versions of the bucket assignment and their summaries |
| `example_page_correct_detections_{book,table}.csv`, `exemplar_blurry_table_2.jpg` | instruction / practice pages |
| `ood_attention_checks_babyview_nonobject_163_manifest.csv` | non-object attention-check tiles (`preprocessing/sample_non_object_attention_tiles.py`) |
| `cache_valid85_filter_index.pkl` | cached filter index |

The aggregate per-category precision is also published in
`data/shared_data_manuscript_2026/` and `data/shared_data_ccn_2026/`.

## Scripts (in the order they were used)

| Script | Does |
|---|---|
| `preprocessing/sample_object_crops_variability.py` | samples exemplars per category into buckets (writes here) |
| `resample_diverse_buckets.py` | rebalances buckets to reduce subject / video concentration |
| `filter_bucket_assignments_100ex_8subj.py` | keeps categories with >= 100 exemplars from >= 8 children, caps per video |
| `strip_ood_attention_checks_bucket_csv.py` | drops the OOD attention-check rows (BabyView crops only) |
| `make_sampled_bucket_assignment_montages.py` | QC montages per category / bucket |
| `make_example_page_practice_category.py` | builds an example / practice page for one category |

Rater responses were analysed in `analysis/manuscript-2026/online_annotation/online_annotation.qmd`.
The montage folder `sampled_object_crops_100ex_8subj_montages_babyview_only` was a link into
`/data2` and is gone.
