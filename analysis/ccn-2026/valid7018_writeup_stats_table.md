# valid7018 writeup stats

Autofilled from `valid7018/valid7018_paper_stats.json` and metric CSVs.

## cohort

| Metric | Value | Notes |
|--------|-------|-------|
| n_exemplars | 7018 |  |
| n_categories | 85 |  |
| n_per_category_min | 57 |  |
| n_per_category_max | 100 |  |

## dispersion_all85_CLIP

| Metric | Value | Notes |
|--------|-------|-------|
| global_mean | 24.46 |  |
| global_sd | 2.48 |  |
| global_min | 17.7 |  |
| global_max | 29.17 |  |
| local_mean | 27.24 |  |
| local_sd | 3.48 |  |
| local_min | 17.56 |  |
| local_max | 34.29 |  |

## dispersion_all85_DINOv3

| Metric | Value | Notes |
|--------|-------|-------|
| global_mean | 32.08 |  |
| global_sd | 2.39 |  |
| global_min | 22.28 |  |
| global_max | 38.28 |  |
| local_mean | 37.17 |  |
| local_sd | 3.55 |  |
| local_min | 24.07 |  |
| local_max | 45.81 |  |

## correlations

| Metric | Value | Notes |
|--------|-------|-------|
| clip_within | rho=0.95, p=<.001, n=85 |  |
| dinov3_within | rho=0.85, p=<.001, n=85 |  |
| cross_global | rho=0.55, p=<.001, n=85 |  |
| cross_local_knn | rho=0.63, p=<.001, n=85 |  |

## montage_categories

| Metric | Value | Notes |
|--------|-------|-------|
| glasses | CLIP global=20.98, local=21.97; DINO global=30.72, local=34.3 |  |
| oven | CLIP global=24.13, local=27.24; DINO global=31.95, local=36.19 |  |
| balloon | CLIP global=25.33, local=28.03; DINO global=32.63, local=38.48 |  |
| paper | CLIP global=26.26, local=28.88; DINO global=31.48, local=37.68 |  |
| book | CLIP global=29.17, local=34.29; DINO global=31.39, local=38.36 |  |

## cdi_semantic_groups

| Metric | Value | Notes |
|--------|-------|-------|
| animals | n=5; CLIP global M=24.91, z=0.18; DINO global M=34.04, z=0.82 |  |
| body_parts | n=12; CLIP global M=20.5, z=-1.6; DINO global M=28.98, z=-1.3 |  |
| clothing | n=13; CLIP global M=24.65, z=0.08; DINO global M=31.64, z=-0.19 |  |
| food_drink | n=2; CLIP global M=22.83, z=-0.66; DINO global M=30.78, z=-0.54 |  |
| furniture_rooms | n=14; CLIP global M=24.68, z=0.09; DINO global M=32.48, z=0.17 |  |
| household | n=25; CLIP global M=25.13, z=0.27; DINO global M=32.94, z=0.36 |  |
| outside | n=7; CLIP global M=26.01, z=0.63; DINO global M=32.28, z=0.09 |  |
| toys | n=5; CLIP global M=27.26, z=1.13; DINO global M=32.86, z=0.33 |  |
| vehicles | n=2; CLIP global M=25.29, z=0.33; DINO global M=33.82, z=0.73 |  |

## writeup_examples

| Metric | Value | Notes |
|--------|-------|-------|
| car | CLIP global=25.89, DINO global=38.28 |  |
| bird | CLIP global=25.73, DINO global=37.43 |  |
| pants | CLIP global=22.19, DINO global=30.03 |  |
| pajamas | CLIP global=26.1, DINO global=31.24 |  |

## frequency_full_dataset

| Metric | Value | Notes |
|--------|-------|-------|
| source | long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv | 0.27-filtered infant-view detections; 129-category denominator |

## frequency_full_dataset_top5

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_chair | proportion=0.1025 | furniture_rooms |
| rank2_lamp | proportion=0.0785 | household |
| rank3_table | proportion=0.0644 | furniture_rooms |
| rank4_couch | proportion=0.0635 | furniture_rooms |
| rank5_pillow | proportion=0.056 | household |

## frequency_full_dataset_bottom5

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_rock | proportion=0.0001 | outside |
| rank2_pony | proportion=0.0001 | animals |
| rank3_swing | proportion=0.0001 | outside |
| rank4_eye | proportion=0.0001 | body_parts |
| rank5_stick | proportion=0.0 | outside |

## frequency_vs_dispersion_full_dataset

| Metric | Value | Notes |
|--------|-------|-------|
| clip | rho=0.182, p=.095, n=85 | x=full dataset detection proportion; y=valid7018 global dispersion |
| dinov3 | rho=0.261, p=.016, n=85 | x=full dataset detection proportion; y=valid7018 global dispersion |

## frequency_notes

| Metric | Value | Notes |
|--------|-------|-------|
| abstract_source | Abstract frequency panels use full infant-view detections (valid129_filtered, 0.27 CLIP filter). Dispersion is always from the 7,018 rater-validated crop sample (uniform per-category sampling by design). |  |
| not_used | 7018 crop counts / annotation-pool valid85 frequencies are not used in abstract panels |  |

## clip_lowest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_clock | global=21.17, local kNN=20.39 |  |
| rank2_glasses | global=20.98, local kNN=21.97 |  |
| rank3_door | global=21.45, local kNN=22.91 |  |
| rank4_butterfly | global=22.85, local kNN=23.69 |  |
| rank5_watch | global=22.14, local kNN=23.97 |  |

## clip_highest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_book | global=29.17, local kNN=34.29 |  |
| rank2_tray | global=28.75, local kNN=33.79 |  |
| rank3_toy | global=28.34, local kNN=33.32 |  |
| rank4_swing | global=28.02, local kNN=32.56 |  |
| rank5_box | global=27.43, local kNN=31.68 |  |

## dino_lowest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_clock | global=32.32, local kNN=29.7 |  |
| rank2_butterfly | global=29.88, local kNN=31.12 |  |
| rank3_slide | global=30.07, local kNN=32.59 |  |
| rank4_stroller | global=29.35, local kNN=32.87 |  |
| rank5_swing | global=29.79, local kNN=33.19 |  |

## dino_highest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_car | global=38.28, local kNN=45.81 |  |
| rank2_dog | global=38.12, local kNN=44.83 |  |
| rank3_bird | global=37.43, local kNN=42.94 |  |
| rank4_bowl | global=35.32, local kNN=42.29 |  |
| rank5_bucket | global=34.07, local kNN=42.14 |  |
