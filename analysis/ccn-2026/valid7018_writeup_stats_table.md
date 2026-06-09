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
| global_mean | 18.12 |  |
| global_sd | 1.84 |  |
| global_min | 13.1 |  |
| global_max | 21.59 |  |
| local_mean | 20.17 |  |
| local_sd | 2.58 |  |
| local_min | 13.01 |  |
| local_max | 25.38 |  |

## dispersion_all85_DINOv3

| Metric | Value | Notes |
|--------|-------|-------|
| global_mean | 23.36 |  |
| global_sd | 1.76 |  |
| global_min | 16.18 |  |
| global_max | 27.81 |  |
| local_mean | 27.05 |  |
| local_sd | 2.6 |  |
| local_min | 17.49 |  |
| local_max | 33.29 |  |

## correlations

| Metric | Value | Notes |
|--------|-------|-------|
| clip_within | rho=0.95, p=<.001, n=85 |  |
| dinov3_within | rho=0.85, p=<.001, n=85 |  |
| cross_global | rho=0.55, p=<.001, n=85 |  |
| cross_local_knn | rho=0.62, p=<.001, n=85 |  |

## montage_categories

| Metric | Value | Notes |
|--------|-------|-------|
| glasses | CLIP global=15.53, local=16.25; DINO global=22.32, local=24.89 |  |
| couch | CLIP global=17.87, local=20.36; DINO global=23.13, local=27.88 |  |
| shoe | CLIP global=18.78, local=21.62; DINO global=23.68, local=27.90 |  |
| paper | CLIP global=19.45, local=21.37; DINO global=22.98, local=27.48 |  |
| book | CLIP global=21.59, local=25.38; DINO global=22.88, local=27.95 |  |

## cdi_semantic_groups

| Metric | Value | Notes |
|--------|-------|-------|
| toys | n=5; CLIP global M=20.21, z=1.14; DINO global M=23.99, z=0.36 |  |
| outside | n=7; CLIP global M=19.26, z=0.62; DINO global M=23.49, z=0.08 |  |
| vehicles | n=2; CLIP global M=18.73, z=0.33; DINO global M=24.59, z=0.70 |  |
| household | n=25; CLIP global M=18.62, z=0.27; DINO global M=24.02, z=0.38 |  |
| animals | n=5; CLIP global M=18.48, z=0.20; DINO global M=24.78, z=0.81 |  |
| furniture_rooms | n=14; CLIP global M=18.26, z=0.08; DINO global M=23.60, z=0.14 |  |
| clothing | n=13; CLIP global M=18.24, z=0.07; DINO global M=23.00, z=-0.20 |  |
| food_drink | n=2; CLIP global M=16.91, z=-0.66; DINO global M=22.46, z=-0.51 |  |
| body_parts | n=12; CLIP global M=15.17, z=-1.60; DINO global M=21.07, z=-1.30 |  |

## writeup_examples

| Metric | Value | Notes |
|--------|-------|-------|
| car | CLIP global=19.18, DINO global=27.81 |  |
| bird | CLIP global=19.09, DINO global=27.25 |  |
| pants | CLIP global=16.42, DINO global=21.79 |  |
| pajamas | CLIP global=19.29, DINO global=22.71 |  |

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
| rank5_pillow | proportion=0.0560 | household |

## frequency_full_dataset_bottom5

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_chips | proportion=0.0001 |  |
| rank2_eye | proportion=0.0001 | body_parts |
| rank3_ankle | proportion=0.0001 |  |
| rank4_stick | proportion=0.0000 | outside |
| rank5_sandwich | proportion=0.0000 |  |

## frequency_vs_dispersion_full_dataset

| Metric | Value | Notes |
|--------|-------|-------|
| clip | rho=0.182, p=.095, n=85 | x=full dataset detection proportion; y=valid7018 global dispersion |
| dinov3 | rho=0.265, p=.014, n=85 | x=full dataset detection proportion; y=valid7018 global dispersion |

## frequency_notes

| Metric | Value | Notes |
|--------|-------|-------|
| abstract_source | Abstract frequency panels use full infant-view detections (valid129_filtered, 0.27 CLIP filter). Dispersion is always from the 7,018 rater-validated crop sample (uniform per-category sampling by design). |  |
| not_used | 7018 crop counts / annotation-pool valid85 frequencies are not used in abstract panels |  |

## clip_lowest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_clock | global=15.68, local kNN=15.08 |  |
| rank2_glasses | global=15.53, local kNN=16.25 |  |
| rank3_door | global=15.88, local kNN=16.95 |  |
| rank4_butterfly | global=16.94, local kNN=17.54 |  |
| rank5_watch | global=16.39, local kNN=17.75 |  |

## clip_highest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_book | global=21.59, local kNN=25.38 |  |
| rank2_tray | global=21.32, local kNN=25.05 |  |
| rank3_toy | global=21.02, local kNN=24.70 |  |
| rank4_swing | global=20.74, local kNN=24.09 |  |
| rank5_box | global=20.32, local kNN=23.46 |  |

## dino_lowest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_clock | global=23.55, local kNN=21.62 |  |
| rank2_butterfly | global=21.81, local kNN=22.69 |  |
| rank3_slide | global=21.89, local kNN=23.70 |  |
| rank4_stroller | global=21.36, local kNN=23.90 |  |
| rank5_swing | global=21.65, local kNN=24.10 |  |

## dino_highest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_car | global=27.81, local kNN=33.29 |  |
| rank2_dog | global=27.70, local kNN=32.56 |  |
| rank3_bird | global=27.25, local kNN=31.26 |  |
| rank4_bowl | global=25.78, local kNN=30.84 |  |
| rank5_bucket | global=24.90, local kNN=30.76 |  |
