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
| global_mean | 4.84 |  |
| global_sd | 0.5 |  |
| global_min | 3.43 |  |
| global_max | 5.75 |  |
| local_mean | 5.26 |  |
| local_sd | 0.68 |  |
| local_min | 3.35 |  |
| local_max | 6.58 |  |

## dispersion_all85_DINOv3

| Metric | Value | Notes |
|--------|-------|-------|
| global_mean | 11.51 |  |
| global_sd | 0.93 |  |
| global_min | 7.82 |  |
| global_max | 13.56 |  |
| local_mean | 13.27 |  |
| local_sd | 1.35 |  |
| local_min | 8.42 |  |
| local_max | 16.25 |  |

## correlations

| Metric | Value | Notes |
|--------|-------|-------|
| clip_within | rho=0.95, p=<.001, n=85 |  |
| dinov3_within | rho=0.87, p=<.001, n=85 |  |
| cross_global | rho=0.48, p=<.001, n=85 |  |
| cross_local_knn | rho=0.59, p=<.001, n=85 |  |

## montage_categories

| Metric | Value | Notes |
|--------|-------|-------|
| clock | CLIP global=4.1, local=3.91; DINO global=11.67, local=10.65 |  |
| plant | CLIP global=4.78, local=4.95; DINO global=10.64, local=12.12 |  |
| blanket | CLIP global=5.02, local=5.68; DINO global=11.04, local=13.23 |  |
| pajamas | CLIP global=5.2, local=5.92; DINO global=11.16, local=13.27 |  |
| book | CLIP global=5.75, local=6.58; DINO global=11.35, local=13.83 |  |

## cdi_semantic_groups

| Metric | Value | Notes |
|--------|-------|-------|
| toys | n=5; CLIP global M=5.32, z=0.95; DINO global M=11.94, z=0.47 |  |
| outside | n=7; CLIP global M=5.13, z=0.58; DINO global M=11.55, z=0.05 |  |
| vehicles | n=2; CLIP global M=5.01, z=0.35; DINO global M=12.01, z=0.55 |  |
| furniture_rooms | n=14; CLIP global M=4.96, z=0.24; DINO global M=11.54, z=0.04 |  |
| household | n=25; CLIP global M=4.95, z=0.22; DINO global M=11.93, z=0.45 |  |
| clothing | n=13; CLIP global M=4.93, z=0.18; DINO global M=11.3, z=-0.23 |  |
| animals | n=5; CLIP global M=4.85, z=0.02; DINO global M=12.21, z=0.76 |  |
| food_drink | n=2; CLIP global M=4.39, z=-0.89; DINO global M=11.2, z=-0.33 |  |
| body_parts | n=12; CLIP global M=4.03, z=-1.6; DINO global M=10.27, z=-1.33 |  |

## writeup_examples

| Metric | Value | Notes |
|--------|-------|-------|
| car | CLIP global=5.09, DINO global=13.56 |  |
| bird | CLIP global=5.01, DINO global=13.47 |  |
| pants | CLIP global=4.39, DINO global=10.65 |  |
| pajamas | CLIP global=5.2, DINO global=11.16 |  |

## clip_lowest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_clock | global=4.1, local kNN=3.91 |  |
| rank2_glasses | global=4.19, local kNN=4.29 |  |
| rank3_door | global=4.33, local kNN=4.48 |  |
| rank4_butterfly | global=4.46, local kNN=4.54 |  |
| rank5_watch | global=4.29, local kNN=4.56 |  |

## clip_highest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_book | global=5.75, local kNN=6.58 |  |
| rank2_tray | global=5.63, local kNN=6.46 |  |
| rank3_swing | global=5.67, local kNN=6.39 |  |
| rank4_toy | global=5.45, local kNN=6.29 |  |
| rank5_table | global=5.58, local kNN=6.12 |  |

## dino_lowest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_clock | global=11.67, local kNN=10.65 |  |
| rank2_butterfly | global=10.85, local kNN=11.22 |  |
| rank3_slide | global=10.8, local kNN=11.65 |  |
| rank4_stroller | global=10.47, local kNN=11.67 |  |
| rank5_swing | global=10.7, local kNN=11.85 |  |

## dino_highest_local_non_body

| Metric | Value | Notes |
|--------|-------|-------|
| rank1_car | global=13.56, local kNN=16.25 |  |
| rank2_dog | global=13.56, local kNN=15.87 |  |
| rank3_bird | global=13.47, local kNN=15.44 |  |
| rank4_bucket | global=12.47, local kNN=15.34 |  |
| rank5_bowl | global=12.84, local kNN=15.29 |  |
