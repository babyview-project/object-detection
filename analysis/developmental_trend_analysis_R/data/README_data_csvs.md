# CSV data files — Developmental trajectory (top 8 densest)

Data for developmental trajectory analyses: RDM (representational dissimilarity matrix) correlations across age for the **top 8 subjects** by recording density. Age range ~9.5–28.5 months; *n* = 100 observations per main correlation analysis.

**Used by:** `developmental_trajectory_stats.qmd`

---

## File list and column reference

### 1. `hours_per_subject_age_mo.csv`

**Purpose:** Video hours and recording counts per subject per age (month).

| Column | Description |
|--------|-------------|
| `subject_id` | Subject identifier |
| `age_mo` | Age in months |
| `hours_video` | Total hours of video at that age |
| `n_recordings` | Number of recordings |

- **Rows:** 115 (header + 114 data rows)

---

### 2. `month_to_month_correlations.csv`

**Purpose:** RDM correlation between **consecutive months** (all fixed categories per subject).

| Column | Description |
|--------|-------------|
| `subject_id` | Subject identifier |
| `age_from` | Younger month (integer) |
| `age_to` | Older month (integer) |
| `age_midpoint` | (age_from + age_to) / 2 |
| `correlation` | RDM correlation between the two months |
| `n_common_categories` | Number of categories used (fixed set per subject) |
| `hours_from`, `hours_to` | Video hours in each month |
| `hours_bin` | Average of hours_from and hours_to |
| `n_rdm_pairs` | Number of RDM pairs in correlation |
| `cor_se` | Standard error of the correlation |

- **Rows:** 101 (header + 100 observations)

---

### 3. `month_to_month_correlations_top50.csv`

**Purpose:** Same as above but restricted to the **top 50 densest categories** per subject.

| Column | Description |
|--------|-------------|
| `subject_id`, `age_from`, `age_to`, `age_midpoint` | As in month_to_month_correlations |
| `correlation` | RDM correlation (top-50 categories only) |
| `n_common_categories` | 50 |
| `n_rdm_pairs` | Number of RDM pairs (1225 for 50 categories) |
| `cor_se` | Standard error of the correlation |

- **Rows:** 101 (header + 100 observations). One-to-one with `month_to_month_correlations.csv` on (subject_id, age_from, age_to) for paired comparisons (all vs top-50).

---

### 4. `youngest_vs_later_month_correlations.csv`

**Purpose:** RDM correlation between each subject’s **youngest month** and each **later month** (all fixed categories).

| Column | Description |
|--------|-------------|
| `subject_id` | Subject identifier |
| `age_youngest` | Age at youngest month (baseline) |
| `age_compared` | Age of the later month |
| `correlation` | RDM correlation (youngest vs compared month) |
| `n_common_categories` | Number of categories (fixed per subject) |
| `hours_compared` | Video hours in the compared month |
| `n_rdm_pairs`, `cor_se` | As in month_to_month |

- **Rows:** 101 (header + 100 observations)

---

### 5. `youngest_vs_later_month_correlations_top50.csv`

**Purpose:** Same as above for the **top 50 densest categories** only.

| Column | Description |
|--------|-------------|
| Same structure as `youngest_vs_later_month_correlations.csv` but without `hours_compared`; `n_common_categories` = 50. |

- **Rows:** 101 (header + 100 observations)

---

### 6. `subject_fixed_categories_all.csv`

**Purpose:** Count of fixed (stable) categories per subject (used for “all categories” RDMs).

| Column | Description |
|--------|-------------|
| `subject_id` | Subject identifier |
| `n_fixed_categories` | Number of categories in the subject’s fixed set |

- **Rows:** 9 (header + 8 subjects)

---

### 7. `subject_fixed_categories_list_all.csv`

**Purpose:** List of category labels in each subject’s fixed set.

| Column | Description |
|--------|-------------|
| `subject_id` | Subject identifier |
| `category` | Category name (e.g. horse, rooster) |

- **Rows:** 751 (header + 750 category–subject rows). Multiple rows per subject, one per category.

---

### 8. `smooth_trajectory_month_to_month.csv`

**Purpose:** Smoothed (fitted) trajectory for **month-to-month** correlation vs age; used for plot bands.

| Column | Description |
|--------|-------------|
| `subject_id` | `"overall"` for population curve or subject ID for subject-specific |
| `age` | Age (continuous grid) |
| `pred` | Predicted correlation |
| `ci_lo`, `ci_hi` | Lower and upper confidence interval bounds |

- **Rows:** 721 (header + 720). Dense age grid for smooth curves.

---

### 9. `smooth_youngest_vs_later.csv`

**Purpose:** Smoothed trajectory for **youngest-vs-later** correlation vs age; used for plot bands.

| Column | Description |
|--------|-------------|
| Same as `smooth_trajectory_month_to_month.csv`: `subject_id`, `age`, `pred`, `ci_lo`, `ci_hi`. |

- **Rows:** 721 (header + 720)

---

## Quick reference

| File | Rows (excl. header) | Main use |
|------|---------------------|----------|
| `hours_per_subject_age_mo.csv` | 114 | Recording density by subject/age |
| `month_to_month_correlations.csv` | 100 | MM trend, LMM, plots (all categories) |
| `month_to_month_correlations_top50.csv` | 100 | MM top-50 vs all comparison |
| `youngest_vs_later_month_correlations.csv` | 100 | YV trend, LMM, plots (all categories) |
| `youngest_vs_later_month_correlations_top50.csv` | 100 | YV top-50 vs all |
| `subject_fixed_categories_all.csv` | 8 | Subject-level category counts |
| `subject_fixed_categories_list_all.csv` | 750 | Category names per subject |
| `smooth_trajectory_month_to_month.csv` | 720 | MM smooth curve/CI bands |
| `smooth_youngest_vs_later.csv` | 720 | YV smooth curve/CI bands |

MM = month-to-month; YV = youngest vs later.
