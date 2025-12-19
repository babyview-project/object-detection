# VSS 2026 RDM Analysis Notebooks

This directory contains Jupyter notebooks for computing and analyzing Representational Dissimilarity Matrices (RDMs) for CLIP and DINOv3 category embeddings.

## Overview

The analysis pipeline consists of six main notebooks:

1. **`01_compute_average_embeddings.ipynb`** - Computes category average embeddings from individual CLIP and DINOv3 embeddings
2. **`02_filter_normalize_and_compute_rdm.ipynb`** - Filters, normalizes, and reorganizes RDMs by category type with optional hierarchical clustering
3. **`03_correlate_rdm_matrices.ipynb`** - Correlates two RDM matrices (e.g., BV_CLIP vs THINGS_CLIP)
4. **`04_correlate_category_embeddings.ipynb`** - Correlates category-level average embeddings between two embedding files
5. **`05_individual_subject_rdm.ipynb`** - Creates individual subject RDMs from grouped embeddings (averaged by category, subject, and age_mo)
6. **`06_developmental_trajectory_rdm.ipynb`** - Creates developmental trajectory RDMs (multiple RDMs per subject, binned by age_mo)

## Prerequisites

- Python 3.x
- Required Python packages:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tqdm`

## Typical Workflow

The analysis follows a sequential pipeline through four notebooks. Each notebook should be executed in order, with outputs from earlier steps serving as inputs to later steps.

### Step 1: Compute Average Embeddings

**Notebook:** `01_compute_average_embeddings.ipynb`

This notebook computes category average embeddings from individual CLIP and DINOv3 embeddings.

**What it does:**
- Loads individual CLIP embeddings from files
- Groups embeddings by category
- Computes category average embeddings for CLIP
- Loads individual DINOv3 embeddings from files (optionally matching from CLIP list)
- Groups embeddings by category
- Computes category average embeddings for DINOv3
- Saves the average embeddings for later RDM computation

**Configuration:**
- Update paths in the configuration cell for:
  - CLIP embedding list and directory
  - DINOv3 embedding list and directory
  - Output directories for CLIP and DINOv3 results
  - Option to match DINOv3 filenames from CLIP list (ensures same images)

**Output:**
- `category_average_embeddings.npz` files for CLIP and DINOv3
- Category information files

### Step 2: Filter, Normalize, and Compute RDM

**Notebook:** `02_filter_normalize_and_compute_rdm.ipynb`

This notebook filters out low-quality categories, normalizes embeddings, and reorganizes the RDM by category type with optional hierarchical clustering.

**What it does:**
- Loads category average embeddings from Step 1
- Filters categories based on inclusion/exclusion lists
- Normalizes embeddings (z-scoring)
- Computes and saves filtered RDM matrices (similarity and distance)
- Organizes categories by type (animals, bodyparts, big objects, small objects)
- Optionally applies hierarchical clustering within each group
- Creates visualizations (RDM heatmaps, dendrograms)

**Configuration:**
- Update paths in the configuration cell for:
  - Input NPZ file with category average embeddings
  - Inclusion or exclusion list file
  - CDI words CSV file for category type information
  - Output directory

**Output:**
- Filtered and normalized RDM matrices (similarity and distance)
- Reorganized RDM visualizations
- Dendrograms for each category group (if enabled)
- Top similar/dissimilar category pairs

### Step 3: Correlate RDM Matrices

**Notebook:** `03_correlate_rdm_matrices.ipynb`

This notebook correlates two RDM matrices (e.g., BV_CLIP vs THINGS_CLIP) using Pearson and Spearman correlations.

**What it does:**
- Loads two RDM distance matrices (e.g., BV_CLIP and THINGS_CLIP)
- Matches categories between matrices if needed
- Extracts lower triangle (excluding diagonal)
- Computes Pearson and Spearman correlations
- Reports correlation statistics

**Configuration:**
- Update paths in the configuration cell for:
  - First RDM matrix path
  - Second RDM matrix path
  - Output directory and filename for results

**Output:**
- Correlation statistics (Pearson r, p-value; Spearman r, p-value)
- Matrix information and statistics
- Results saved to text file

### Step 4: Correlate Category Embeddings

**Notebook:** `04_correlate_category_embeddings.ipynb`

This notebook correlates category-level average embeddings between two embedding files (e.g., bv_clip and things_clip).

**What it does:**
- Loads category average embeddings from two sources
- Finds matching categories between the two sets
- Computes correlations (Pearson, Spearman, Cosine) for each category
- Reports summary statistics and top/bottom categories

**Configuration:**
- Update paths in the configuration cell for:
  - First category average embeddings file
  - Second category average embeddings file
  - Output directory and filename for results

**Output:**
- Summary statistics (mean, std, median, min, max for each metric)
- Top 10 and bottom 10 categories by Pearson correlation
- Detailed correlation results saved to text file

---

## Notebook Documentation

### 1. `01_compute_average_embeddings.ipynb`

Computes category average embeddings from individual CLIP and DINOv3 embeddings.

**Key Features:**
- Loads embeddings from `.npy` files
- Groups embeddings by category
- Computes category average embeddings for CLIP and DINOv3
- Supports parallel loading for efficiency
- Option to match DINOv3 filenames from CLIP list (ensures same images)

**Configuration:**
- Update paths in the configuration cell:
  - `CLIP_EMBEDDING_LIST`: Path to text file with CLIP embedding paths (one per line), or None to scan directory
  - `CLIP_EMBEDDINGS_DIR`: Base directory for CLIP embeddings
  - `CLIP_OUTPUT_DIR`: Directory where CLIP results will be saved
  - `DINOV3_EMBEDDING_LIST`: Path to text file with DINOv3 embedding paths, or None to scan directory
  - `DINOV3_EMBEDDINGS_DIR`: Base directory for DINOv3 embeddings
  - `DINOV3_OUTPUT_DIR`: Directory where DINOv3 results will be saved
  - `MATCH_FROM_CLIP_LIST`: If True, match DINOv3 filenames from CLIP list
  - `NUM_WORKERS`: Number of parallel workers (None = auto-detect)
  - `USE_PARALLEL`: Enable parallel loading

**Output Files:**
- `category_average_embeddings.npz`: Category average embeddings (numpy array and category names)
- `category_average_embeddings.csv`: Category averages (CSV format)
- `category_average_info.txt`: Summary information about categories
- `category_names.txt`: List of category names

---

### 2. `02_filter_normalize_and_compute_rdm.ipynb`

Filters out low-quality categories, normalizes embeddings, and reorganizes the RDM by category type with optional hierarchical clustering.

**Key Features:**
- Filters categories by exclusion or inclusion lists
- Normalizes embeddings using z-scoring
- Computes cosine similarity and distance matrices
- Organizes categories by type (animals, bodyparts, big objects, small objects)
- Applies hierarchical clustering within each group
- Creates filtered RDM visualizations
- Generates dendrograms for category groups
- Identifies top similar/dissimilar category pairs

**Configuration:**
- Update paths in the configuration cell:
  - `INPUT_NPZ_PATH`: Path to NPZ file with category average embeddings from Step 1
  - `EXCLUSION_FILE` or `INCLUSION_FILE`: Text file with categories to exclude/include (one per line)
  - `CDI_PATH`: Path to CDI words CSV file (default: `./data/cdi_words.csv`)
  - `OUTPUT_DIR`: Output directory for filtered results
  - `USE_CLUSTERING`: Enable hierarchical clustering within groups
  - `SAVE_DENDROGRAMS`: Save dendrogram plots for each category group

**Output Files:**
- `similarity_matrix_filtered.npy` / `.csv`: Filtered similarity matrix
- `distance_matrix_filtered.npy` / `.csv`: Filtered distance matrix
- `distance_matrix_filtered_original.npy`: Original distance matrix before normalization
- `rdm_organized_filtered.png`: Filtered RDM heatmap (viridis colormap)
- `rdm_organized_filtered_coolwarm.png`: Filtered RDM heatmap (coolwarm colormap)
- `dendrogram_*.png` / `.pdf`: Dendrograms for each category group (if enabled)
- `top_N_similar_pairs.png` / `.txt`: Top N most similar category pairs
- `top_N_dissimilar_pairs.png` / `.txt`: Top N most dissimilar category pairs
- `category_names_filtered.txt`: List of filtered category names
- `category_organization_filtered.txt`: Category organization information

---

### 3. `03_correlate_rdm_matrices.ipynb`

Correlates two RDM matrices (e.g., BV_CLIP vs THINGS_CLIP) using Pearson and Spearman correlations.

**Key Features:**
- Supports both `.npy` and `.csv` input formats
- Matches categories between matrices if needed
- Extracts lower triangle (excluding diagonal)
- Computes Pearson and Spearman correlations
- Reports correlation statistics

**Configuration:**
- Update paths in the configuration cell:
  - `INPUT_RDM1_PATH`: Path to first RDM matrix file (.npy or .csv)
  - `INPUT_RDM2_PATH`: Path to second RDM matrix file (.npy or .csv)
  - `OUTPUT_DIR`: Output directory for correlation results
  - `OUTPUT_FILENAME`: Output filename for results

**Output:**
- Correlation statistics (Pearson r, p-value; Spearman r, p-value)
- Matrix information and statistics
- Results saved to text file

---

### 4. `04_correlate_category_embeddings.ipynb`

Correlates category-level average embeddings between two embedding files (e.g., bv_clip and things_clip).

**Key Features:**
- Loads category average embeddings from two sources
- Finds matching categories between the two sets
- Computes correlations (Pearson, Spearman, Cosine) for each category
- Reports summary statistics and top/bottom categories

**Configuration:**
- Update paths in the configuration cell:
  - `AVG_CAT_EMB_PATH1`: Path to first category average embeddings file (.npz)
  - `AVG_CAT_EMB_PATH2`: Path to second category average embeddings file (.npz)
  - `OUTPUT_DIR`: Output directory for correlation results
  - `OUTPUT_FILENAME`: Output filename for results

**Output:**
- Summary statistics (mean, std, median, min, max for each metric)
- Top 10 and bottom 10 categories by Pearson correlation
- Detailed correlation results saved to text file

---

### 5. `05_individual_subject_rdm.ipynb`

Creates Representational Dissimilarity Matrices (RDMs) for each individual subject using grouped embeddings.

**Key Features:**
- Loads grouped embeddings from category subfolders (format: `{category}/{subject_id}_{age_mo}_month_level_avg.npy`)
- Aggregates embeddings per subject across all age_mo bins (weighted average)
- Applies **global normalization** across all subjects for cross-subject comparability
- Computes cosine distance RDMs for each subject
- Handles data density differences with configurable minimum category thresholds
- Saves individual RDMs, metadata, and summary statistics
- Creates visualizations of sample RDMs and data density analysis

**Configuration:**
- Update paths in the configuration cell:
  - `embeddings_dir`: Path to directory containing category subfolders with grouped embeddings
  - `categories_file`: Optional path to text file with categories to include (one per line)
  - `output_dir`: Output directory for individual subject RDMs
  - `min_categories_per_subject`: Minimum categories required per subject to compute RDM (default: 10)
  - `weight_by_sample_count`: Whether to weight by sample count when aggregating across age_mo (default: True)

**Prerequisites:**
- Grouped embeddings from preprocessing pipeline (see `preprocessing/group_embeddings_by_agemo.py`)
- Embeddings should be organized as: `{category}/{subject_id}_{age_mo}_month_level_avg.npy`

**Output Files:**
- `individual_subject_rdms/rdm_{subject_id}.npy` - RDM matrix per subject (numpy format)
- `individual_subject_rdms/rdm_{subject_id}.csv` - RDM matrix with category labels (CSV format)
- `individual_subject_rdms/metadata_{subject_id}.csv` - Metadata per subject (n_categories, mean_distance, etc.)
- `individual_subject_rdms/summary_statistics.csv` - Summary statistics across all subjects
- `individual_subject_rdms/sample_rdms.png` - Visualization of sample RDMs
- `individual_subject_rdms/data_density_analysis.png` - Data density analysis plots

**Normalization Details:**
- **Global normalization**: Computes mean and std across ALL embeddings from ALL subjects
- Formula: `(embedding - global_mean) / global_std` applied to all subjects
- Enables meaningful cross-subject RDM comparisons
- Data density differences are handled naturally (subjects with more data contribute more to global stats)

---

### 6. `06_developmental_trajectory_rdm.ipynb`

Creates multiple Representational Dissimilarity Matrices (RDMs) for each individual subject, binned by age in months (age_mo), to track developmental changes in object representations.

**Key Features:**
- Loads grouped embeddings organized by subject, age_mo, and category
- Bins embeddings by age_mo for each subject (supports exact age or binned age)
- Applies **per-subject, per-age-bin normalization** for within-subject developmental tracking
- Computes RDM for each subject at each age_mo bin
- Analyzes RDM stability across development (correlations between consecutive age bins)
- Handles data density differences with configurable minimum category thresholds per age bin
- Visualizes developmental trajectories and RDM stability

**Configuration:**
- Update paths in the configuration cell:
  - `embeddings_dir`: Path to directory containing category subfolders with grouped embeddings
  - `categories_file`: Optional path to text file with categories to include (one per line)
  - `output_dir`: Output directory for developmental trajectory RDMs
  - `min_categories_per_age_bin`: Minimum categories required per age bin (default: 8)
  - `min_age_bins_per_subject`: Minimum age bins required per subject (default: 2)
  - `age_binning_strategy`: 'exact' (use exact age_mo) or 'binned' (group into bins)
  - `age_bin_size`: Size of age bins if using 'binned' strategy (default: 3 months)

**Prerequisites:**
- Grouped embeddings from preprocessing pipeline (see `preprocessing/group_embeddings_by_agemo.py`)
- Embeddings should be organized as: `{category}/{subject_id}_{age_mo}_month_level_avg.npy`

**Output Files:**
- `developmental_trajectory_rdms/{subject_id}/rdm_age_{age_mo}.npy` - RDM per subject-age combination
- `developmental_trajectory_rdms/{subject_id}/rdm_age_{age_mo}.csv` - RDM with category labels
- `developmental_trajectory_rdms/{subject_id}/metadata_age_{age_mo}.csv` - Metadata per age bin
- `developmental_trajectory_rdms/trajectory_correlations.csv` - RDM correlations between consecutive age bins
- `developmental_trajectory_rdms/summary_statistics.csv` - Summary statistics
- `developmental_trajectory_rdms/trajectory_{subject_id}.png` - Visualization of developmental trajectory for sample subjects
- `developmental_trajectory_rdms/rdm_stability_analysis.png` - RDM stability analysis plots

**Normalization Details:**
- **Per-subject, per-age-bin normalization**: Each age bin is normalized independently
- Formula: `(embedding - age_bin_mean) / age_bin_std` for each subject-age combination
- Focuses on relative structure changes within each subject over time
- Appropriate for tracking developmental changes in representation structure
- Allows comparison of RDM structure changes across age bins within subjects

**Trajectory Analysis:**
- Computes Spearman correlations between consecutive age bins for each subject
- Tracks RDM stability across development
- Analyzes how RDM structure changes as a function of age difference and mean age

---

### Step 5: Individual Subject RDM Analysis

**Notebook:** `05_individual_subject_rdm.ipynb`

This notebook creates Representational Dissimilarity Matrices (RDMs) for each individual subject using grouped embeddings that have been averaged by category, subject, and age_mo.

**What it does:**
- Loads grouped embeddings from category subfolders
- Aggregates embeddings per subject across all age_mo bins (weighted average)
- Applies **global normalization** across all subjects (enables cross-subject RDM comparisons)
- Computes RDM for each subject using cosine distance
- Handles data density differences (minimum category threshold per subject)
- Visualizes and saves individual subject RDMs

**Key Features:**
- **Global normalization**: All subjects normalized using the same global mean/std, enabling cross-subject RDM comparisons
- **Data density handling**: Filters subjects with insufficient categories
- **Missing category handling**: Only includes categories present for each subject
- **Weighted averaging**: If a subject has multiple age_mo bins, embeddings are weighted by sample count

**Configuration:**
- Update paths in the configuration cell for:
  - `embeddings_dir`: Directory containing category subfolders with grouped embeddings
  - `categories_file`: Optional text file with categories to include (one per line)
  - `output_dir`: Output directory for individual subject RDMs
  - `min_categories_per_subject`: Minimum categories required per subject (default: 10)
  - `weight_by_sample_count`: Whether to weight by sample count when aggregating (default: True)

**Output:**
- `individual_subject_rdms/rdm_{subject_id}.npy` - RDM matrices per subject
- `individual_subject_rdms/rdm_{subject_id}.csv` - RDM with category labels
- `individual_subject_rdms/metadata_{subject_id}.csv` - Metadata per subject
- `individual_subject_rdms/summary_statistics.csv` - Summary across all subjects
- Visualization plots (sample RDMs, data density analysis)

**Normalization Strategy:**
- **Global normalization**: Computes mean and std across ALL embeddings from ALL subjects
- Applies the same normalization to each subject: `(embedding - global_mean) / global_std`
- This ensures all subjects are on the same scale for meaningful cross-subject RDM comparisons
- Data density differences are naturally handled (subjects with more data contribute more to global stats)

---

### Step 6: Developmental Trajectory RDM Analysis

**Notebook:** `06_developmental_trajectory_rdm.ipynb`

This notebook creates multiple Representational Dissimilarity Matrices (RDMs) for each individual subject, binned by age in months (age_mo). This allows tracking how object representations change developmentally within each subject.

**What it does:**
- Loads grouped embeddings organized by subject, age_mo, and category
- Bins embeddings by age_mo for each subject
- Applies **per-subject, per-age-bin normalization** (focuses on relative structure changes within subjects)
- Computes RDM for each subject at each age_mo bin
- Handles data density differences (minimum category threshold per age bin)
- Analyzes RDM stability across development
- Visualizes developmental trajectories

**Key Features:**
- **Per-subject, per-age-bin normalization**: Each age bin normalized relative to that subject's data at that age
- **Age binning**: Groups embeddings by age_mo to track developmental changes
- **Data density handling**: Minimum category threshold per age bin
- **Trajectory analysis**: Compares RDMs across age bins to see developmental changes
- **RDM correlation analysis**: Computes correlations between consecutive age bins

**Configuration:**
- Update paths in the configuration cell for:
  - `embeddings_dir`: Directory containing category subfolders with grouped embeddings
  - `categories_file`: Optional text file with categories to include (one per line)
  - `output_dir`: Output directory for developmental trajectory RDMs
  - `min_categories_per_age_bin`: Minimum categories required per age bin (default: 8)
  - `min_age_bins_per_subject`: Minimum age bins required per subject (default: 2)
  - `age_binning_strategy`: 'exact' (use exact age_mo) or 'binned' (group into bins)
  - `age_bin_size`: Size of age bins if using 'binned' strategy (default: 3 months)

**Output:**
- `developmental_trajectory_rdms/{subject_id}/rdm_age_{age_mo}.npy` - RDM per subject-age combination
- `developmental_trajectory_rdms/{subject_id}/rdm_age_{age_mo}.csv` - RDM with category labels
- `developmental_trajectory_rdms/trajectory_correlations.csv` - RDM correlations across ages
- `developmental_trajectory_rdms/summary_statistics.csv` - Summary statistics
- Visualization plots (trajectory plots, RDM stability analysis)

**Normalization Strategy:**
- **Per-subject, per-age-bin normalization**: Each age bin is normalized independently
- Formula: `(embedding - age_bin_mean) / age_bin_std` for each subject-age combination
- This focuses on relative structure changes within each subject over time
- Appropriate for tracking developmental changes in representation structure
- Allows comparison of RDM structure changes across age bins within subjects

---

## File Format Requirements

### Embedding Files
- **Input embeddings**: `.npy` files containing 1D embedding vectors
- **Category average embeddings**: `.npz` files with keys:
  - `embeddings`: numpy array of shape (n_categories, embedding_dim)
  - `categories`: numpy array of category names
- **Grouped embeddings** (for notebooks 05 and 06):
  - Directory structure: `{category}/{subject_id}_{age_mo}_month_level_avg.npy`
  - Each `.npy` file contains an averaged embedding for a specific category-subject-age_mo combination
  - Filename format: `{subject_id}_{age_mo}_month_level_avg.npy`

### Category Lists
- **Exclusion/Inclusion files**: Plain text files with one category name per line
- **CDI words CSV**: CSV file with columns:
  - `uni_lemma`: Category name
  - `is_animate`, `is_bodypart`, `is_small`, `is_big`: Boolean flags

### RDM Matrices
- **NPY format**: Numpy array saved with `np.save()`
- **CSV format**: Pandas DataFrame with category names as index and columns

---

## Tips and Best Practices

1. **Matching Embeddings**: In `01_compute_average_embeddings.ipynb`, set `MATCH_FROM_CLIP_LIST = True` when comparing CLIP and DINOv3 to ensure the same images are used for both models.

2. **Filtering Categories**: Filter out low-quality categories in `02_filter_normalize_and_compute_rdm.ipynb` before computing correlations to get more reliable results.

3. **Hierarchical Clustering**: The `02_filter_normalize_and_compute_rdm.ipynb` notebook uses hierarchical clustering within category groups by default. Set `USE_CLUSTERING = False` if you prefer alphabetical sorting.

4. **Parallel Processing**: The `01_compute_average_embeddings.ipynb` notebook uses parallel loading by default. For very large datasets, you may want to adjust `NUM_WORKERS` in the configuration cell.

5. **Memory Usage**: For large embedding sets, consider processing in batches or using memory-efficient loading options. You can disable parallel processing by setting `USE_PARALLEL = False`.

6. **Visualization**: The RDM heatmaps are saved as high-resolution PNG files (300 DPI) suitable for publication.

7. **Notebook Execution**: 
   - Execute notebooks 01-04 sequentially as each step depends on outputs from previous steps
   - Notebooks 05 and 06 are independent and can be run separately (they require grouped embeddings from the preprocessing pipeline)
   - Make sure to update configuration paths in each notebook before running

8. **Normalization Strategy**:
   - **Notebook 05 (Individual Subject RDMs)**: Uses global normalization across all subjects to enable cross-subject RDM comparisons
   - **Notebook 06 (Developmental Trajectories)**: Uses per-subject, per-age-bin normalization to focus on relative structure changes within subjects over time
   - The normalization approach is chosen based on the analysis goal (cross-subject comparison vs. within-subject developmental tracking)

9. **Data Density Considerations**:
   - Both notebooks handle data density differences by filtering subjects/age bins with insufficient categories
   - Global normalization in notebook 05 naturally handles density differences (subjects with more data contribute more to global stats)
   - Adjust `min_categories_per_subject` and `min_categories_per_age_bin` based on your data distribution

10. **Configuration Management**: Keep track of your configuration settings for reproducibility. Consider saving configuration cells separately or documenting your parameter choices.

---

## Troubleshooting

### Common Issues

1. **"No matching categories found"**: Check that category names match exactly (case-sensitive, no extra whitespace). Verify your CDI words CSV file has the correct category names.

2. **"Matrix is not square"**: Ensure your RDM matrices are properly formatted symmetric matrices. This usually indicates an issue in Step 2.

3. **"File not found"**: Verify all file paths in the configuration cells are correct. Use absolute paths if relative paths don't work. Make sure outputs from previous steps exist before running subsequent notebooks.

4. **Pandas CSV export errors**: The notebooks include fallback mechanisms, but if you encounter persistent CSV errors, you may need to reinstall pandas or check file permissions.

5. **Memory errors**: For very large datasets, reduce `NUM_WORKERS` in the configuration cell or set `USE_PARALLEL = False` in `01_compute_average_embeddings.ipynb` for sequential processing.

6. **Kernel crashes**: If a notebook kernel crashes, restart it and re-run all cells from the beginning. Make sure all required packages are installed.

7. **Path issues**: If you're having trouble with paths, try using absolute paths in the configuration cells. Check that all input files exist before running each notebook.

8. **Missing embeddings**: For notebooks 05 and 06, if many embeddings are skipped, check:
   - That the grouped embeddings directory structure matches expected format
   - That filenames follow the pattern `{subject_id}_{age_mo}_month_level_avg.npy`
   - That the categories file (if used) contains valid category names matching folder names

9. **Normalization issues**: If RDM values seem unusual:
   - Verify that embeddings are being normalized correctly (check normalization cell output)
   - For notebook 05, ensure global normalization is computed across all subjects
   - For notebook 06, verify that each age bin is normalized independently

10. **Data density warnings**: If many subjects are excluded due to insufficient categories:
    - Lower the `min_categories_per_subject` or `min_categories_per_age_bin` thresholds
    - Check the data density analysis plots to understand the distribution
    - Consider whether your data quality is sufficient for RDM analysis

---

