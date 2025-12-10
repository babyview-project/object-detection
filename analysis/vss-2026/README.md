# VSS 2026 RDM Analysis Scripts

This directory contains scripts for computing and analyzing Representational Dissimilarity Matrices (RDMs) for CLIP and DINOv3 category embeddings.

## Overview

The analysis pipeline consists of four main scripts:

1. **`compute_clip_category_rdm.py`** - Computes pairwise RDM matrices from category embeddings
2. **`reorganize_clip_rdm.py`** - Filters and reorganizes RDMs by category type with optional hierarchical clustering
3. **`correlate_rdm_matrices.py`** - Correlates two RDM matrices (e.g., CLIP vs DINOv3)
4. **`correlate_category_embeddings.py`** - Correlates category-level average embeddings between two embedding files

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

### Step 1: Compute RDM Matrices

First, compute RDM matrices for your embeddings (CLIP and/or DINOv3):

```bash
# For CLIP embeddings
python compute_clip_category_rdm.py \
  --embedding_list /path/to/clip_embeddings_list.txt \
  --embeddings_dir /path/to/clip_embeddings \
  --output_dir ./clip_rdm_results \
  --cdi_path ./data/cdi_words.csv \
  --embedding_type clip

# For DINOv3 embeddings (matching CLIP list to ensure same images)
python compute_clip_category_rdm.py \
  --embedding_list /path/to/clip_embeddings_list.txt \
  --embeddings_dir /path/to/dinov3_embeddings \
  --output_dir ./dinov3_rdm_results \
  --cdi_path ./data/cdi_words.csv \
  --embedding_type dinov3 \
  --match_from_list
```

### Step 2: Filter and Reorganize RDMs (Optional)

Filter out low-quality categories and reorganize by type:

```bash
# Using exclusion file
python reorganize_clip_rdm.py \
  --npz_path ./clip_rdm_results/category_average_embeddings.npz \
  --exclusion_file ./data/categories_with_zero_precision.txt \
  --cdi_path ./data/cdi_words.csv \
  --output_dir ./clip_rdm_results_filtered \
  --save_dendrograms

# Or using inclusion file
python reorganize_clip_rdm.py \
  --npz_path ./clip_rdm_results/category_average_embeddings.npz \
  --inclusion_file ./data/categories_to_include.txt \
  --cdi_path ./data/cdi_words.csv \
  --output_dir ./clip_rdm_results_filtered
```

### Step 3: Correlate RDM Matrices

Compare RDM matrices between different models:

```bash
python correlate_rdm_matrices.py \
  --rdm1 ./clip_rdm_results_filtered/distance_matrix_filtered.npy \
  --rdm2 ./dinov3_rdm_results_filtered/distance_matrix_filtered.npy \
  --output ./rdm_correlation_results.txt
```

### Step 4: Correlate Category Embeddings (Optional)

Compare category-level embeddings directly:

```bash
python correlate_category_embeddings.py \
  --embeddings1 ./clip_rdm_results/category_average_embeddings.npz \
  --embeddings2 ./dinov3_rdm_results/category_average_embeddings.npz \
  --output ./category_embedding_correlations.txt \
  --save_per_category
```

---

## Script Documentation

### 1. `compute_clip_category_rdm.py`

Computes pairwise Representational Dissimilarity Matrices (RDMs) for CLIP or DINOv3 category embeddings.

**Key Features:**
- Loads embeddings from `.npy` files
- Computes category average embeddings
- Generates cosine similarity and distance matrices
- Organizes categories by type (animate, bodyparts, small, big objects)
- Creates visualization heatmaps

**Required Arguments:**
- `--embeddings_dir`: Base directory containing embedding files
- `--output_dir`: Output directory for results
- `--cdi_path`: Path to CDI words CSV file

**Optional Arguments:**
- `--embedding_list`: Text file with embedding paths (one per line)
- `--match_from_list`: Match filenames from embedding_list to embeddings_dir
- `--embedding_type`: Type of embeddings (`clip` or `dinov3`, default: `clip`)
- `--num_workers`: Number of parallel workers (default: auto-detect, max 16)
- `--no_parallel`: Disable parallel loading

**Example Usage:**

```bash
# CLIP embeddings with list file
python compute_clip_category_rdm.py \
  --embedding_list /data/clip_embeddings_list.txt \
  --embeddings_dir /data/clip_embeddings \
  --output_dir ./clip_rdm_results \
  --cdi_path ./data/cdi_words.csv \
  --embedding_type clip

# DINOv3 embeddings (auto-scan directory)
python compute_clip_category_rdm.py \
  --embeddings_dir /data/dinov3_embeddings \
  --output_dir ./dinov3_rdm_results \
  --cdi_path ./data/cdi_words.csv \
  --embedding_type dinov3

# DINOv3 embeddings matching CLIP list
python compute_clip_category_rdm.py \
  --embedding_list /data/clip_embeddings_list.txt \
  --embeddings_dir /data/dinov3_embeddings \
  --output_dir ./dinov3_rdm_results \
  --cdi_path ./data/cdi_words.csv \
  --embedding_type dinov3 \
  --match_from_list
```

**Output Files:**
- `category_average_embeddings.npz`: Category average embeddings
- `category_average_embeddings.csv`: Category averages (CSV format)
- `similarity_matrix.npy` / `.csv`: Cosine similarity matrix
- `distance_matrix.npy` / `.csv`: Cosine distance matrix (1 - similarity)
- `rdm_full.png`: Full RDM heatmap
- `rdm_organized_by_type.png`: RDM organized by category type

---

### 2. `reorganize_clip_rdm.py`

Filters and reorganizes CLIP category RDM by loading saved embeddings and excluding/including categories.

**Key Features:**
- Filters categories by exclusion or inclusion lists
- Organizes categories by type with hierarchical clustering
- Creates filtered RDM visualizations
- Generates dendrograms for category groups
- Identifies top similar/dissimilar category pairs

**Required Arguments:**
- `--npz_path`: Path to NPZ file with category average embeddings
- Either `--exclusion_file` OR `--inclusion_file` must be provided

**Optional Arguments:**
- `--exclusion_file`: Text file with categories to exclude (one per line)
- `--inclusion_file`: Text file with categories to include (one per line)
- `--cdi_path`: Path to CDI words CSV (default: `./data/cdi_words.csv`)
- `--output_dir`: Output directory (default: `./clip_rdm_results_filtered`)
- `--organization_file`: Use custom category organization from file
- `--no_clustering`: Disable hierarchical clustering within groups
- `--save_dendrograms`: Save dendrogram plots for each category group

**Example Usage:**

```bash
# Filter by exclusion
python reorganize_clip_rdm.py \
  --npz_path ./clip_rdm_results/category_average_embeddings.npz \
  --exclusion_file ./data/low_precision_categories.txt \
  --cdi_path ./data/cdi_words.csv \
  --output_dir ./clip_rdm_results_filtered \
  --save_dendrograms

# Filter by inclusion
python reorganize_clip_rdm.py \
  --npz_path ./clip_rdm_results/category_average_embeddings.npz \
  --inclusion_file ./data/high_quality_categories.txt \
  --cdi_path ./data/cdi_words.csv \
  --output_dir ./clip_rdm_results_filtered

# Use custom organization file
python reorganize_clip_rdm.py \
  --npz_path ./clip_rdm_results/category_average_embeddings.npz \
  --exclusion_file ./data/low_precision_categories.txt \
  --organization_file ./clip_rdm_results/category_organization.txt \
  --output_dir ./clip_rdm_results_filtered
```

**Output Files:**
- `similarity_matrix_filtered.npy` / `.csv`: Filtered similarity matrix
- `distance_matrix_filtered.npy` / `.csv`: Filtered distance matrix
- `rdm_organized_filtered.png`: Filtered RDM heatmap (viridis colormap)
- `rdm_organized_filtered_coolwarm.png`: Filtered RDM heatmap (coolwarm colormap)
- `dendrogram_*.png` / `.pdf`: Dendrograms for each category group (if `--save_dendrograms`)
- `top_N_similar_pairs.png` / `.txt`: Top N most similar category pairs
- `top_N_dissimilar_pairs.png` / `.txt`: Top N most dissimilar category pairs

---

### 3. `correlate_rdm_matrices.py`

Correlates the lower triangle of two RDM matrices using Pearson and Spearman correlations.

**Key Features:**
- Supports both `.npy` and `.csv` input formats
- Computes Pearson and Spearman correlations
- Option to include/exclude diagonal elements
- Option to use upper or lower triangle

**Required Arguments:**
- `--rdm1`: Path to first RDM matrix file (.npy or .csv)
- `--rdm2`: Path to second RDM matrix file (.npy or .csv)

**Optional Arguments:**
- `--output`: Output file path (default: print to stdout)
- `--include_diagonal`: Include diagonal elements in correlation
- `--use_upper_triangle`: Use upper triangle instead of lower triangle

**Example Usage:**

```bash
# Basic correlation (NPY files)
python correlate_rdm_matrices.py \
  --rdm1 ./clip_rdm_results_filtered/distance_matrix_filtered.npy \
  --rdm2 ./dinov3_rdm_results_filtered/distance_matrix_filtered.npy \
  --output ./rdm_correlation_results.txt

# Using CSV files
python correlate_rdm_matrices.py \
  --rdm1 ./clip_rdm_results_filtered/distance_matrix_filtered.csv \
  --rdm2 ./dinov3_rdm_results_filtered/distance_matrix_filtered.csv \
  --output ./rdm_correlation_results.txt

# Include diagonal
python correlate_rdm_matrices.py \
  --rdm1 ./clip_rdm_results_filtered/distance_matrix_filtered.npy \
  --rdm2 ./dinov3_rdm_results_filtered/distance_matrix_filtered.npy \
  --include_diagonal \
  --output ./rdm_correlation_results.txt
```

**Output:**
- Correlation statistics (Pearson r, p-value; Spearman r, p-value)
- Matrix information and statistics
- Results saved to text file or printed to stdout

---

### 4. `correlate_category_embeddings.py`

Correlates within-category average embeddings between two embedding files.

**Key Features:**
- Computes correlations for each matching category
- Multiple similarity metrics: Pearson, Spearman, cosine similarity, Euclidean distance
- Summary statistics across all categories
- Optional per-category detailed results

**Required Arguments:**
- `--embeddings1`: Path to first embeddings file (.npz)
- `--embeddings2`: Path to second embeddings file (.npz)

**Optional Arguments:**
- `--output`: Output file path (default: print to stdout)
- `--save_per_category`: Save detailed per-category results to CSV
- `--min_correlation`: Only report categories with correlation above threshold

**Example Usage:**

```bash
# Basic correlation
python correlate_category_embeddings.py \
  --embeddings1 ./clip_rdm_results/category_average_embeddings.npz \
  --embeddings2 ./dinov3_rdm_results/category_average_embeddings.npz \
  --output ./category_embedding_correlations.txt

# Save per-category results
python correlate_category_embeddings.py \
  --embeddings1 ./clip_rdm_results/category_average_embeddings.npz \
  --embeddings2 ./dinov3_rdm_results/category_average_embeddings.npz \
  --output ./category_embedding_correlations.txt \
  --save_per_category

# Filter by minimum correlation
python correlate_category_embeddings.py \
  --embeddings1 ./clip_rdm_results/category_average_embeddings.npz \
  --embeddings2 ./dinov3_rdm_results/category_average_embeddings.npz \
  --output ./category_embedding_correlations.txt \
  --min_correlation 0.5
```

**Output:**
- Summary statistics (mean, std, median, min, max for each metric)
- Top 10 and bottom 10 categories by Pearson correlation
- Per-category CSV file (if `--save_per_category`)

---

## File Format Requirements

### Embedding Files
- **Input embeddings**: `.npy` files containing 1D embedding vectors
- **Category average embeddings**: `.npz` files with keys:
  - `embeddings`: numpy array of shape (n_categories, embedding_dim)
  - `categories`: numpy array of category names

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

1. **Matching Embeddings**: Use `--match_from_list` when comparing CLIP and DINOv3 to ensure the same images are used for both models.

2. **Filtering Categories**: Filter out low-quality categories before computing correlations to get more reliable results.

3. **Hierarchical Clustering**: The `reorganize_clip_rdm.py` script uses hierarchical clustering within category groups by default. Disable with `--no_clustering` if you prefer alphabetical sorting.

4. **Parallel Processing**: The `compute_clip_category_rdm.py` script uses parallel loading by default. For very large datasets, you may want to adjust `--num_workers`.

5. **Memory Usage**: For large embedding sets, consider processing in batches or using memory-efficient loading options.

6. **Visualization**: The RDM heatmaps are saved as high-resolution PNG files (300 DPI) suitable for publication.

---

## Troubleshooting

### Common Issues

1. **"No matching categories found"**: Check that category names match exactly (case-sensitive, no extra whitespace).

2. **"Matrix is not square"**: Ensure your RDM matrices are properly formatted symmetric matrices.

3. **"File not found"**: Verify all file paths are correct and use absolute paths if needed.

4. **Pandas CSV export errors**: The scripts include fallback mechanisms, but if you encounter persistent CSV errors, you may need to reinstall pandas.

5. **Memory errors**: For very large datasets, reduce `--num_workers` or use `--no_parallel` for sequential processing.

---

## Citation

If you use these scripts in your research, please cite appropriately and acknowledge the original sources for CLIP and DINOv3 models.

