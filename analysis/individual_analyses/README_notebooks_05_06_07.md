# Individual Subject RDM Analysis Workflow

## Overview

This workflow creates Representational Dissimilarity Matrices (RDMs) for individual subjects to analyze how object representations are structured and how they develop over time. The analysis consists of three sequential notebooks that normalize embeddings, compute individual subject RDMs, and analyze developmental trajectories.

## Workflow Overview

```
Notebook 05: Normalize Grouped Embeddings
    ↓
    (Normalized embeddings)
    ↓
Notebook 06: Individual Subject RDMs (aggregated across all ages)
    ↓
Notebook 07: Developmental Trajectory RDMs (split by age)
```

### Workflow Steps

1. **Notebook 05** normalizes age-month level embeddings across all subjects and categories, computing global normalization statistics
2. **Notebook 06** uses the normalized embeddings to create a single RDM per subject, aggregating data across all age months
3. **Notebook 07** extends the analysis by splitting each subject's data into "younger" and "older" periods based on median age, creating two RDMs per subject to track developmental changes

## Notebook 05: Normalize Grouped Embeddings

### Purpose

Normalizes CLIP embeddings at the age-month level (before aggregation). This ensures that embeddings are standardized across all subjects and age months, which is critical for downstream analyses that compare representations across different time points and individuals.

### Key Steps

1. **Load all age-month level embeddings**: Loads all `{subject_id}_{age_mo}_month_level_avg.npy` files from grouped age-month embeddings across all categories (163 categories)

2. **Compute global normalization statistics**: 
   - Excludes subject 00270001 from normalization statistics computation
   - Computes feature-wise mean and standard deviation across ALL age-month level embeddings (~34,550 embeddings from 31 subjects across age months 6-37)

3. **Normalize embeddings**: Applies z-score normalization to each embedding: `(embedding - global_mean) / global_std`

4. **Save normalized embeddings**: Saves normalized embeddings to output directory, preserving the original category folder structure

### Key Features

- **Feature-wise normalization**: Normalizes each embedding dimension independently across all subjects/age bins
- **Age-month level normalization**: Normalizes at the raw data level (before aggregation), preserving age_mo information
- **Subject exclusion**: Subject 00270001 is excluded from normalization statistics but embeddings are still normalized
- **Directory structure preservation**: Maintains original category-based folder structure in output directory

### Inputs

- **Embeddings directory**: Path to grouped age-month embeddings (e.g., `facebook_dinov3-vitb16-pretrain-lvd1689m_grouped_by_age-mo`)
- **Categories file** (optional): Text file listing categories to include (default: `../../data/things_bv_overlap_categories_exclude_zero_precisions.txt`)

### Outputs

- **Normalized embeddings directory**: Directory containing normalized embeddings with same structure as input
  - Format: `{category}/{subject_id}_{age_mo}_month_level_avg.npy`
  - Example: `facebook_dinov3-vitb16-pretrain-lvd1689m_grouped_by_age-mo_normalized`

### Dependencies

- Requires grouped age-month embeddings from previous preprocessing steps
- Uses numpy for array operations
- Uses pathlib for file management

### Notes

- This normalization happens at the raw data level, preserving age_mo information in normalized form for downstream analyses
- The global statistics are computed across all subjects and age months (excluding subject 00270001) to ensure consistent normalization
- Normalized embeddings are used as input for notebooks 06 and 07

---

## Notebook 06: Compute and Visualize Individual Subject RDMs

### Purpose

Creates Representational Dissimilarity Matrices (RDMs) for each individual subject. Each subject's RDM shows the similarity structure of object categories based on their averaged embeddings, allowing comparison of how different subjects organize object representations. This analysis aggregates data across all age months for each subject to create a single RDM per subject.

### Key Steps

1. **Load normalized embeddings**: Loads normalized age-month level embeddings from notebook 05

2. **Aggregate embeddings per subject**: 
   - Averages embeddings across all age_mo bins for each subject-category combination
   - Creates a single embedding vector per subject-category pair

3. **Organize categories**: 
   - Uses either a predefined category list (for consistent ordering across subjects) or automatic organization by type
   - Supports hierarchical clustering within category groups for better visualization

4. **Compute RDMs**: 
   - Computes cosine distance between all category pairs for each subject
   - Handles missing categories by placing NA values for categories not present for each subject
   - Ensures all RDMs have the same dimensions for easy comparison

5. **Visualize and save**: 
   - Creates RDM visualizations with NA cells blacked out
   - Saves RDMs as both CSV files and NumPy arrays
   - Optionally saves dendrograms for category groups

### Key Features

- **Normalized embeddings**: Uses pre-normalized embeddings from notebook 05
- **Consistent category ordering**: Supports loading a predefined category list to ensure all subjects' RDMs have the same category order for easy visual comparison
- **Missing category handling**: Places NA values for categories not present for each subject, ensuring all RDMs have the same structure
- **NA visualization**: Blackouts NA cells in RDM visualizations to clearly indicate missing data
- **Data density handling**: Subjects with more data get more reliable RDMs, but all RDMs maintain the same structure
- **Age-month aggregation**: Averages embeddings across all age_mo bins for each subject-category combination

### Inputs

- **Normalized embeddings directory**: Path to normalized embeddings from notebook 05
- **CDI words CSV**: Path to CDI words file for category type information (default: `../../data/cdi_words.csv`)
- **Predefined category list** (optional): Text file with category order (one per line) for consistent RDM ordering
- **Categories file** (optional): Text file listing categories to include

### Outputs

- **Individual subject RDMs**: 
  - CSV files: `{output_dir}/rdm_{subject_id}.csv`
  - NumPy arrays: `{output_dir}/rdm_{subject_id}.npy`
  - Metadata: `{output_dir}/metadata_{subject_id}.csv`
- **Visualizations**: 
  - RDM plots: `{output_dir}/individual_rdm_plots/rdm_{subject_id}.png`
  - Dendrograms (if enabled): `{output_dir}/individual_rdm_plots/dendrograms/{group_name}_dendrogram.png`
- **Summary statistics**: `{output_dir}/summary_statistics.csv`

### Configuration Options

- **Use clustering**: Enable hierarchical clustering within category groups
- **Use predefined category list**: Load category order from external file for consistent ordering
- **Save dendrograms**: Save dendrogram plots for each category group

### Dependencies

- Requires normalized embeddings from notebook 05
- Uses scikit-learn for distance computations
- Uses scipy for hierarchical clustering
- Uses matplotlib and seaborn for visualization

### Notes

- Each subject gets one RDM that aggregates data across all their age months
- RDMs use cosine distance to measure dissimilarity between category embeddings
- Missing categories are handled gracefully with NA values, allowing comparison across subjects with different data availability
- For developmental trajectory analysis (how RDMs change over time), see notebook 07

---

## Notebook 07: Developmental Trajectory RDM Analysis

### Purpose

Creates two Representational Dissimilarity Matrices (RDMs) for each individual subject, split by a median age threshold computed across all participants. This allows tracking how object representations change developmentally within each subject over time. Unlike notebook 06 which creates a single RDM per subject aggregating all age months, this notebook splits each subject's data into "younger" and "older" periods based on a median age threshold.

### Key Steps

1. **Load normalized embeddings**: Loads normalized age-month level embeddings from notebook 05

2. **Calculate median age threshold**: 
   - Computes the overall median age across all participants
   - Uses this threshold to split each subject's data into two age bins

3. **Split data by age**: 
   - For each subject, splits data into "younger" (age_mo <= median) and "older" (age_mo > median) bins
   - Applies minimum category threshold per age bin to ensure sufficient data

4. **Compute RDMs**: 
   - Computes one RDM for the younger period and one for the older period for each subject
   - Uses cosine distance between category pairs
   - Handles missing categories with NA values

5. **Visualize and analyze**: 
   - Creates visualizations for each subject's developmental trajectory
   - Compares RDMs between younger and older periods within subjects
   - Computes cross-subject correlations and category group correlations

### Key Features

- **Median split**: Uses overall median age across all participants to split each subject's data
- **Two RDMs per subject**: One for "younger" period, one for "older" period
- **Data density handling**: Minimum category threshold per age bin to ensure reliable RDMs
- **Trajectory analysis**: Compare RDMs between younger and older periods to see developmental changes
- **Missing data handling**: Only includes subjects with sufficient data in both bins
- **Cross-subject comparisons**: Analyzes correlations between subjects' developmental trajectories

### Inputs

- **Normalized embeddings directory**: Path to normalized embeddings from notebook 05
- **CDI words CSV**: Path to CDI words file for category type information (default: `../../data/cdi_words.csv`)
- **Predefined category list** (optional): Text file with category order (one per line) for consistent RDM ordering
- **Categories file** (optional): Text file listing categories to include

### Outputs

- **Developmental trajectory RDMs**: 
  - CSV files: `{output_dir}/{subject_id}/rdm_age_{age_mo}.csv`
  - NumPy arrays: `{output_dir}/{subject_id}/rdm_age_{age_mo}.npy`
  - Metadata: `{output_dir}/{subject_id}/metadata_age_{age_mo}.csv`
- **Visualizations**: 
  - Individual subject plots: `{output_dir}/{subject_id}/rdm_comparison.png`
  - Cross-subject correlations: `{output_dir}/cross_kid_correlations_visualization.png`
  - Category group correlations: `{output_dir}/cross_kid_category_group_correlations_visualization.png`
- **Analysis results**: 
  - Cross-kid correlations: `{output_dir}/cross_kid_correlations.csv`
  - Category group correlations: `{output_dir}/cross_kid_category_group_correlations.csv`

### Configuration Options

- **Minimum categories per age bin**: Threshold for minimum number of categories required to compute RDM (default: 8)
- **Use clustering**: Enable hierarchical clustering within category groups
- **Use predefined category list**: Load category order from external file for consistent ordering
- **Save dendrograms**: Save dendrogram plots for each category group

### Dependencies

- Requires normalized embeddings from notebook 05
- Uses scikit-learn for distance computations
- Uses scipy for hierarchical clustering and correlation analysis
- Uses matplotlib and seaborn for visualization

### Notes

- The median age threshold is computed across all participants, ensuring consistent splitting across subjects
- Subjects must have sufficient data in both age bins to be included in the analysis
- This analysis complements notebook 06 by providing a developmental perspective on how representations change over time
- The output directory name includes the embedding type (e.g., `developmental_trajectory_rdms_clip` or `developmental_trajectory_rdms_dinov3`)

---

## Relationship Between Notebooks

- **Notebook 05** provides the foundation by normalizing embeddings that are used by both notebooks 06 and 07
- **Notebook 06** creates single RDMs per subject (aggregated across all ages), providing a snapshot of each subject's overall representation structure
- **Notebook 07** extends notebook 06 by splitting data into age bins, enabling analysis of how representations develop over time within subjects
- Both notebooks 06 and 07 can be run on the same normalized embeddings to provide complementary views of the data

## Common Configuration

All notebooks share some common configuration:

- **Excluded subject**: Subject 00270001 is excluded from normalization statistics and analyses
- **Categories**: Uses 163 categories from `things_bv_overlap_categories_exclude_zero_precisions.txt`
- **Embedding types**: Can work with both CLIP and DINOv3 embeddings (detected automatically from directory paths)
- **Category ordering**: Supports predefined category lists for consistent ordering across analyses
