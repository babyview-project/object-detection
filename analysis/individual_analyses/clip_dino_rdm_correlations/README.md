# CLIP vs DINO Individual Subject RDM Correlations

This directory contains results from correlating individual subject RDMs generated using CLIP embeddings vs DINO embeddings.

## Files

- `clip_dino_rdm_correlations.csv`: Correlation results for each subject
- `clip_dino_correlation_visualization.png`: Visualization of correlation results
- `clip_dino_correlation_visualization.pdf`: PDF version of visualization

## Methodology

### Data Sources
- **CLIP RDMs**: `individual_subject_rdms_clip/npy/rdm_{subject_id}.npy`
- **DINO RDMs**: `individual_subject_rdms_dinov3/npy/rdm_{subject_id}.npy`

### Correlation Computation

For each subject:
1. Load both CLIP and DINO RDMs (saved as CSV with category labels)
2. Extract available categories from each RDM (categories with non-NaN values)
3. Find common categories present in both RDMs
4. Extract submatrices for common categories only
5. Compute correlations using:
   - **Spearman correlation**: Rank-based, robust to outliers
   - **Pearson correlation**: Linear correlation
6. Only use upper triangle (excluding diagonal) to avoid double-counting

### Key Features

- **Handles missing categories**: Uses only categories present in both RDMs
- **Preserves category order**: Maintains predefined category ordering for consistency
- **Robust to outliers**: Spearman correlation is less sensitive to extreme values

## Results Summary

Based on 31 subjects:

- **Spearman Correlation**: Mean = 0.852, Std = 0.039, Range = [0.749, 0.904]
- **Pearson Correlation**: Mean = 0.868, Std = 0.035, Range = [0.780, 0.913]
- **Common Categories**: Mean = 148.3, Std = 16.8, Range = [99, 162]

## Interpretation

High correlations (mean ~0.85) indicate that:
- CLIP and DINO embeddings capture similar representational structures
- Individual subjects show consistent patterns across both embedding types
- The representational dissimilarity structure is robust to the choice of embedding model

## Scripts

- `correlate_clip_dino_rdms.py`: Main script to compute correlations
- `visualize_clip_dino_correlations.py`: Script to create visualizations

## Usage

```bash
# Compute correlations
python3 correlate_clip_dino_rdms.py

# Create visualizations
python3 visualize_clip_dino_correlations.py
```
