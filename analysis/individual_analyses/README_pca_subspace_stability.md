# PCA Subspace Stability Analysis

## Motivation

**Question (Clíona):** How much is the **across-family similarity** of individual subject RDMs driven by the embedding spaces (CLIP or DINOv3), even though we feed different input images? Are models capturing nuances in the visual input, or are they converging those inputs?

We compare **individual subject RDMs** (one RDM per subject, aggregated across age like notebook 06), not developmental trajectory (younger vs older).

**Idea:** Restrict to the subspace spanned by the **observed object crops** (age-month level embeddings), then recompute individual subject RDMs and **across-subject similarity** (mean pairwise RDM correlation) in this subspace.

## What the analysis does

1. **Load** all normalized age-month level embeddings (same source as notebook 05/06).
2. **Stack** them into a matrix and **fit PCA** (e.g. 95% variance).
3. **Project** all embeddings onto the PCA subspace.
4. **Build one RDM per subject** in full space and in PCA subspace (aggregate across all age_mo per subject–category, like notebook 06).
5. **Compute across-subject similarity**: mean pairwise Spearman correlation between subject RDMs (using only overlapping categories per pair).
6. **Compare** mean pairwise correlation in full space vs in the PCA subspace.

## Interpretation

- **Mean pairwise correlation drops in PCA subspace** → The full embedding space may be contributing shared structure across subjects beyond the observed inputs (embedding space driving across-family similarity).
- **Mean pairwise correlation similar in PCA subspace** → The observed input diversity is sufficient to explain across-family similarity (models capturing nuances).

## Files

- **`pca_subspace_stability.py`** – Core logic: load, PCA fit, project, build subject RDMs, mean pairwise RDM correlation. Can be run from the command line.
- **`10_pca_subspace_stability.ipynb`** – Runs the analysis and plots comparison (full space vs PCA subspace).

## Usage

### From command line

```bash
cd analysis/individual_analyses
python pca_subspace_stability.py \
  --embeddings_dir /path/to/clip_embeddings_grouped_by_age-mo_normalized \
  --output_dir individual_subject_rdms_clip_pca_subspace \
  --n_components 0.95
```

Use the same `embeddings_dir` and category list as in notebook 06.

### From notebook

1. Open `10_pca_subspace_stability.ipynb`.
2. Set `normalized_embeddings_dir` (and optionally `embedding_type`) to match your paths.
3. Run all cells. The script computes both full-space and PCA-subspace subject RDMs and mean pairwise correlation; the notebook loads the summary and plots bar chart + pairwise correlation histograms.

## Outputs

- `across_subject_rdm_similarity_summary.csv` – Rows for "full" and "pca_subspace": mean_pairwise_rdm_correlation, n_subjects, n_pairs.
- `pairwise_rdm_correlations_full.csv` – Per subject-pair RDM correlation (full space).
- `pairwise_rdm_correlations_pca_subspace.csv` – Per subject-pair RDM correlation (PCA subspace).
- `pca_mean.npy`, `pca_components.npy` – Fitted PCA.
- `pca_variance.csv` – Per-component and cumulative variance explained.
- `across_subject_similarity_full_vs_pca_subspace.png` – From the notebook: bar chart and pairwise correlation histograms.

## Dependencies

Same as notebooks 05–07: `numpy`, `pandas`, `scikit-learn`, `scipy`, `tqdm`.
