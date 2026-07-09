#!/usr/bin/env python3
"""
Compute correlation between CLIP and DINOv3 category-wise BV–THINGS correlations.

For each category (e.g. "cat"), we have:
  - bv_things_clip: correlation between BabyView and THINGS embeddings for that category (CLIP)
  - bv_things_dinov3: same for DINOv3

This script computes how correlated these two vectors are across categories:
  - Do categories that have high CLIP correlation also tend to have high DINOv3 correlation?

Uses the same embedding CSVs as the correlation reports so we get all 163 categories.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_embeddings_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Load embedding CSV: index = category, columns = dim_*."""
    df = pd.read_csv(csv_path, index_col=0)
    if df.index.name is None or str(df.index.name).startswith("Unnamed"):
        if "Unnamed: 0" in df.columns:
            df = df.set_index("Unnamed: 0")
    dim_cols = [c for c in df.columns if re.match(r"dim_\d+", str(c))]
    if not dim_cols:
        dim_cols = [c for c in df.columns if c != "category"]
    X = df[dim_cols].values.astype(np.float64)
    categories = [str(i).strip().lower() for i in df.index]
    return categories, X


def compute_pearson_per_category(
    bv_csv: Path, things_csv: Path, category_include_file: Path | None
) -> dict[str, float]:
    """Return dict category -> pearson_r for BV vs THINGS embeddings."""
    cats_bv, X_bv = load_embeddings_csv(bv_csv)
    cats_th, X_th = load_embeddings_csv(things_csv)
    set_th = set(cats_th)
    matching = [c for c in cats_bv if c in set_th]
    if category_include_file and category_include_file.exists():
        with open(category_include_file) as f:
            included = set(line.strip().lower() for line in f if line.strip())
        matching = [c for c in matching if c in included]
    cat_to_idx_bv = {c: i for i, c in enumerate(cats_bv)}
    cat_to_idx_th = {c: i for i, c in enumerate(cats_th)}
    out = {}
    for cat in matching:
        i_bv = cat_to_idx_bv[cat]
        i_th = cat_to_idx_th[cat]
        v1 = X_bv[i_bv]
        v2 = X_th[i_th]
        mask = np.isfinite(v1) & np.isfinite(v2)
        v1c, v2c = v1[mask], v2[mask]
        if len(v1c) >= 3:
            r, _ = stats.pearsonr(v1c, v2c)
        else:
            r = np.nan
        out[cat] = r
    return out


def main():
    base = Path(__file__).resolve().parent
    comp_dir = base.parent  # bv_things_comp_12252025
    category_include = base.parent.parent.parent.parent / "data" / "things_bv_overlap_categories_exclude_zero_precisions.txt"

    bv_clip = comp_dir / "bv_clip_filtered_zscored_hierarchical_163cats" / "normalized_filtered_embeddings_alphabetical.csv"
    things_clip = comp_dir / "things_clip_filtered_zscored_hierarchical_163cats" / "normalized_filtered_embeddings_alphabetical.csv"
    bv_dinov3 = comp_dir / "bv_dinov3_filtered_zscored_hierarchical_163cats" / "normalized_filtered_embeddings_alphabetical.csv"
    things_dinov3 = comp_dir / "things_dinov3_filtered_zscored_hierarchical_163cats" / "normalized_filtered_embeddings_alphabetical.csv"

    clip_r = compute_pearson_per_category(bv_clip, things_clip, category_include)
    dinov3_r = compute_pearson_per_category(bv_dinov3, things_dinov3, category_include)

    common = sorted(set(clip_r) & set(dinov3_r))
    clip_vals = np.array([clip_r[c] for c in common])
    dinov3_vals = np.array([dinov3_r[c] for c in common])
    # Drop NaNs for correlation
    valid = np.isfinite(clip_vals) & np.isfinite(dinov3_vals)
    n_valid = int(np.sum(valid))
    clip_clean = clip_vals[valid]
    dinov3_clean = dinov3_vals[valid]

    pearson_r, pearson_p = stats.pearsonr(clip_clean, dinov3_clean)
    spearman_r, spearman_p = stats.spearmanr(clip_clean, dinov3_clean)

    print("=" * 60)
    print("CLIP vs DINOv3 category-wise BV–THINGS correlations")
    print("=" * 60)
    print(f"Categories (shared, valid): {n_valid} of {len(common)}")
    print()
    print("Correlation across categories (CLIP r vs DINOv3 r):")
    print(f"  Pearson  r = {pearson_r:.4f}, p = {pearson_p:.2e}")
    print(f"  Spearman r = {spearman_r:.4f}, p = {spearman_p:.2e}")
    print()
    print("Interpretation: categories that have higher BV–THINGS agreement")
    print("in CLIP tend to have higher agreement in DINOv3 (and vice versa).")
    print("=" * 60)

    out_csv = base / "clip_vs_dinov3_category_correlations.csv"
    with open(out_csv, "w") as f:
        f.write("category,clip_pearson_r,dinov3_pearson_r\n")
        for c in common:
            f.write(f"{c},{clip_r[c]:.6f},{dinov3_r[c]:.6f}\n")
    print(f"Wrote aligned table: {out_csv}")


if __name__ == "__main__":
    main()
