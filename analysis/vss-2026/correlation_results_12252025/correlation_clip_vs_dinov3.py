#!/usr/bin/env python3
"""
Compute correlation between CLIP and DINOv3 category-wise BV–THINGS correlations.

For each category (e.g. "cat"), we have:
  - bv_things_clip: correlation between BabyView and THINGS embeddings for that category (CLIP)
  - bv_things_dinov3: same for DINOv3

This script computes how correlated these two vectors are across categories:
  - Do categories that have high CLIP correlation also tend to have high DINOv3 correlation?
"""

import re
from pathlib import Path

import numpy as np
from scipy import stats


def parse_correlation_file(path: Path) -> dict[str, float]:
    """Parse the text report and return dict of category -> pearson_r."""
    text = path.read_text()
    # Lines like: "   1. zebra                          pearson_r=0.754053, ..."
    pattern = re.compile(
        r"^\s*\d+\.\s+(\w+)\s+pearson_r=([-\d.]+)",
        re.MULTILINE,
    )
    out = {}
    for m in pattern.finditer(text):
        cat, r = m.group(1), float(m.group(2))
        out[cat] = r
    return out


def main():
    base = Path(__file__).resolve().parent
    clip_path = base / "bv_things_clip_category_embeddings_correlations.txt"
    dinov3_path = base / "bv_things_dinov3_category_embeddings_correlations.txt"

    clip_r = parse_correlation_file(clip_path)
    dinov3_r = parse_correlation_file(dinov3_path)

    # Align by category
    common = sorted(set(clip_r) & set(dinov3_r))
    assert len(common) > 0, "No common categories"
    clip_vals = np.array([clip_r[c] for c in common])
    dinov3_vals = np.array([dinov3_r[c] for c in common])

    # Correlation between the two correlation vectors
    pearson_r, pearson_p = stats.pearsonr(clip_vals, dinov3_vals)
    spearman_r, spearman_p = stats.spearmanr(clip_vals, dinov3_vals)

    print("=" * 60)
    print("CLIP vs DINOv3 category-wise BV–THINGS correlations")
    print("=" * 60)
    print(f"Categories (shared): {len(common)}")
    print()
    print("Correlation across categories (CLIP r vs DINOv3 r):")
    print(f"  Pearson  r = {pearson_r:.4f}, p = {pearson_p:.2e}")
    print(f"  Spearman r = {spearman_r:.4f}, p = {spearman_p:.2e}")
    print()
    print("Interpretation: categories that have higher BV–THINGS agreement")
    print("in CLIP tend to have ___ agreement in DINOv3 (and vice versa).")
    print("=" * 60)

    # Optional: save aligned table for inspection
    out_csv = base / "clip_vs_dinov3_category_correlations.csv"
    with open(out_csv, "w") as f:
        f.write("category,clip_pearson_r,dinov3_pearson_r\n")
        for c in common:
            f.write(f"{c},{clip_r[c]:.6f},{dinov3_r[c]:.6f}\n")
    print(f"Wrote aligned table: {out_csv}")


if __name__ == "__main__":
    main()
