#!/usr/bin/env python3
"""
CLIP vs DINOv3 comparison for BabyView: compare per–CDI-category metrics (displacement,
direction alignment, magnitude consistency) between CLIP and DINOv3 embedding spaces.
Reads stats from bv_things_results_*/bv_things_semantic_category_stats_*.csv (or recomputes
from coordinate CSVs) and saves comparison plots.

Usage:
  python clip_vs_dinov3_babyview_comparison.py [--method umap|tsne] [--out-dir DIR]
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_BASE = SCRIPT_DIR


def load_or_compute_stats(emb: str, method: str):
    """Load stats CSV if present; else load coordinates and compute stats. Returns DataFrame indexed by cdi_category."""
    folder = RESULTS_BASE / f"bv_things_results_{emb}"
    stats_path = folder / f"bv_things_semantic_category_stats_{method}.csv"
    if stats_path.exists():
        df = pd.read_csv(stats_path, index_col=0)
        return df
    # Recompute from coordinate CSV
    csv_path = folder / f"{method}_bv_things_coordinates.csv"
    if not csv_path.exists():
        return None
    from visualize_cdi_categories import analyze_bv_things_csv
    _, stats = analyze_bv_things_csv(csv_path)
    stats.to_csv(stats_path)
    return stats


def run_comparison(method: str, out_dir: Path):
    clip_stats = load_or_compute_stats("clip", method)
    dino_stats = load_or_compute_stats("dinov3", method)
    if clip_stats is None or dino_stats is None:
        print("Missing CLIP or DINOv3 stats; run visualize_tsne_umap_bv_things.py and/or visualize_cdi_categories.py first.")
        return
    # Align by common CDI categories
    common = clip_stats.index.intersection(dino_stats.index)
    clip_stats = clip_stats.loc[common]
    dino_stats = dino_stats.loc[common]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"clip_vs_dinov3_{method}"

    # 1) Scatter: CLIP vs DINOv3 displacement_mean
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(clip_stats["displacement_mean"], dino_stats["displacement_mean"], alpha=0.8, s=60)
    for cat in common:
        ax.annotate(cat, (clip_stats.loc[cat, "displacement_mean"], dino_stats.loc[cat, "displacement_mean"]),
                    fontsize=7, alpha=0.9, xytext=(3, 3), textcoords="offset points")
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.02]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("CLIP: Mean displacement (BV → THINGs)")
    ax.set_ylabel("DINOv3: Mean displacement (BV → THINGs)")
    ax.set_title(f"CLIP vs DINOv3: Mean displacement by CDI category ({method.upper()})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_displacement_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_dir / f"{prefix}_displacement_scatter.png")

    # 2) Scatter: CLIP vs DINOv3 direction_alignment
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(clip_stats["direction_alignment"], dino_stats["direction_alignment"], alpha=0.8, s=60)
    for cat in common:
        ax.annotate(cat, (clip_stats.loc[cat, "direction_alignment"], dino_stats.loc[cat, "direction_alignment"]),
                    fontsize=7, alpha=0.9, xytext=(3, 3), textcoords="offset points")
    lims = [0, 1.02]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("CLIP: Direction alignment")
    ax.set_ylabel("DINOv3: Direction alignment")
    ax.set_title(f"CLIP vs DINOv3: Direction alignment by CDI category ({method.upper()})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_direction_alignment_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_dir / f"{prefix}_direction_alignment_scatter.png")

    # 3) Scatter: CLIP vs DINOv3 magnitude_consistency
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(clip_stats["magnitude_consistency"], dino_stats["magnitude_consistency"], alpha=0.8, s=60)
    for cat in common:
        ax.annotate(cat, (clip_stats.loc[cat, "magnitude_consistency"], dino_stats.loc[cat, "magnitude_consistency"]),
                    fontsize=7, alpha=0.9, xytext=(3, 3), textcoords="offset points")
    lims = [0, 1.02]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("CLIP: Magnitude consistency")
    ax.set_ylabel("DINOv3: Magnitude consistency")
    ax.set_title(f"CLIP vs DINOv3: Magnitude consistency by CDI category ({method.upper()})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_magnitude_consistency_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_dir / f"{prefix}_magnitude_consistency_scatter.png")

    # 4) Summary: correlation of metrics across categories (CLIP vs DINOv3)
    r_disp = np.corrcoef(clip_stats["displacement_mean"], dino_stats["displacement_mean"])[0, 1]
    r_dir = np.corrcoef(clip_stats["direction_alignment"], dino_stats["direction_alignment"])[0, 1]
    r_mag = np.corrcoef(clip_stats["magnitude_consistency"], dino_stats["magnitude_consistency"])[0, 1]
    summary_path = out_dir / f"{prefix}_correlations.txt"
    with open(summary_path, "w") as f:
        f.write(f"CLIP vs DINOv3 correlation across CDI categories ({method.upper()})\n")
        f.write(f"  displacement_mean:    r = {r_disp:.4f}\n")
        f.write(f"  direction_alignment: r = {r_dir:.4f}\n")
        f.write(f"  magnitude_consistency: r = {r_mag:.4f}\n")
    print("Saved:", summary_path)
    print("Done. Plots in", out_dir.resolve())


def main():
    ap = argparse.ArgumentParser(description="CLIP vs DINOv3 BabyView comparison")
    ap.add_argument("--method", default="umap", choices=["umap", "tsne"], help="Reduction method")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: manuscript-2026/plots)")
    args = ap.parse_args()
    out_dir = args.out_dir or (RESULTS_BASE / "plots")
    run_comparison(args.method, out_dir)


if __name__ == "__main__":
    main()
