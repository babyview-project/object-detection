#!/usr/bin/env python3
"""
Visualize CDI semantic categories for BabyView vs THINGs: displacement, direction alignment,
and magnitude consistency by CDI category. Reads coordinate CSVs from bv_things_results_*/
and saves plots to plots/ (or --out-dir).

Usage:
  python visualize_cdi_categories.py [--method umap|tsne] [--emb clip|dinov3|both] [--out-dir DIR]
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_BASE = SCRIPT_DIR


def load_bv_things_coordinates(csv_path: Path):
    """Load a *_{tsne,umap}_bv_things_coordinates.csv; return (df, bv_x, bv_y, things_x, things_y)."""
    df = pd.read_csv(csv_path)
    coord_cols = [c for c in df.columns if c not in ("category", "cdi_category")]
    bv_x = [c for c in coord_cols if c.startswith("bv_") and c.endswith("_x")][0]
    bv_y = [c for c in coord_cols if c.startswith("bv_") and c.endswith("_y")][0]
    things_x = [c for c in coord_cols if c.startswith("things_") and c.endswith("_x")][0]
    things_y = [c for c in coord_cols if c.startswith("things_") and c.endswith("_y")][0]
    return df, bv_x, bv_y, things_x, things_y


def direction_alignment(ux, uy):
    """Mean resultant length of unit vectors (ux, uy)."""
    if len(ux) == 0:
        return np.nan
    mx, my = np.mean(ux), np.mean(uy)
    return np.sqrt(mx * mx + my * my)


def analyze_bv_things_csv(csv_path: Path):
    """Load CSV, compute displacement and direction alignment by cdi_category. Returns (df, displacement_stats)."""
    df, bv_x, bv_y, things_x, things_y = load_bv_things_coordinates(csv_path)
    dx = (df[things_x] - df[bv_x]).values
    dy = (df[things_y] - df[bv_y]).values
    magnitude = np.sqrt(dx * dx + dy * dy)
    df = df.copy()
    df["displacement"] = magnitude
    eps = 1e-12
    length = np.maximum(magnitude, eps)
    df["dir_x"] = dx / length
    df["dir_y"] = dy / length
    grp = df.groupby("cdi_category", sort=True)
    disp = grp["displacement"].agg(["mean", "std", "count", "median"]).round(4)
    disp.columns = ["displacement_mean", "displacement_std", "n_words", "displacement_median"]
    alignment = grp.apply(lambda g: direction_alignment(g["dir_x"].values, g["dir_y"].values), include_groups=False)
    disp["direction_alignment"] = alignment.round(4)
    disp["displacement_cv"] = (disp["displacement_std"] / disp["displacement_mean"]).round(4)
    disp["magnitude_consistency"] = (1 / (1 + disp["displacement_cv"])).round(4)
    disp = disp.sort_values("displacement_mean", ascending=False)
    return df, disp


def plot_displacement_by_category(displacement_stats: pd.DataFrame, out_path: Path, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = displacement_stats.index
    x = np.arange(len(cats))
    ax.bar(x, displacement_stats["displacement_mean"], yerr=displacement_stats["displacement_std"].fillna(0),
           capsize=3, color="steelblue", alpha=0.8, edgecolor="navy")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Mean displacement (BV → THINGs)")
    ax.set_title(f"BV vs THINGs: Mean displacement by semantic category (CDI){title_suffix}")
    ax.set_xlabel("Semantic category")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def plot_direction_alignment(displacement_stats: pd.DataFrame, out_path: Path, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = displacement_stats.index
    x = np.arange(len(cats))
    ax.bar(x, displacement_stats["direction_alignment"], color="coral", alpha=0.8, edgecolor="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Direction alignment (0 = scattered, 1 = same direction)")
    ax.set_title(f"BV vs THINGs: Direction consistency within semantic category{title_suffix}")
    ax.set_xlabel("Semantic category")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def plot_magnitude_consistency(displacement_stats: pd.DataFrame, out_path: Path, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = displacement_stats.index
    x = np.arange(len(cats))
    ax.bar(x, displacement_stats["magnitude_consistency"], color="seagreen", alpha=0.8, edgecolor="darkgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("Magnitude consistency (0–1)")
    ax.set_title(f"BV vs THINGs: Magnitude consistency within semantic category{title_suffix}")
    ax.set_xlabel("Semantic category")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def run_one(emb: str, method: str, out_dir: Path):
    """Load coordinates for one embedding/method, compute stats, save plots and optional CSV."""
    folder = RESULTS_BASE / f"bv_things_results_{emb}"
    csv_path = folder / f"{method}_bv_things_coordinates.csv"
    if not csv_path.exists():
        print("Skip", emb, method, "(no CSV)")
        return None
    _, stats = analyze_bv_things_csv(csv_path)
    suffix = f" ({emb.upper()} {method.upper()})"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{emb}_{method}"
    plot_displacement_by_category(stats, out_dir / f"{prefix}_displacement_by_cdi_category.png", suffix)
    plot_direction_alignment(stats, out_dir / f"{prefix}_direction_alignment_by_cdi_category.png", suffix)
    plot_magnitude_consistency(stats, out_dir / f"{prefix}_magnitude_consistency_by_cdi_category.png", suffix)
    stats_path = folder / f"bv_things_semantic_category_stats_{method}.csv"
    stats.to_csv(stats_path)
    print("Saved:", stats_path)
    return stats


def main():
    ap = argparse.ArgumentParser(description="Visualize CDI categories for BV vs THINGs")
    ap.add_argument("--method", default="umap", choices=["umap", "tsne"], help="Reduction method")
    ap.add_argument("--emb", default="both", choices=["clip", "dinov3", "both"], help="Embedding(s) to plot")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots (default: manuscript-2026/plots)")
    args = ap.parse_args()
    out_dir = args.out_dir or (RESULTS_BASE / "plots")
    embs = ["clip", "dinov3"] if args.emb == "both" else [args.emb]
    for emb in embs:
        run_one(emb, args.method, out_dir)
    print("Done. Plots in", out_dir.resolve())


if __name__ == "__main__":
    main()
