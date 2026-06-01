#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


CCN_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = CCN_DIR / "plotC_knn_diversity_outputs"

CLIP_SUMMARY_CSV = OUT_DIR / "bv_within_category_knn_clip_k5_summary.csv"
COMPARE_CSV = OUT_DIR / "bv_within_category_knn_clip_vs_dinov3_k5_comparison.csv"


def set_vector_export_style() -> None:
    # Keep text editable in PDF/PS output.
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["savefig.dpi"] = 300


def save_pdf(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved vector PDF: {out_path}")


def make_rank_high_to_low_pdf() -> None:
    df = pd.read_csv(CLIP_SUMMARY_CSV).sort_values("mean_knn_dist", ascending=False).reset_index(drop=True)
    df["rank_high_to_low"] = df.index + 1

    fig, ax = plt.subplots(figsize=(10, 28))
    ax.barh(df["category"], df["mean_knn_dist"], color="#1f77b4")
    ax.invert_yaxis()
    ax.set_xlabel("Mean within-category kNN distance (k=5)")
    ax.set_ylabel("Category")
    ax.set_title("CLIP category diversity rank (high to low)")

    out_pdf = OUT_DIR / "bv_within_category_knn_clip_k5_rank_high_to_low.pdf"
    save_pdf(fig, out_pdf)


def make_top_bottom_extremes_pdf() -> None:
    df = pd.read_csv(CLIP_SUMMARY_CSV).sort_values("mean_knn_dist", ascending=False).reset_index(drop=True)
    df["rank_high_to_low"] = df.index + 1

    def make_extremes_df(src: pd.DataFrame, n: int) -> pd.DataFrame:
        top = src.head(n).copy()
        top["group"] = f"Top {n}"
        bottom = src.tail(n).copy().sort_values("mean_knn_dist", ascending=True)
        bottom["group"] = f"Bottom {n}"
        return pd.concat([top, bottom], ignore_index=True)

    def plot_extremes(ax: plt.Axes, src: pd.DataFrame, n: int) -> None:
        e = make_extremes_df(src, n)
        labels = [f"{r.category} (#{int(r.rank_high_to_low)})" for r in e.itertuples()]
        colors = ["#1f77b4" if g.startswith("Top") else "#d62728" for g in e["group"]]
        ax.barh(labels, e["mean_knn_dist"], color=colors)
        ax.set_title(f"Top/Bottom {n}")
        ax.set_xlabel("Mean within-category kNN distance")
        ax.invert_yaxis()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    plot_extremes(axes[0], df, 5)
    plot_extremes(axes[1], df, 10)
    fig.suptitle("Category diversity extremes, CLIP k=5")

    out_pdf = OUT_DIR / "bv_within_category_knn_clip_k5_top_bottom_extremes_5_10.pdf"
    save_pdf(fig, out_pdf)


def make_clip_vs_dino_comparison_pdf() -> None:
    df = pd.read_csv(COMPARE_CSV)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        df["clip_mean_knn_dist"],
        df["dinov3_mean_knn_dist"],
        s=35,
        alpha=0.8,
        color="#2a9d8f",
        edgecolor="black",
        linewidth=0.4,
    )
    lo = min(df["clip_mean_knn_dist"].min(), df["dinov3_mean_knn_dist"].min())
    hi = max(df["clip_mean_knn_dist"].max(), df["dinov3_mean_knn_dist"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#666666", linewidth=1)
    ax.set_xlabel("CLIP mean kNN distance")
    ax.set_ylabel("DINOv3 mean kNN distance")
    ax.set_title("Within-category diversity: DINOv3 vs CLIP (k=5)")

    out_pdf = OUT_DIR / "bv_within_category_knn_clip_vs_dinov3_k5_comparison.pdf"
    save_pdf(fig, out_pdf)


def main() -> None:
    set_vector_export_style()
    make_rank_high_to_low_pdf()
    make_top_bottom_extremes_pdf()
    make_clip_vs_dino_comparison_pdf()


if __name__ == "__main__":
    main()
