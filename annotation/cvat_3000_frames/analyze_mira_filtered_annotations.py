#!/usr/bin/env python3
"""Quick analysis of Mira_filtered_category_annotation_results.csv."""

import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(__file__).resolve().parent / "annotation_results_merged.csv"
OUT_DIR = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(CSV_PATH)

    print("=" * 60)
    print("Mira filtered category annotation results — summary")
    print("=" * 60)

    # Overall
    n_rows = len(df)
    n_categories = df["category_name"].nunique()
    det_acc = df["correct_detection"].mean()
    cdi_acc = df["correct_cdi"].mean()
    both_correct = ((df["correct_detection"] == 1) & (df["correct_cdi"] == 1)).mean()
    agreement = (df["correct_detection"] == df["correct_cdi"]).mean()

    print(f"\nTotal rows:        {n_rows:,}")
    print(f"Unique categories: {n_categories}")
    print(f"Detection accuracy (correct_detection): {det_acc:.2%}")
    print(f"CDI accuracy (correct_cdi):             {cdi_acc:.2%}")
    print(f"Both correct:                          {both_correct:.2%}")
    print(f"Detection == CDI (agreement):           {agreement:.2%}")

    # Per-category stats
    df["both_correct"] = (df["correct_detection"] == 1) & (df["correct_cdi"] == 1)
    agg = df.groupby("category_name").agg(
        count=("correct_detection", "count"),
        det_acc=("correct_detection", "mean"),
        cdi_acc=("correct_cdi", "mean"),
        both_correct=("both_correct", "mean"),
    ).round(4)
    agg = agg.sort_values("count", ascending=False).reset_index()

    print("\n" + "=" * 60)
    print("Per-category summary (sorted by sample count)")
    print("=" * 60)
    print(agg.to_string(index=False))

    # Top / bottom by detection accuracy (min 5 samples)
    min_n = 5
    agg_min = agg[agg["count"] >= min_n].copy()
    if len(agg_min) > 0:
        top_det = agg_min.nlargest(10, "det_acc")[["category_name", "count", "det_acc", "cdi_acc"]]
        bot_det = agg_min.nsmallest(10, "det_acc")[["category_name", "count", "det_acc", "cdi_acc"]]
        print("\n" + "-" * 60)
        print(f"Top 10 categories by detection accuracy (min {min_n} samples)")
        print("-" * 60)
        print(top_det.to_string(index=False))
        print("\n" + "-" * 60)
        print(f"Bottom 10 categories by detection accuracy (min {min_n} samples)")
        print("-" * 60)
        print(bot_det.to_string(index=False))

    # Save per-category summary
    out_csv = OUT_DIR / "Mira_filtered_category_annotation_summary.csv"
    agg.to_csv(out_csv, index=False)
    print(f"\nPer-category summary saved to: {out_csv}")

    # Plots
    plot_overall_metrics(det_acc, cdi_acc, both_correct, agreement)
    plot_per_category_accuracy(agg)
    plot_detection_vs_cdi_scatter(agg)
    plot_sample_count_distribution(agg)
    print(f"\nPlots saved to: {OUT_DIR}")


def plot_overall_metrics(det_acc, cdi_acc, both_correct, agreement):
    """Bar chart of overall accuracy metrics."""
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["Detection", "CDI", "Both correct", "Agreement"]
    values = [det_acc, cdi_acc, both_correct, agreement]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"]
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy / rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.1%}", ha="center", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "mira_annotation_overall_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_category_accuracy(agg, top_n=30):
    """Bar chart of per-category detection and CDI accuracy (top N by sample count)."""
    plot_df = agg.nlargest(top_n, "count")
    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.28)))
    y_pos = np.arange(n) * 0.5
    h = 0.2
    ax.barh(y_pos - h / 2 - 0.02, plot_df["det_acc"], height=h, color="steelblue", alpha=0.85, label="Detection")
    ax.barh(y_pos + h / 2 + 0.02, plot_df["cdi_acc"], height=h, color="coral", alpha=0.85, label="CDI")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["category_name"], fontsize=9)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "mira_annotation_per_category_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_detection_vs_cdi_scatter(agg, min_count=5):
    """Scatter: detection accuracy vs CDI accuracy per category (point size = count)."""
    plot_df = agg[agg["count"] >= min_count].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = (plot_df["count"] / plot_df["count"].max() * 200 + 20).clip(20, 250)
    sc = ax.scatter(plot_df["det_acc"], plot_df["cdi_acc"], s=sizes, alpha=0.6, c="steelblue", edgecolors="black", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Detection accuracy")
    ax.set_ylabel("CDI accuracy")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "mira_annotation_detection_vs_cdi_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_count_distribution(agg):
    """Histogram of per-category sample counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(agg["count"], bins=min(40, max(10, agg["count"].nunique())), color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Samples per category")
    ax.set_ylabel("Number of categories")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "mira_annotation_sample_count_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
