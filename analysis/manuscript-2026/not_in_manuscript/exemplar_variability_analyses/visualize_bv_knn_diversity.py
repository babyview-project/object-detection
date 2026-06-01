#!/usr/bin/env python3
"""
Visualize BV within-category kNN diversity results.

Reads the summary CSVs produced by bv_within_category_knn_diversity.py and
optionally the centroid summary from bv_to_things_centroid_distances.py to
produce:

1. Bar plot: categories ranked by mean_knn_dist (low = more micro-structure).
   Options: all categories, or top/bottom N.
2. k=5 vs k=10: scatter of mean_knn_dist at k=5 vs k=10 (from multi_k summary).
3. kNN vs centroid spread: scatter mean_knn_dist vs mean_bv_to_bv_centroid to
   see categories that have high spread but low kNN (micro-structure) vs high
   spread and high kNN (no local consistency).
4. Per-exemplar distribution: violin/box of mean_knn_dist for selected
   categories (requires per-exemplar CSV).

Usage:
  cd analysis/manuscript-2026/exemplar_variability_analyses
  python visualize_bv_knn_diversity.py --embedding clip
  python visualize_bv_knn_diversity.py --embedding clip --centroid-summary bv_to_things_centroid_clip_summary.csv --violin-categories crayon,cat,zebra,chair
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def load_knn_summary(embedding: str, k: int, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"bv_within_category_knn_{embedding}_k{k}_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"kNN summary not found: {path}")
    return pd.read_csv(path)


def load_multi_k_summary(embedding: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"bv_within_category_knn_{embedding}_multi_k_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_centroid_summary(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path)


def plot_rank_bars(
    df: pd.DataFrame,
    out_path: Path,
    n_show: int = 40,
    show: str = "both",
    title_suffix: str = "",
):
    """Bar plot of categories by mean_knn_dist (ascending = most micro-structure first)."""
    # df is already sorted by mean_knn_dist ascending
    if show == "top":
        plot_df = df.head(n_show)
    elif show == "bottom":
        plot_df = df.tail(n_show).iloc[::-1].reset_index(drop=True)  # keep ascending for display
    else:
        plot_df = df.head(n_show)
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.22)))
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df["mean_knn_dist"], xerr=plot_df["std_knn_dist"], capsize=2, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["category"], fontsize=9)
    ax.set_xlabel("Mean kNN distance (lower = more micro-structure)")
    ax.set_ylabel("Category")
    ax.set_title(f"Within-category kNN diversity (ranked){title_suffix}")
    ax.invert_yaxis()  # top of list at top of plot
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_k5_vs_k10(multi_df: pd.DataFrame, out_path: Path, embedding: str):
    """Scatter: mean_knn_dist at k=5 vs k=10."""
    k5 = multi_df[multi_df["k_used"] == 5].set_index("category")["mean_knn_dist"]
    k10 = multi_df[multi_df["k_used"] == 10].set_index("category")["mean_knn_dist"]
    common = k5.index.intersection(k10.index)
    x = k5.loc[common].values
    y = k10.loc[common].values
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, alpha=0.6, s=25)
    # Diagonal
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Mean kNN distance (k=5)")
    ax.set_ylabel("Mean kNN distance (k=10)")
    ax.set_title(f"k=5 vs k=10 within-category kNN diversity ({embedding})")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_knn_vs_centroid_spread(
    knn_df: pd.DataFrame,
    centroid_df: pd.DataFrame,
    out_path: Path,
    embedding: str,
    k: int,
    label_n: int = 12,
):
    """Scatter: mean_knn_dist (x) vs mean_bv_to_bv_centroid (y). Label extremes."""
    merge = knn_df.merge(
        centroid_df[["category", "mean_bv_to_bv_centroid"]],
        on="category",
        how="inner",
    )
    if merge.empty:
        print("  No overlap between kNN and centroid summaries; skip kNN vs centroid plot.")
        return
    x = merge["mean_knn_dist"].values
    y = merge["mean_bv_to_bv_centroid"].values
    cats = merge["category"].values
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x, y, alpha=0.6, s=30)
    # Label extremes: low kNN, high kNN, high spread, low spread
    n = len(merge)
    low_knn = merge.nsmallest(label_n // 2, "mean_knn_dist")
    high_knn = merge.nlargest(label_n // 2, "mean_knn_dist")
    to_label = pd.concat([low_knn, high_knn]).drop_duplicates()
    for _, row in to_label.iterrows():
        ax.annotate(
            row["category"],
            (row["mean_knn_dist"], row["mean_bv_to_bv_centroid"]),
            fontsize=8,
            alpha=0.9,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean kNN distance (lower = more micro-structure)")
    ax.set_ylabel("Mean BV-to-BV centroid distance (overall spread)")
    ax.set_title(f"kNN diversity vs centroid spread ({embedding}, k={k})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_violins_per_category(
    exemplar_path: Path,
    categories: list[str],
    out_path: Path,
    k: int,
    embedding: str,
):
    """Violin plot of per-exemplar mean_knn_dist for selected categories."""
    df = pd.read_csv(exemplar_path)
    df = df[df["category"].isin(categories)]
    if df.empty:
        print("  No data for violin categories; skip.")
        return
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.2), 5))
    parts = ax.violinplot(
        [df[df["category"] == c]["mean_knn_dist"].values for c in categories],
        positions=range(len(categories)),
        showmeans=True,
        showmedians=True,
    )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel("Per-exemplar mean kNN distance")
    ax.set_xlabel("Category")
    ax.set_title(f"Within-category distribution of kNN distance (k={k}, {embedding})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize BV within-category kNN diversity")
    parser.add_argument("--embedding", type=str, default="clip", choices=("clip", "dinov3"))
    parser.add_argument("--k", type=int, default=5, help="Primary k for single-k plots")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory with kNN summary CSVs (default: script dir)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to save figures (default: script dir)")
    parser.add_argument("--centroid-summary", type=Path, default=None,
                        help="Path to bv_to_things_centroid_*_summary.csv for joint plot")
    parser.add_argument("--bar-n", type=int, default=40,
                        help="Number of categories in bar plot (default 40)")
    parser.add_argument("--violin-categories", type=str, default=None,
                        help="Comma-separated categories for violin plot (e.g. crayon,cat,zebra)")
    parser.add_argument("--no-multi-k", action="store_true",
                        help="Skip k5 vs k10 plot even if multi_k summary exists")
    args = parser.parse_args()

    data_dir = args.data_dir or SCRIPT_DIR
    out_dir = args.out_dir or SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"bv_within_category_knn_{args.embedding}"
    # Default centroid path if not given
    if args.centroid_summary is None:
        args.centroid_summary = data_dir / f"bv_to_things_centroid_{args.embedding}_summary.csv"

    knn_df = load_knn_summary(args.embedding, args.k, data_dir)
    multi_df = load_multi_k_summary(args.embedding, data_dir)
    centroid_df = load_centroid_summary(args.centroid_summary)

    # 1. Bar plot (top N by micro-structure = lowest mean_knn_dist)
    plot_rank_bars(
        knn_df,
        out_dir / f"{prefix}_k{args.k}_rank_bars.png",
        n_show=args.bar_n,
        title_suffix=f" (k={args.k})",
    )

    # 2. k5 vs k10
    if not args.no_multi_k and multi_df is not None and (multi_df["k_used"] == 10).any():
        plot_k5_vs_k10(
            multi_df,
            out_dir / f"{prefix}_k5_vs_k10_scatter.png",
            args.embedding,
        )

    # 3. kNN vs centroid spread
    if centroid_df is not None:
        plot_knn_vs_centroid_spread(
            knn_df,
            centroid_df,
            out_dir / f"{prefix}_k{args.k}_vs_centroid_spread.png",
            args.embedding,
            args.k,
        )

    # 4. Violins for selected categories
    if args.violin_categories:
        cats = [c.strip() for c in args.violin_categories.split(",") if c.strip()]
        exemplar_path = data_dir / f"{prefix}_k{args.k}_per_exemplar.csv"
        if exemplar_path.exists():
            plot_violins_per_category(
                exemplar_path,
                cats,
                out_dir / f"{prefix}_k{args.k}_violins_selected.png",
                args.k,
                args.embedding,
            )
        else:
            print(f"  Per-exemplar CSV not found: {exemplar_path}. Run with --save-exemplar-csv first.")

    print("Done.")


if __name__ == "__main__":
    main()
