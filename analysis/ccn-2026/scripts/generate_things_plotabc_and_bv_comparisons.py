#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CCN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CCN_DIR.parent.parent
DEFAULT_INPUT_DIR = CCN_DIR / "plotC_knn_diversity_outputs" / "new_things_embeddings_20260428"
DEFAULT_OUT_DIR = CCN_DIR / "plot_things_and_bv_comparisons_outputs"
DEFAULT_SEMANTIC_CSV = REPO_ROOT / "data" / "long_tailed_dist_prop_included_categories.csv"

CDI_SEMANTIC_COLORS = {
    "animals": "#4DB8A8",
    "body_parts": "#E87A5F",
    "clothing": "#9B7EC8",
    "food_drink": "#E8A54C",
    "furniture_rooms": "#6BAB7A",
    "household": "#D97B9E",
    "outside": "#5B9BD5",
    "people": "#E8C44C",
    "toys": "#B07CC8",
    "vehicles": "#6BA3D5",
    "other": "#8B9A9E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate THINGS Plot A/B/C analog figures and BV-vs-THINGS comparison visualizations "
            "from local/global metric CSVs."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--valid-set", default="valid129", choices=["valid85", "valid129"])
    parser.add_argument("--top-n", type=int, default=5, help="Plot A analog: top and bottom category count.")
    parser.add_argument("--semantic-csv", type=Path, default=DEFAULT_SEMANTIC_CSV)
    return parser.parse_args()


def read_metric_tables(input_dir: Path, embedding: str, k: int, valid_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    things_path = input_dir / f"things_{embedding}_local_global_k{k}_{valid_set}.csv"
    cmp_path = input_dir / f"things_vs_bv_{embedding}_local_global_k{k}_{valid_set}.csv"
    if not things_path.exists():
        raise FileNotFoundError(f"Missing THINGS metrics file: {things_path}")
    if not cmp_path.exists():
        raise FileNotFoundError(f"Missing comparison metrics file: {cmp_path}")
    return pd.read_csv(things_path), pd.read_csv(cmp_path)


def _load_semantics(semantic_csv: Path) -> pd.DataFrame:
    if not semantic_csv.exists():
        return pd.DataFrame(columns=["category", "cdi_semantic"])
    sem = pd.read_csv(semantic_csv, usecols=["category", "cdi_semantic"]).dropna(subset=["category"])
    sem["category"] = sem["category"].astype(str).str.strip().str.lower()
    sem["cdi_semantic"] = sem["cdi_semantic"].astype(str).str.strip()
    return sem


def _attach_semantic_and_sizes(df: pd.DataFrame, semantic_df: pd.DataFrame, n_col: str) -> pd.DataFrame:
    out = df.copy()
    out["category"] = out["category"].astype(str).str.strip().str.lower()
    out = out.merge(semantic_df, on="category", how="left")
    out["cdi_semantic"] = out["cdi_semantic"].fillna("other")
    out["dot_color"] = out["cdi_semantic"].map(lambda s: CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]))
    nmin, nmax = out[n_col].min(), out[n_col].max()
    if np.isclose(nmin, nmax):
        out["dot_size"] = 70.0
    else:
        out["dot_size"] = 40.0 + (out[n_col] - nmin) / (nmax - nmin) * 220.0
    return out


def save_plot_a_analog(
    clip_df: pd.DataFrame, dino_df: pd.DataFrame, out_dir: Path, top_n: int, semantic_df: pd.DataFrame
) -> None:
    merged = clip_df[["category", "global_dispersion", "mean_knn_dist"]].rename(
        columns={"global_dispersion": "global_clip", "mean_knn_dist": "knn_clip"}
    ).merge(
        dino_df[["category", "global_dispersion", "mean_knn_dist"]].rename(
            columns={"global_dispersion": "global_dino", "mean_knn_dist": "knn_dino"}
        ),
        on="category",
        how="inner",
    )
    merged["n_exemplars"] = merged["category"].map(clip_df.set_index("category")["n_exemplars"]).fillna(0).astype(float)
    merged["global_mean_across_models"] = merged[["global_clip", "global_dino"]].mean(axis=1)
    merged["knn_mean_across_models"] = merged[["knn_clip", "knn_dino"]].mean(axis=1)
    merged = merged.sort_values("global_mean_across_models", ascending=True).reset_index(drop=True)
    merged["rank_low_to_high_variability"] = np.arange(1, len(merged) + 1)
    merged = _attach_semantic_and_sizes(merged, semantic_df, "n_exemplars")

    selected = pd.concat([merged.head(top_n), merged.tail(top_n)], axis=0).copy()
    selected["selection_group"] = ["low_variability"] * top_n + ["high_variability"] * top_n
    selected_csv = out_dir / "plotA_things_selected_categories_low_to_high_variability.csv"
    selected.to_csv(selected_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ordered = merged.sort_values("global_mean_across_models", ascending=False)
    axes[0].barh(ordered["category"], ordered["global_mean_across_models"], color="#4c78a8")
    axes[0].set_title("THINGS Plot A analog: global variability ranking")
    axes[0].set_xlabel("Mean centroid distance (CLIP+DINOv3)")
    axes[0].set_ylabel("Category")

    for cat in selected["category"].tolist():
        idx = ordered.index[ordered["category"] == cat]
        if len(idx):
            y = list(ordered["category"]).index(cat)
            axes[0].get_yticklabels()[y].set_color("#d62728")

    axes[1].scatter(
        merged["global_mean_across_models"],
        merged["knn_mean_across_models"],
        c=merged["dot_color"],
        s=merged["dot_size"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_title("THINGS Plot A analog: selected extremes")
    axes[1].set_xlabel("Mean centroid distance (CLIP+DINOv3)")
    axes[1].set_ylabel("Mean kNN distance (CLIP+DINOv3)")
    for r in selected.itertuples():
        axes[1].text(
            r.global_mean_across_models,
            r.knn_mean_across_models,
            f" {r.category}",
            fontsize=8,
            va="center",
        )

    out_png = out_dir / "plotA_things_low_to_high_variability.png"
    out_pdf = out_dir / "plotA_things_low_to_high_variability.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_plot_b_analog(
    things_df: pd.DataFrame, embedding: str, out_dir: Path, semantic_df: pd.DataFrame
) -> None:
    df = things_df.sort_values("global_dispersion", ascending=False).reset_index(drop=True)
    df = _attach_semantic_and_sizes(df, semantic_df, "n_exemplars")
    colors = df["dot_color"].tolist()

    fig_h = max(12, 0.22 * len(df))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.barh(df["category"], df["global_dispersion"], color=colors, alpha=0.95)
    ax.invert_yaxis()
    ax.set_title(
        f"THINGS Plot B analog ({embedding.upper()}): distance-to-centroid by category",
        fontsize=13,
    )
    ax.set_xlabel("Mean distance to category centroid")
    ax.set_ylabel("Category")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.25)

    present_semantics = [s for s in CDI_SEMANTIC_COLORS.keys() if s in set(df["cdi_semantic"])]
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=CDI_SEMANTIC_COLORS[s],
            markeredgecolor="none",
            markersize=8,
            label=s,
        )
        for s in present_semantics
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="CDI semantic", loc="lower right", frameon=True)

    out_png = out_dir / f"plotB_things_centroid_distance_ranked_{embedding}.png"
    out_pdf = out_dir / f"plotB_things_centroid_distance_ranked_{embedding}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_plot_c_analog(things_df: pd.DataFrame, embedding: str, out_dir: Path) -> None:
    df = things_df.sort_values("mean_knn_dist", ascending=False).reset_index(drop=True)
    df["rank_high_to_low"] = np.arange(1, len(df) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].plot(df["rank_high_to_low"], df["mean_knn_dist"], marker="o", linewidth=1.5, markersize=3)
    axes[0].set_title(f"THINGS Plot C analog ({embedding.upper()}): local kNN diversity rank")
    axes[0].set_xlabel("Rank (high to low mean kNN distance)")
    axes[0].set_ylabel("Mean within-category kNN distance")
    axes[0].grid(alpha=0.25)

    n_ext = min(10, len(df) // 2)
    top = df.head(n_ext).copy()
    top["group"] = "Top"
    bottom = df.tail(n_ext).copy().sort_values("mean_knn_dist", ascending=True)
    bottom["group"] = "Bottom"
    ext = pd.concat([top, bottom], axis=0)
    labels = [f"{r.category} (#{int(r.rank_high_to_low)})" for r in ext.itertuples()]
    colors = ["#1f77b4"] * len(top) + ["#d62728"] * len(bottom)
    axes[1].barh(labels, ext["mean_knn_dist"], color=colors, alpha=0.9)
    axes[1].set_title(f"THINGS Plot C analog ({embedding.upper()}): top/bottom diversity")
    axes[1].set_xlabel("Mean within-category kNN distance")
    axes[1].invert_yaxis()

    out_png = out_dir / f"plotC_things_knn_diversity_{embedding}.png"
    out_pdf = out_dir / f"plotC_things_knn_diversity_{embedding}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_bv_vs_things_comparison(
    cmp_df: pd.DataFrame, embedding: str, out_dir: Path, semantic_df: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cmp_df = cmp_df.copy()
    cmp_df["n_exemplars"] = cmp_df[["things_n_exemplars", "bv_n_exemplars"]].mean(axis=1)
    cmp_df = _attach_semantic_and_sizes(cmp_df, semantic_df, "n_exemplars")

    axes[0].scatter(
        cmp_df["things_global_dispersion"],
        cmp_df["bv_global_dispersion"],
        c=cmp_df["dot_color"],
        s=cmp_df["dot_size"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    mn = min(cmp_df["things_global_dispersion"].min(), cmp_df["bv_global_dispersion"].min())
    mx = max(cmp_df["things_global_dispersion"].max(), cmp_df["bv_global_dispersion"].max())
    axes[0].plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.0, color="black")
    axes[0].set_title(f"BV vs THINGS global dispersion ({embedding.upper()})")
    axes[0].set_xlabel("THINGS mean centroid distance")
    axes[0].set_ylabel("BabyView mean centroid distance")

    delta_knn = cmp_df["bv_mean_knn_dist"] - cmp_df["things_mean_knn_dist"]
    axes[1].hist(delta_knn, bins=25, alpha=0.9, color="#1f77b4")
    axes[1].axvline(delta_knn.mean(), color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"BV - THINGS local kNN deltas ({embedding.upper()})")
    axes[1].set_xlabel("Delta mean kNN distance")
    axes[1].set_ylabel("Category count")

    out_png = out_dir / f"bv_vs_things_measures_comparison_{embedding}.png"
    out_pdf = out_dir / f"bv_vs_things_measures_comparison_{embedding}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_cross_model_bv_things_panel(
    clip_cmp: pd.DataFrame, dino_cmp: pd.DataFrame, out_dir: Path, semantic_df: pd.DataFrame
) -> None:
    clip_df = clip_cmp[["category", "things_mean_knn_dist", "bv_mean_knn_dist", "things_global_dispersion", "bv_global_dispersion"]].rename(
        columns={
            "things_mean_knn_dist": "things_knn_clip",
            "bv_mean_knn_dist": "bv_knn_clip",
            "things_global_dispersion": "things_global_clip",
            "bv_global_dispersion": "bv_global_clip",
        }
    )
    dino_df = dino_cmp[["category", "things_mean_knn_dist", "bv_mean_knn_dist", "things_global_dispersion", "bv_global_dispersion"]].rename(
        columns={
            "things_mean_knn_dist": "things_knn_dino",
            "bv_mean_knn_dist": "bv_knn_dino",
            "things_global_dispersion": "things_global_dino",
            "bv_global_dispersion": "bv_global_dino",
        }
    )
    merged = clip_df.merge(dino_df, on="category", how="inner")
    merged["n_exemplars"] = (
        clip_cmp[["category", "things_n_exemplars", "bv_n_exemplars"]]
        .assign(n_exemplars=lambda x: x[["things_n_exemplars", "bv_n_exemplars"]].mean(axis=1))
        .set_index("category")["n_exemplars"]
        .reindex(merged["category"])
        .to_numpy()
    )
    merged = _attach_semantic_and_sizes(merged, semantic_df, "n_exemplars")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(
        merged["things_global_clip"],
        merged["bv_global_clip"],
        c=merged["dot_color"],
        s=merged["dot_size"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0, 0].set_title("CLIP global: BV vs THINGS")
    axes[0, 0].set_xlabel("THINGS centroid distance")
    axes[0, 0].set_ylabel("BV centroid distance")

    axes[0, 1].scatter(
        merged["things_knn_clip"],
        merged["bv_knn_clip"],
        c=merged["dot_color"],
        s=merged["dot_size"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0, 1].set_title("CLIP local kNN: BV vs THINGS")
    axes[0, 1].set_xlabel("THINGS mean kNN distance")
    axes[0, 1].set_ylabel("BV mean kNN distance")

    axes[1, 0].scatter(
        merged["things_global_dino"],
        merged["bv_global_dino"],
        c=merged["dot_color"],
        s=merged["dot_size"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1, 0].set_title("DINOv3 global: BV vs THINGS")
    axes[1, 0].set_xlabel("THINGS centroid distance")
    axes[1, 0].set_ylabel("BV centroid distance")

    axes[1, 1].scatter(
        merged["things_knn_dino"],
        merged["bv_knn_dino"],
        c=merged["dot_color"],
        s=merged["dot_size"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1, 1].set_title("DINOv3 local kNN: BV vs THINGS")
    axes[1, 1].set_xlabel("THINGS mean kNN distance")
    axes[1, 1].set_ylabel("BV mean kNN distance")

    out_png = out_dir / "bv_vs_things_measures_comparison_cross_model_2x2.png"
    out_pdf = out_dir / "bv_vs_things_measures_comparison_cross_model_2x2.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_things_variability_correlation_2x2(
    things_clip: pd.DataFrame, things_dino: pd.DataFrame, out_dir: Path, semantic_df: pd.DataFrame
) -> None:
    merged = things_clip[["category", "mean_knn_dist", "global_dispersion", "n_exemplars"]].rename(
        columns={
            "mean_knn_dist": "knn_clip",
            "global_dispersion": "global_clip",
            "n_exemplars": "n_exemplars_clip",
        }
    ).merge(
        things_dino[["category", "mean_knn_dist", "global_dispersion", "n_exemplars"]].rename(
            columns={
                "mean_knn_dist": "knn_dino",
                "global_dispersion": "global_dino",
                "n_exemplars": "n_exemplars_dino",
            }
        ),
        on="category",
        how="inner",
    )
    merged["n_exemplars"] = merged[["n_exemplars_clip", "n_exemplars_dino"]].mean(axis=1)
    merged = _attach_semantic_and_sizes(merged, semantic_df, "n_exemplars")

    def _annotate_spread_labels(
        ax: plt.Axes, df: pd.DataFrame, xcol: str, ycol: str, n_labels: int = 6, n_middle: int = 2
    ) -> None:
        z = (
            (df[xcol] - df[xcol].mean()) / (df[xcol].std(ddof=0) + 1e-9)
        ) ** 2 + (
            (df[ycol] - df[ycol].mean()) / (df[ycol].std(ddof=0) + 1e-9)
        ) ** 2
        tmp = df.assign(_outlier_score=z)
        outliers = tmp.sort_values("_outlier_score", ascending=False).head(n_labels)
        middle = tmp.sort_values("_outlier_score", ascending=True).head(n_middle)
        label_df = pd.concat([outliers, middle], axis=0).drop_duplicates(subset=["category"])
        offsets = [(20, 14), (-24, 14), (20, -16), (-26, -16), (26, 0), (-28, 0)]
        for i, r in enumerate(label_df.itertuples()):
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(
                str(r.category),
                (getattr(r, xcol), getattr(r, ycol)),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.75, alpha=0.9),
            )

    def _panel(ax: plt.Axes, xcol: str, ycol: str, title: str, xlabel: str, ylabel: str) -> None:
        ax.scatter(
            merged[xcol],
            merged[ycol],
            c=merged["dot_color"],
            s=merged["dot_size"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )
        pear = merged[[xcol, ycol]].corr().iloc[0, 1]
        spear = merged[[xcol, ycol]].corr(method="spearman").iloc[0, 1]
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.text(
            0.02,
            0.98,
            f"Pearson={pear:+.2f}, Spearman rho={spear:+.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#cccccc", alpha=0.85),
        )
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.18, linewidth=0.6)
        _annotate_spread_labels(ax, merged, xcol, ycol, n_labels=6)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _panel(
        axes[0, 0],
        "global_clip",
        "global_dino",
        "THINGS global dispersion: CLIP vs DINOv3",
        "CLIP global dispersion",
        "DINOv3 global dispersion",
    )
    _panel(
        axes[0, 1],
        "knn_clip",
        "knn_dino",
        "THINGS local kNN: CLIP vs DINOv3",
        "CLIP mean kNN distance",
        "DINOv3 mean kNN distance",
    )
    _panel(
        axes[1, 0],
        "global_clip",
        "knn_clip",
        "THINGS CLIP: global vs local",
        "CLIP global dispersion",
        "CLIP mean kNN distance",
    )
    _panel(
        axes[1, 1],
        "global_dino",
        "knn_dino",
        "THINGS DINOv3: global vs local",
        "DINOv3 global dispersion",
        "DINOv3 mean kNN distance",
    )

    semantic_present = [s for s in CDI_SEMANTIC_COLORS.keys() if s in set(merged["cdi_semantic"])]
    semantic_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CDI_SEMANTIC_COLORS[s],
            markeredgecolor="none",
            markersize=8,
            label=s,
        )
        for s in semantic_present
    ]
    if semantic_handles:
        fig.legend(
            handles=semantic_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(6, len(semantic_handles)),
            frameon=True,
            title="CDI semantic",
            prop={"size": 10},
            title_fontsize=11,
        )

    quantiles = np.quantile(merged["n_exemplars"], [0.25, 0.5, 0.75]).tolist()
    size_handles = []
    for q in quantiles:
        if np.isclose(merged["n_exemplars"].min(), merged["n_exemplars"].max()):
            s = 70.0
        else:
            s = 40.0 + (q - merged["n_exemplars"].min()) / (
                merged["n_exemplars"].max() - merged["n_exemplars"].min()
            ) * 220.0
        ms = max(4.0, np.sqrt(s))
        size_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#777777",
                markeredgecolor="none",
                markersize=ms,
                label=f"n={int(round(q))}",
            )
        )
    if size_handles:
        fig.legend(
            handles=size_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=True,
            title="Exemplar count",
            prop={"size": 10},
            title_fontsize=11,
        )

    out_png = out_dir / "things_variability_correlation_2x2_semantic_size.png"
    out_pdf = out_dir / "things_variability_correlation_2x2_semantic_size.pdf"
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.1, wspace=0.24, hspace=0.32)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_bv_vs_things_matched_category_correlation_2x2(
    clip_cmp: pd.DataFrame, dino_cmp: pd.DataFrame, out_dir: Path, semantic_df: pd.DataFrame
) -> None:
    clip = clip_cmp.copy()
    dino = dino_cmp.copy()
    clip["embedding"] = "clip"
    dino["embedding"] = "dinov3"
    combo = pd.concat([clip, dino], ignore_index=True)
    combo["category"] = combo["category"].astype(str).str.strip().str.lower()
    combo = combo.merge(semantic_df, on="category", how="left")
    combo["cdi_semantic"] = combo["cdi_semantic"].fillna("other")
    combo["dot_color"] = combo["cdi_semantic"].map(lambda s: CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]))
    # Fixed marker size: avoid implying comparability of exemplar counts across BV and THINGS.
    combo["dot_size"] = 85.0

    def _annotate_outliers(ax: plt.Axes, df: pd.DataFrame, xcol: str, ycol: str, n_labels: int = 5) -> None:
        dist = np.abs(df[ycol] - df[xcol])
        lab = df.assign(_residual=dist).sort_values("_residual", ascending=False).head(n_labels)
        offsets = [(18, 10), (-20, 10), (18, -12), (-22, -12), (20, 0)]
        for i, r in enumerate(lab.itertuples()):
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(
                str(r.category),
                (getattr(r, xcol), getattr(r, ycol)),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.75, alpha=0.9),
            )

    def _panel(ax: plt.Axes, df: pd.DataFrame, xcol: str, ycol: str, title: str, xlabel: str, ylabel: str) -> None:
        ax.scatter(
            df[xcol],
            df[ycol],
            c=df["dot_color"],
            s=df["dot_size"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )
        # Regression and identity lines
        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, m * xs + b, color="#333333", linewidth=1.4, alpha=0.9)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#666666", linewidth=1.0, alpha=0.8)

        pear = df[[xcol, ycol]].corr().iloc[0, 1]
        spear = df[[xcol, ycol]].corr(method="spearman").iloc[0, 1]

        ax.set_title(f"{title}\nPearson={pear:+.2f}, Spearman={spear:+.2f}", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.18, linewidth=0.6)
        _annotate_outliers(ax, df, xcol, ycol, n_labels=5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    clip_df = combo[combo["embedding"] == "clip"].copy()
    dino_df = combo[combo["embedding"] == "dinov3"].copy()

    _panel(
        axes[0, 0],
        clip_df,
        "things_global_dispersion",
        "bv_global_dispersion",
        "CLIP global dispersion (matched categories)",
        "THINGS global dispersion",
        "BabyView global dispersion",
    )
    _panel(
        axes[0, 1],
        clip_df,
        "things_mean_knn_dist",
        "bv_mean_knn_dist",
        "CLIP local kNN distance (matched categories)",
        "THINGS mean kNN distance",
        "BabyView mean kNN distance",
    )
    _panel(
        axes[1, 0],
        dino_df,
        "things_global_dispersion",
        "bv_global_dispersion",
        "DINOv3 global dispersion (matched categories)",
        "THINGS global dispersion",
        "BabyView global dispersion",
    )
    _panel(
        axes[1, 1],
        dino_df,
        "things_mean_knn_dist",
        "bv_mean_knn_dist",
        "DINOv3 local kNN distance (matched categories)",
        "THINGS mean kNN distance",
        "BabyView mean kNN distance",
    )

    semantic_present = [s for s in CDI_SEMANTIC_COLORS.keys() if s in set(combo["cdi_semantic"])]
    semantic_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CDI_SEMANTIC_COLORS[s],
            markeredgecolor="none",
            markersize=8,
            label=s,
        )
        for s in semantic_present
    ]
    if semantic_handles:
        fig.legend(
            handles=semantic_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(6, len(semantic_handles)),
            frameon=True,
            title="CDI semantic",
            prop={"size": 10},
            title_fontsize=11,
        )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.08, wspace=0.23, hspace=0.28)
    out_png = out_dir / "bv_vs_things_matched_category_correlation_2x2.png"
    out_pdf = out_dir / "bv_vs_things_matched_category_correlation_2x2.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _binomial_sign_test_pvalue(deltas: np.ndarray) -> float:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan
    k = int(np.sum(d > 0))
    # Two-sided exact sign test under p=0.5.
    cdf_k = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    sf_km1 = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    p = 2.0 * min(cdf_k, sf_km1)
    return float(min(1.0, p))


def save_bv_vs_things_stats_summary(clip_cmp: pd.DataFrame, dino_cmp: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    specs = [
        ("clip", clip_cmp),
        ("dinov3", dino_cmp),
    ]
    metric_specs = [
        ("global_dispersion", "things_global_dispersion", "bv_global_dispersion"),
        ("mean_knn_dist", "things_mean_knn_dist", "bv_mean_knn_dist"),
        ("mean_pairwise_dist", "things_mean_pairwise_dist", "bv_mean_pairwise_dist"),
    ]
    for embedding, df in specs:
        for metric, tcol, bcol in metric_specs:
            delta = (df[bcol] - df[tcol]).to_numpy(dtype=float)  # BV - THINGS
            ci_lo, ci_hi = _bootstrap_mean_ci(delta, n_boot=10000, alpha=0.05, seed=42)
            rows.append(
                {
                    "embedding": embedding,
                    "metric": metric,
                    "n_categories": int(np.sum(np.isfinite(delta))),
                    "things_mean": float(np.nanmean(df[tcol])),
                    "bv_mean": float(np.nanmean(df[bcol])),
                    "mean_delta_bv_minus_things": float(np.nanmean(delta)),
                    "median_delta_bv_minus_things": float(np.nanmedian(delta)),
                    "bootstrap95_mean_delta_lo": ci_lo,
                    "bootstrap95_mean_delta_hi": ci_hi,
                    "sign_test_pvalue_two_sided": _binomial_sign_test_pvalue(delta),
                }
            )
    out = pd.DataFrame(rows)
    out_path = out_dir / "bv_vs_things_paired_stats_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved stats summary: {out_path}")


def save_bv_vs_things_delta_forest_plot(out_dir: Path) -> None:
    stats_path = out_dir / "bv_vs_things_paired_stats_summary.csv"
    if not stats_path.exists():
        return
    df = pd.read_csv(stats_path).copy()
    if df.empty:
        return

    label_map = {
        ("clip", "global_dispersion"): "CLIP global dispersion",
        ("clip", "mean_knn_dist"): "CLIP local kNN distance",
        ("clip", "mean_pairwise_dist"): "CLIP mean pairwise distance",
        ("dinov3", "global_dispersion"): "DINOv3 global dispersion",
        ("dinov3", "mean_knn_dist"): "DINOv3 local kNN distance",
        ("dinov3", "mean_pairwise_dist"): "DINOv3 mean pairwise distance",
    }
    order = [
        ("clip", "global_dispersion"),
        ("clip", "mean_knn_dist"),
        ("clip", "mean_pairwise_dist"),
        ("dinov3", "global_dispersion"),
        ("dinov3", "mean_knn_dist"),
        ("dinov3", "mean_pairwise_dist"),
    ]
    df["label"] = df.apply(lambda r: label_map.get((r["embedding"], r["metric"]), f'{r["embedding"]} {r["metric"]}'), axis=1)
    df["order"] = df.apply(lambda r: order.index((r["embedding"], r["metric"])), axis=1)
    df = df.sort_values("order").reset_index(drop=True)

    y = np.arange(len(df))
    mean = df["mean_delta_bv_minus_things"].to_numpy(dtype=float)
    lo = df["bootstrap95_mean_delta_lo"].to_numpy(dtype=float)
    hi = df["bootstrap95_mean_delta_hi"].to_numpy(dtype=float)
    xerr = np.vstack([mean - lo, hi - mean])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(mean, y, xerr=xerr, fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=2, capsize=4)
    ax.axvline(0.0, linestyle="--", color="#666666", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=10)
    ax.set_xlabel("Paired delta (BV - THINGS), mean with 95% bootstrap CI", fontsize=11, fontweight="bold")
    ax.set_title("BV vs THINGS paired effects (valid85)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()

    for i, r in df.iterrows():
        ax.text(
            hi[i] + 0.02 * max(1.0, np.nanmax(np.abs(hi))),
            y[i],
            f"p={r['sign_test_pvalue_two_sided']:.1e}",
            va="center",
            fontsize=9,
            color="#333333",
        )

    out_png = out_dir / "bv_vs_things_paired_delta_forest_valid85.png"
    out_pdf = out_dir / "bv_vs_things_paired_delta_forest_valid85.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved forest plot: {out_png}")


def main() -> None:
    args = parse_args()
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["savefig.dpi"] = 300
    args.out_dir.mkdir(parents=True, exist_ok=True)
    semantic_df = _load_semantics(args.semantic_csv)

    things_clip, clip_cmp = read_metric_tables(args.input_dir, "clip", args.k, args.valid_set)
    things_dino, dino_cmp = read_metric_tables(args.input_dir, "dinov3", args.k, args.valid_set)

    save_plot_a_analog(things_clip, things_dino, args.out_dir, top_n=args.top_n, semantic_df=semantic_df)
    save_plot_b_analog(things_clip, "clip", args.out_dir, semantic_df=semantic_df)
    save_plot_b_analog(things_dino, "dinov3", args.out_dir, semantic_df=semantic_df)
    save_plot_c_analog(things_clip, "clip", args.out_dir)
    save_plot_c_analog(things_dino, "dinov3", args.out_dir)
    save_bv_vs_things_comparison(clip_cmp, "clip", args.out_dir, semantic_df=semantic_df)
    save_bv_vs_things_comparison(dino_cmp, "dinov3", args.out_dir, semantic_df=semantic_df)
    save_cross_model_bv_things_panel(clip_cmp, dino_cmp, args.out_dir, semantic_df=semantic_df)
    save_bv_vs_things_matched_category_correlation_2x2(clip_cmp, dino_cmp, args.out_dir, semantic_df=semantic_df)
    save_things_variability_correlation_2x2(things_clip, things_dino, args.out_dir, semantic_df=semantic_df)
    save_bv_vs_things_stats_summary(clip_cmp, dino_cmp, args.out_dir)
    save_bv_vs_things_delta_forest_plot(args.out_dir)

    print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
