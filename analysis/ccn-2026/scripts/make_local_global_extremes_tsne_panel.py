#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


CCN_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = CCN_DIR / "plotC_knn_diversity_outputs"
EXTREMES_CSV = OUT_DIR / "ccn2026_local_global_extreme_categories_clip_dino.csv"
TSNE_CLIP_CSV = (
    CCN_DIR.parent / "manuscript-2026" / "tsne_cdi_results_clip" / "tsne_cdi_coordinates.csv"
)
TSNE_DINO_CSV = (
    CCN_DIR.parent / "manuscript-2026" / "tsne_cdi_results_dinov3" / "tsne_cdi_coordinates.csv"
)


def get_overlap_categories(ext_df: pd.DataFrame, view: str, top_n: int = 12) -> list[str]:
    clip = (
        ext_df[(ext_df["embedding"] == "clip") & (ext_df["extreme_view"] == view)]
        .sort_values("extreme_rank")
        .head(top_n)["category"]
        .tolist()
    )
    dino = (
        ext_df[(ext_df["embedding"] == "dino") & (ext_df["extreme_view"] == view)]
        .sort_values("extreme_rank")
        .head(top_n)["category"]
        .tolist()
    )
    return [c for c in clip if c in set(dino)]


def normalize_tsne_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns={"class_name": "category"}).copy()
    needed = {"category", "tsne_x", "tsne_y"}
    missing = needed - set(renamed.columns)
    if missing:
        raise ValueError(f"Missing required t-SNE columns: {sorted(missing)}")
    renamed["category"] = renamed["category"].astype(str).str.strip()
    return renamed


def plot_panel(
    ax: plt.Axes, df: pd.DataFrame, categories: list[str], title: str, palette: list[str]
) -> None:
    ax.scatter(df["tsne_x"], df["tsne_y"], s=16, alpha=0.20, c="#B0B0B0", linewidths=0)

    sub = df[df["category"].isin(categories)].copy()
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
    sub["color"] = sub["category"].map(color_map)

    ax.scatter(
        sub["tsne_x"],
        sub["tsne_y"],
        s=70,
        c=sub["color"],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
        zorder=3,
    )

    for r in sub.itertuples():
        ax.text(r.tsne_x, r.tsne_y, f" {r.category}", fontsize=8, va="center", ha="left")

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def main() -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ext_df = pd.read_csv(EXTREMES_CSV)
    clip_tsne = normalize_tsne_columns(pd.read_csv(TSNE_CLIP_CSV))
    dino_tsne = normalize_tsne_columns(pd.read_csv(TSNE_DINO_CSV))

    high_overlap = get_overlap_categories(ext_df, "high_local_global_ratio", top_n=12)
    low_overlap = get_overlap_categories(ext_df, "low_local_global_ratio", top_n=12)

    overlap_df = pd.DataFrame(
        {
            "extreme_group": (["high_local_global_ratio"] * len(high_overlap))
            + (["low_local_global_ratio"] * len(low_overlap)),
            "category": high_overlap + low_overlap,
        }
    )
    overlap_csv = OUT_DIR / "ccn2026_local_global_overlap_categories_used_for_tsne_2x2.csv"
    overlap_df.to_csv(overlap_csv, index=False)

    high_palette = ["#d73027", "#fc8d59", "#e34a33", "#f46d43", "#fdae61", "#ef3b2c"]
    low_palette = ["#4575b4", "#74add1", "#2b8cbe", "#3182bd", "#6baed6", "#08519c"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    plot_panel(
        axes[0, 0],
        clip_tsne,
        high_overlap,
        f"CLIP t-SNE: high local/global overlap (n={len(high_overlap)})",
        high_palette,
    )
    plot_panel(
        axes[0, 1],
        clip_tsne,
        low_overlap,
        f"CLIP t-SNE: low local/global overlap (n={len(low_overlap)})",
        low_palette,
    )
    plot_panel(
        axes[1, 0],
        dino_tsne,
        high_overlap,
        f"DINOv3 t-SNE: high local/global overlap (n={len(high_overlap)})",
        high_palette,
    )
    plot_panel(
        axes[1, 1],
        dino_tsne,
        low_overlap,
        f"DINOv3 t-SNE: low local/global overlap (n={len(low_overlap)})",
        low_palette,
    )

    fig.suptitle(
        "Local-vs-global dispersion extremes projected on category t-SNE (CLIP vs DINOv3)",
        fontsize=14,
        y=1.01,
    )

    out_png = OUT_DIR / "ccn2026_local_global_extremes_tsne_overlap_2x2.png"
    out_pdf = OUT_DIR / "ccn2026_local_global_extremes_tsne_overlap_2x2.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {overlap_csv}")
    print(f"high_overlap={high_overlap}")
    print(f"low_overlap={low_overlap}")


if __name__ == "__main__":
    main()
