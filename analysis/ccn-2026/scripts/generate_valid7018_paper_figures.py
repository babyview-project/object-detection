#!/usr/bin/env python3
"""Generate valid7018 paper figures (1A montages, 1B t-SNE, 1C scatter).

All category picks exclude body-part labels (see ``BODY_PART_CATEGORIES``).
Uses the 7,018 human-validated cohort metrics and crop paths.

Run from repo root::

  conda activate vislearnlabpy
  python analysis/ccn-2026/scripts/compute_valid7018_local_global.py --from-zip
  python analysis/ccn-2026/scripts/generate_valid7018_paper_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

CCN_DIR = Path(__file__).resolve().parent.parent
CCN_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = CCN_DIR.parent.parent
MANUSCRIPT_SCRIPTS = REPO_ROOT / "analysis" / "manuscript-2026" / "scripts"
# Metrics (CSV/JSON, in git) live in valid7018/; scratch PNG/PDF regen → valid7018/figures/;
# committed abstract panels → abstract_figures/ (via publish_abstract_figures).
SHARED_ROOT = REPO_ROOT / "data" / "shared_data_ccn_2026"
SHARED_INPUTS = SHARED_ROOT / "inputs"
DEFAULT_METRICS_DIR = CCN_DIR / "valid7018"
DEFAULT_SCRATCH_DIR = DEFAULT_METRICS_DIR / "figures"
DEFAULT_ABSTRACT_DIR = CCN_DIR / "abstract_figures"
DEFAULT_EMBEDDING_ZIP = SHARED_ROOT / "embeddings" / "valid7018_bv_embeddings.zip"
DEFAULT_MONTAGE_ZIP = SHARED_ROOT / "montages" / "valid7018_montage_crops.zip"

for p in (CCN_SCRIPTS, MANUSCRIPT_SCRIPTS, str(CCN_DIR)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exemplar_set_zscore_embeddings import (  # noqa: E402
    CATEGORY_FILES,
    PER_CLASS_PRECISION_CSV,
    PER_FILE_PRECISION_CSV,
    SAMPLED_EXEMPLAR_CSV,
    build_valid85_sampled_exemplar_table,
    load_config,
)

BODY_PART_CATEGORIES = frozenset(
    {
        "ankle",
        "arm",
        "ear",
        "eye",
        "face",
        "finger",
        "foot",
        "hair",
        "hand",
        "leg",
        "mouth",
        "neck",
        "nose",
        "toe",
        "tooth",
    }
)

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

CDI_SEMANTIC_ORDER = [
    "animals",
    "body_parts",
    "clothing",
    "food_drink",
    "furniture_rooms",
    "household",
    "outside",
    "people",
    "toys",
    "vehicles",
    "other",
]

_ANNOTATE_OFFSETS_SELECTED = [(10, 8), (10, -12), (-8, 10), (-10, -10), (14, 2), (-12, -14)]
_ANNOTATE_OFFSETS_OUTLIER = [
    (12, -14), (-14, 12), (10, 14), (-12, -10), (8, 12),
    (16, 4), (-16, -4), (4, 16), (-4, -16), (18, -8), (-18, 8),
    (6, -18), (-6, 18), (14, 10), (-14, -10), (20, 0), (-20, 0),
]


def parse_confidence(stem: str) -> float:
    parts = str(stem).split("_")
    if len(parts) < 2:
        return float("nan")
    try:
        return float(parts[1])
    except ValueError:
        return float("nan")


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def load_semantic_map() -> dict[str, str]:
    sem_csv = _first_existing(
        SHARED_INPUTS / "category_cdi_semantic_map.csv",
        REPO_ROOT / "data" / "long_tailed_dist_prop_included_categories.csv",
        SHARED_INPUTS / "long_tailed_dist_prop_included_categories_valid85.csv",
    )
    if sem_csv is None:
        return {}
    df = pd.read_csv(sem_csv, usecols=["category", "cdi_semantic"]).dropna()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["cdi_semantic"] = df["cdi_semantic"].astype(str).str.strip().str.lower()
    return dict(zip(df["category"], df["cdi_semantic"]))


def attach_semantic_style(df: pd.DataFrame, semantic_map: dict[str, str], n_col: str) -> pd.DataFrame:
    out = df.copy()
    out["cdi_semantic"] = out["category"].map(lambda c: semantic_map.get(str(c).strip().lower(), "other")).fillna("other")
    out["dot_color"] = out["cdi_semantic"].map(lambda s: CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]))
    nmin, nmax = out[n_col].min(), out[n_col].max()
    if np.isclose(nmin, nmax):
        out["dot_size"] = 70.0
    else:
        out["dot_size"] = 40.0 + (out[n_col] - nmin) / (nmax - nmin) * 220.0
    return out


def _style_corr_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.margins(x=0.16, y=0.16)


def _add_spearman_box(ax, rho: float, p: float) -> None:
    ax.text(
        0.02,
        0.98,
        f"Spearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB", alpha=0.92),
    )


def annotate_selected_categories(
    ax,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    selected_categories: list[str],
    *,
    n_outliers: int = 3,
    exclude_body_parts_from_outliers: bool = True,
) -> None:
    cat_lower = df["category"].astype(str).str.strip().str.lower()
    selected = {str(c).strip().lower() for c in selected_categories}
    sub_sel = df.loc[cat_lower.isin(selected)]
    for j, row in enumerate(sub_sel.itertuples(index=False)):
        ox, oy = _ANNOTATE_OFFSETS_SELECTED[j % len(_ANNOTATE_OFFSETS_SELECTED)]
        ax.annotate(
            row.category,
            (getattr(row, xcol), getattr(row, ycol)),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#141414",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#555555", alpha=0.93),
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, alpha=0.9),
            clip_on=True,
            zorder=12,
        )

    pool = df.loc[~cat_lower.isin(selected)].copy()
    if exclude_body_parts_from_outliers:
        pool = pool.loc[~pool["category"].isin(BODY_PART_CATEGORIES)]
    if len(pool) <= n_outliers:
        return
    xv = pool[xcol].to_numpy(dtype=float)
    yv = pool[ycol].to_numpy(dtype=float)
    coef = np.polyfit(xv, yv, 1)
    pred = coef[0] * xv + coef[1]
    pool["_abs_resid"] = np.abs(yv - pred)
    outliers = pool.nlargest(n_outliers, "_abs_resid")
    for j, row in enumerate(outliers.itertuples(index=False)):
        ox, oy = _ANNOTATE_OFFSETS_OUTLIER[j % len(_ANNOTATE_OFFSETS_OUTLIER)]
        ax.annotate(
            row.category,
            (getattr(row, xcol), getattr(row, ycol)),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.75, alpha=0.9),
            clip_on=True,
            zorder=11,
        )


def add_cdi_legends(
    fig,
    df: pd.DataFrame,
    n_col: str,
    *,
    size_legend_title: str = "Valid exemplars",
) -> None:
    from matplotlib.lines import Line2D

    semantic_present = [s for s in CDI_SEMANTIC_ORDER if s in set(df["cdi_semantic"])]
    semantic_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CDI_SEMANTIC_COLORS[s],
            markeredgecolor="none",
            markersize=8,
            label=s.replace("_", " "),
        )
        for s in semantic_present
    ]
    fig.legend(
        handles=semantic_handles,
        title="CDI semantic",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, min(len(semantic_handles), 6)),
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )

    nmin, nmax = df[n_col].min(), df[n_col].max()
    qvals = np.unique(np.round(np.quantile(df[n_col], [0.2, 0.5, 0.8])).astype(int))
    size_handles = []
    for q in qvals:
        if np.isclose(nmin, nmax):
            ms = float(70.0**0.5)
        else:
            s = 40.0 + (q - nmin) / (nmax - nmin) * 220.0
            ms = float(max(3.5, s**0.5))
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#777777",
                markeredgecolor="none",
                markersize=ms,
                label=f"n={q}",
            )
        )
    fig.legend(
        handles=size_handles,
        title=size_legend_title,
        loc="upper left",
        bbox_to_anchor=(0.82, 0.55),
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )


def filter_non_body(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["category"].isin(BODY_PART_CATEGORIES)].copy()


def select_evenly_spaced(categories: list[str], n: int) -> list[str]:
    if len(categories) < n:
        raise ValueError(f"Need at least {n} categories, got {len(categories)}")
    idx = [int(round(i * (len(categories) - 1) / (n - 1))) for i in range(n)]
    return [categories[i] for i in idx]


SEMANTIC_DIVERSE_TSNE_PRIORITY = (
    "household",
    "clothing",
    "toys",
    "furniture_rooms",
    "outside",
    "animals",
    "food_drink",
    "vehicles",
    "other",
)


def select_tsne_categories_semantic_diverse(
    clip_nb: pd.DataFrame,
    semantic_map: dict[str, str],
    n: int = 5,
    exclude_semantics: frozenset[str] = frozenset({"body_parts"}),
) -> list[str]:
    """Pick one non-body-part category per CDI semantic group (median global dispersion)."""
    df = clip_nb.copy()
    df["cdi_semantic"] = df["category"].map(
        lambda c: semantic_map.get(str(c).strip().lower(), "other")
    )
    reps: dict[str, str] = {}
    for sem, g in df.groupby("cdi_semantic"):
        if sem in exclude_semantics or g.empty:
            continue
        g = g.sort_values("global_dispersion")
        reps[sem] = str(g.iloc[len(g) // 2]["category"])

    picked: list[str] = []
    for sem in SEMANTIC_DIVERSE_TSNE_PRIORITY:
        if sem in reps and len(picked) < n:
            picked.append(reps[sem])
    for sem in sorted(reps):
        if reps[sem] not in picked and len(picked) < n:
            picked.append(reps[sem])
    if len(picked) < n:
        raise ValueError(f"Need {n} semantic-diverse categories, found {len(picked)}")
    return picked[:n]


def build_exemplar_crop_index(exemplar_df: pd.DataFrame) -> dict[str, list[tuple[Path, float]]]:
    df = exemplar_df.copy()
    df["confidence"] = df["stem"].map(parse_confidence)
    df = df.dropna(subset=["confidence"])
    out: dict[str, list[tuple[Path, float]]] = {}
    for cat, g in df.groupby("category"):
        rows = sorted(
            [(Path(p), float(c)) for p, c in zip(g["path"], g["confidence"])],
            key=lambda x: -x[1],
        )
        out[str(cat)] = rows
    return out


def make_montage(images: list[Image.Image], n_cols: int, cell_size: tuple[int, int]) -> Image.Image:
    n_rows = (len(images) + n_cols - 1) // n_cols
    out = Image.new("RGB", (n_cols * cell_size[0], n_rows * cell_size[1]), (255, 255, 255))
    for idx, img in enumerate(images):
        row, col = idx // n_cols, idx % n_cols
        if img.size != cell_size:
            img = img.resize(cell_size, Image.Resampling.LANCZOS)
        out.paste(img, (col * cell_size[0], row * cell_size[1]))
    return out


def load_crop_images(crop_index: dict[str, list[tuple[Path, float]]], cat: str, n: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    for crop_path, _ in crop_index.get(cat, [])[:n]:
        if not crop_path.is_file():
            continue
        try:
            images.append(Image.open(crop_path).convert("RGB"))
        except OSError:
            continue
    return images


def write_paper_stats(
    clip: pd.DataFrame,
    dino: pd.DataFrame,
    merged: pd.DataFrame,
    semantic_map: dict[str, str],
    out_path: Path,
    cdi_summary_path: Path | None = None,
) -> dict:
    clip_nb = filter_non_body(clip)
    dino_nb = filter_non_body(dino)

    def summ(s: pd.Series) -> dict:
        s = s.dropna()
        return {
            "mean": round(float(s.mean()), 2),
            "sd": round(float(s.std(ddof=1)), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "n": int(len(s)),
        }

    stats = {
        "n_exemplars": int(clip.n_exemplars.sum()),
        "n_categories": len(clip),
        "n_per_category": {"min": int(clip.n_exemplars.min()), "max": int(clip.n_exemplars.max())},
        "exclude_body_parts_from_figures": sorted(BODY_PART_CATEGORIES),
        "clip_global_all": summ(clip.global_dispersion),
        "dino_global_all": summ(dino.global_dispersion),
        "clip_local_all": summ(clip.mean_knn_dist),
        "dino_local_all": summ(dino.mean_knn_dist),
        "clip_global_non_body": summ(clip_nb.global_dispersion),
        "dino_global_non_body": summ(dino_nb.global_dispersion),
        "clip_local_non_body": summ(clip_nb.mean_knn_dist),
        "dino_local_non_body": summ(dino_nb.mean_knn_dist),
        "montage_categories_low_to_high_global": select_evenly_spaced(
            clip_nb.sort_values("global_dispersion")["category"].tolist(), 5
        ),
    }
    stats["tsne_categories"] = list(stats["montage_categories_low_to_high_global"])
    stats["tsne_categories_semantic_diverse"] = select_tsne_categories_semantic_diverse(
        clip_nb, semantic_map, n=5
    )

    for model, df, col in [("clip", clip_nb, "mean_knn_dist"), ("dino", dino_nb, "mean_knn_dist")]:
        stats[f"{model}_lowest_local_non_body"] = (
            df.nsmallest(5, col)[["category", col, "global_dispersion"]].to_dict("records")
        )
        stats[f"{model}_highest_local_non_body"] = (
            df.nlargest(5, col)[["category", col, "global_dispersion"]].to_dict("records")
        )

    corr_rows = []
    for label, gcol, lcol in [
        ("clip_within", "clip_global_dispersion", "clip_mean_knn_dist"),
        ("dinov3_within", "dinov3_global_dispersion", "dinov3_mean_knn_dist"),
        ("cross_global", "clip_global_dispersion", "dinov3_global_dispersion"),
        ("cross_local_knn", "clip_mean_knn_dist", "dinov3_mean_knn_dist"),
    ]:
        sub = merged.dropna(subset=[gcol, lcol])
        rho, p = spearmanr(sub[gcol], sub[lcol])
        corr_rows.append({"comparison": label, "spearman_rho": float(rho), "p_value": float(p)})
    stats["correlations"] = corr_rows

    plot_df = build_plot_df(merged, semantic_map)
    cdi_summary = compute_cdi_semantic_summary(plot_df)
    stats["cdi_semantic_summary"] = cdi_summary.to_dict("records")
    high = cdi_summary.iloc[0]
    low = cdi_summary.iloc[-1]
    stats["cdi_semantic_text"] = {
        "highest_global_group": high["cdi_semantic"],
        "highest_global_clip_z": round(float(high["clip_global_z_mean"]), 2),
        "lowest_global_group": low["cdi_semantic"],
        "lowest_global_clip_z": round(float(low["clip_global_z_mean"]), 2),
    }
    if cdi_summary_path is not None:
        cdi_summary.to_csv(cdi_summary_path, index=False)

    freq_check = {}
    for config_key in FREQUENCY_DISPERSION_CONFIGS:
        freq_check[config_key] = {}
        for model in ("clip", "dinov3"):
            try:
                freq_check[config_key][model] = compute_frequency_dispersion_stats(config_key, model, semantic_map)
            except FileNotFoundError as exc:
                freq_check[config_key][model] = {"error": str(exc)}
    stats["frequency_vs_global_dispersion"] = freq_check
    stats["frequency_definition_sensitivity"] = compute_frequency_definition_sensitivity(semantic_map)
    stats["frequency_plot_notes"] = {
        "abstract_frequency_source": (
            "Abstract frequency panels use full infant-view detections (valid129_filtered, 0.27 CLIP filter). "
            "Dispersion is always from the 7,018 rater-validated crop sample (uniform per-category sampling by design)."
        ),
        "dot_size_encodes": "n rater-validated crops used to estimate dispersion (not detection frequency)",
        "renormalization_note": (
            "Right abstract panel renormalizes 0.27-filtered detection proportions over the 85 plotted categories. "
            "Spearman rho is unchanged vs the 129-category denominator."
        ),
        "annotation_pool_note": (
            "valid85_detections uses the smaller VQA/annotation detection pool (~10k instances) — exploratory only."
        ),
        "raw_counts_note": (
            "Spearman rho is identical for proportions vs raw counts within each detection pool "
            "(monotonic transform). Differences across panels reflect different detection pools, not count vs proportion."
        ),
    }
    stats["abstract_frequency_config_keys"] = list(ABSTRACT_FREQUENCY_CONFIG_KEYS)
    stats["cdi_group_n_categories"] = cdi_summary.set_index("cdi_semantic")["n_categories"].astype(int).to_dict()

    out_path.write_text(json.dumps(stats, indent=2))
    return stats


def build_plot_df(merged: pd.DataFrame, semantic_map: dict[str, str]) -> pd.DataFrame:
    df = merged.copy()
    df["cdi_semantic"] = df["category"].map(lambda c: semantic_map.get(str(c).strip().lower(), "other")).fillna("other")
    df["dot_color"] = df["cdi_semantic"].map(lambda s: CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]))
    nmin, nmax = df["clip_n_exemplars"].min(), df["clip_n_exemplars"].max()
    if np.isclose(nmin, nmax):
        df["dot_size"] = 70.0
    else:
        df["dot_size"] = 40.0 + (df["clip_n_exemplars"] - nmin) / (nmax - nmin) * 220.0
    for col in ("clip_global_dispersion", "dinov3_global_dispersion", "clip_mean_knn_dist", "dinov3_mean_knn_dist"):
        mu, sd = df[col].mean(), df[col].std(ddof=1)
        df[f"{col}_z"] = (df[col] - mu) / sd if sd > 1e-12 else 0.0
    return df


def compute_cdi_semantic_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sem_name, g in df.groupby("cdi_semantic"):
        rows.append(
            {
                "cdi_semantic": sem_name,
                "n_categories": int(len(g)),
                "clip_global_mean": float(g["clip_global_dispersion"].mean()),
                "clip_global_sd": float(g["clip_global_dispersion"].std(ddof=1)),
                "dino_global_mean": float(g["dinov3_global_dispersion"].mean()),
                "dino_global_sd": float(g["dinov3_global_dispersion"].std(ddof=1)),
                "clip_local_mean": float(g["clip_mean_knn_dist"].mean()),
                "dino_local_mean": float(g["dinov3_mean_knn_dist"].mean()),
                "clip_global_z_mean": float(g["clip_global_dispersion_z"].mean()),
                "dino_global_z_mean": float(g["dinov3_global_dispersion_z"].mean()),
                "clip_local_z_mean": float(g["clip_mean_knn_dist_z"].mean()),
                "dino_local_z_mean": float(g["dinov3_mean_knn_dist_z"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("clip_global_z_mean", ascending=False)


def annotate_cross_model_disagreement(
    ax,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    selected_categories: list[str],
    *,
    n_disagree: int = 3,
) -> None:
    annotate_selected_categories(ax, df, xcol, ycol, selected_categories, n_outliers=0, exclude_body_parts_from_outliers=False)
    work = df.copy()
    xv = work[xcol].to_numpy(dtype=float)
    yv = work[ycol].to_numpy(dtype=float)
    coef = np.polyfit(xv, yv, 1)
    pred = coef[0] * xv + coef[1]
    work["_abs_resid"] = np.abs(yv - pred)
    selected = {str(c).strip().lower() for c in selected_categories}
    pool = work.loc[~work["category"].astype(str).str.lower().isin(selected)]
    pool = pool.loc[~pool["category"].isin(BODY_PART_CATEGORIES)]
    for j, row in enumerate(pool.nlargest(n_disagree, "_abs_resid").itertuples(index=False)):
        ox, oy = _ANNOTATE_OFFSETS_OUTLIER[j % len(_ANNOTATE_OFFSETS_OUTLIER)]
        ax.annotate(
            row.category,
            (getattr(row, xcol), getattr(row, ycol)),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#999999", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.75, alpha=0.9),
            clip_on=True,
            zorder=11,
        )


def figure_1c_cross_model(df: pd.DataFrame, stats: dict, out_dir: Path) -> None:
    label_categories = stats.get("montage_categories_low_to_high_global", [])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.subplots_adjust(left=0.07, right=0.80, top=0.88, bottom=0.20, wspace=0.22)

    panels = [
        (axes[0], "clip_global_dispersion", "dinov3_global_dispersion", "Cross-model global dispersion", "Global dispersion (CLIP)", "Global dispersion (DINOv3)"),
        (axes[1], "clip_mean_knn_dist", "dinov3_mean_knn_dist", "Cross-model local dispersion", "Local dispersion (CLIP)", "Local dispersion (DINOv3)"),
    ]
    for ax, xcol, ycol, title, xlabel, ylabel in panels:
        ax.scatter(
            df[xcol],
            df[ycol],
            c=df["dot_color"],
            s=df["dot_size"],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )
        rho, p = spearmanr(df[xcol], df[ycol])
        _add_spearman_box(ax, rho, p)
        _style_corr_axes(ax, title, xlabel, ylabel)
        annotate_cross_model_disagreement(ax, df, xcol, ycol, label_categories, n_disagree=3)

    add_cdi_legends(fig, df, "clip_n_exemplars")
    fig.suptitle("Category-level agreement across CLIP and DINOv3 (valid7018)", fontsize=12, fontweight="bold", y=0.98)
    for ext in ("png", "pdf"):
        path = out_dir / f"fig1C_valid7018_cross_model_k5.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def figure_1b_cdi_semantic_groups(df: pd.DataFrame, cdi_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_groups = [s for s in CDI_SEMANTIC_ORDER if s in set(cdi_summary["cdi_semantic"])]
    plot_groups = [s for s in plot_groups if s != "people"]  # no categories in valid85
    x = np.arange(len(plot_groups))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.22, wspace=0.22)

    for ax, clip_z, dino_z, ylabel, title in [
        (axes[0], "clip_global_dispersion_z", "dinov3_global_dispersion_z", "Global dispersion (model z-score)", "Variability by CDI semantic group: global"),
        (axes[1], "clip_mean_knn_dist_z", "dinov3_mean_knn_dist_z", "Local dispersion (model z-score)", "Variability by CDI semantic group: local"),
    ]:
        clip_data = [df.loc[df["cdi_semantic"] == s, clip_z].to_numpy() for s in plot_groups]
        dino_data = [df.loc[df["cdi_semantic"] == s, dino_z].to_numpy() for s in plot_groups]
        bp1 = ax.boxplot(
            clip_data,
            positions=x - width / 2,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#111111", linewidth=1.5),
        )
        bp2 = ax.boxplot(
            dino_data,
            positions=x + width / 2,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#111111", linewidth=1.5),
        )
        for patch in bp1["boxes"]:
            patch.set(facecolor="#5B9BD5", alpha=0.55, edgecolor="#2E6DA4")
        for patch in bp2["boxes"]:
            patch.set(facecolor="#E8A54C", alpha=0.55, edgecolor="#C87A1A")
        for i, sem in enumerate(plot_groups):
            sub = df.loc[df["cdi_semantic"] == sem]
            ax.scatter(
                np.full(len(sub), x[i] - width / 2),
                sub[clip_z],
                s=16,
                c=sub["dot_color"],
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
            ax.scatter(
                np.full(len(sub), x[i] + width / 2),
                sub[dino_z],
                s=16,
                c=sub["dot_color"],
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s.replace('_', chr(10))}\n(n={int(cdi_summary.loc[cdi_summary.cdi_semantic==s, 'n_categories'].iloc[0])})" for s in plot_groups], fontsize=8, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")

    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], color="#5B9BD5", lw=8, label="CLIP"),
            Line2D([0], [0], color="#E8A54C", lw=8, label="DINOv3"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=True,
        fontsize=10,
        title="Model",
        title_fontsize=10,
    )
    fig.suptitle("Within-group category variability (z-scored within each model)", fontsize=12, fontweight="bold", y=0.98)
    for ext in ("png", "pdf"):
        path = out_dir / f"fig1B_valid7018_cdi_semantic_groups.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def _frequency_table_path(name: str, *candidates: Path) -> Path:
    found = _first_existing(*candidates)
    if found is None:
        raise FileNotFoundError(f"Frequency table not found for cohort={name}")
    return found


def _resolve_frequency_table_paths() -> dict[str, Path]:
    return {
        "valid85": _frequency_table_path(
            "valid85",
            SHARED_INPUTS / "long_tailed_dist_prop_included_categories_valid85.csv",
            REPO_ROOT / "data" / "long_tailed_dist_prop_included_categories_valid85.csv",
        ),
        "valid129_filtered": _frequency_table_path(
            "valid129_filtered",
            SHARED_INPUTS / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
            REPO_ROOT
            / "data"
            / "shared_data_manuscript_2026"
            / "inputs"
            / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv",
        ),
        "valid129_counts": _frequency_table_path(
            "valid129_counts",
            SHARED_INPUTS / "long_tailed_dist_prop_included_categories_valid129.csv",
            REPO_ROOT
            / "data"
            / "shared_data_manuscript_2026"
            / "inputs"
            / "long_tailed_dist_prop_included_categories_valid129.csv",
        ),
    }

DISPERSION_TABLE_PATHS = {
    ("valid85", "clip"): DEFAULT_METRICS_DIR / "bv_valid7018_clip_local_global_k5.csv",
    ("valid85", "dinov3"): DEFAULT_METRICS_DIR / "bv_valid7018_dinov3_local_global_k5.csv",
}

# Cross frequency/dispersion pairings on the same 85 categories (dispersion from valid7018).
# Abstract panels use full_dataset_* (infant-view detection pool). valid85_detections is the
# smaller annotation/VQA pool (~10k instances) and is exploratory only — not for the abstract.
FREQUENCY_DISPERSION_CONFIGS = {
    "full_dataset_frequency": {
        "frequency_cohort": "valid129_filtered",
        "dispersion_cohort": "valid85",
        "frequency_mode": "proportion",
        "renormalize_proportion": False,
        "panel_title": "Infant-view frequency (full dataset, 0.27-filtered)",
        "xlabel": "Detection proportion (0.27-filtered pool; 129-category denominator)",
        "frequency_note": (
            "Numerator/denominator: full BabyView infant-view detections after 0.27 CLIP filter "
            "(valid129 pool). Not the 7,018 rater-validated crop sample."
        ),
        "dispersion_note": "Y and dot size: valid7018 per-crop dispersion (7,018 crops). Dot size is not frequency.",
        "n_resid": 10,
        "n_quad": 3,
        "n_extreme": 2,
        "exclude_body_parts": True,
    },
    "full_dataset_frequency_renorm85": {
        "frequency_cohort": "valid129_filtered",
        "dispersion_cohort": "valid85",
        "frequency_mode": "proportion",
        "renormalize_proportion": True,
        "panel_title": "Same counts, renormalized over these 85 categories",
        "xlabel": "Detection proportion (renormalized over these 85 categories)",
        "frequency_note": (
            "Same 0.27-filtered detection counts as left panel; denominator restricted to "
            "these 85 categories (44 valid129-only categories excluded)."
        ),
        "dispersion_note": "Same valid7018 per-crop dispersion and dot size as left column.",
        "n_resid": 10,
        "n_quad": 3,
        "n_extreme": 2,
        "exclude_body_parts": True,
    },
    "valid85_detections": {
        "frequency_cohort": "valid85",
        "dispersion_cohort": "valid85",
        "frequency_mode": "proportion",
        "renormalize_proportion": False,
        "panel_title": "Annotation-pool frequency (exploratory; not full dataset)",
        "xlabel": "Detection proportion (VQA/annotation pool, valid85 set)",
        "frequency_note": (
            "Exploratory only: counts from the smaller annotation/VQA detection pool "
            "(~10k instances), not full infant-view detections."
        ),
        "dispersion_note": "Y and dot size: valid7018 per-crop dispersion (7,018 crops). Dot size is not frequency.",
        "n_resid": 10,
        "n_quad": 3,
        "n_extreme": 2,
        "exclude_body_parts": True,
    },
    "valid85_manuscript_frequency": {
        "frequency_cohort": "valid129_filtered",
        "dispersion_cohort": "valid85",
        "frequency_mode": "proportion",
        "renormalize_proportion": True,
        "panel_title": "Manuscript frequency (0.27-filtered, renorm. over 85)",
        "xlabel": "Detection proportion (manuscript counts, renormalized over these 85 categories)",
        "frequency_note": (
            "Alias of full_dataset_frequency_renorm85 for backward-compatible figure stems."
        ),
        "dispersion_note": "Same valid7018 per-crop dispersion and dot size as full-dataset panels.",
        "n_resid": 10,
        "n_quad": 3,
        "n_extreme": 2,
        "exclude_body_parts": True,
    },
}

ABSTRACT_FREQUENCY_CONFIG_KEYS = (
    "full_dataset_frequency",
    "full_dataset_frequency_renorm85",
)

DOT_SIZE_LEGEND_TITLE = "n crops (dispersion estimate)"
DOT_SIZE_LEGEND_NOTE = "Dot size encodes sample size for the variability estimate, not detection frequency."


def load_frequency_table(cohort: str = "valid85") -> pd.DataFrame:
    freq_csv = _resolve_frequency_table_paths().get(cohort)
    if freq_csv is None or not freq_csv.is_file():
        raise FileNotFoundError(f"Frequency table not found for cohort={cohort}: {freq_csv}")
    df = pd.read_csv(freq_csv)
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")
    if "count_instances" in df.columns:
        df["count_instances"] = pd.to_numeric(df["count_instances"], errors="coerce")
    return df


def _frequency_x_column(cfg: dict) -> str:
    return "count_instances" if cfg.get("frequency_mode") == "count_instances" else "proportion"


def build_frequency_dispersion_df(config_key: str, model: str, semantic_map: dict[str, str]) -> pd.DataFrame:
    cfg = FREQUENCY_DISPERSION_CONFIGS[config_key]
    freq = load_frequency_table(cfg["frequency_cohort"])
    disp = load_dispersion_table(cfg["dispersion_cohort"], model)
    gcol = f"{model}_global_dispersion"
    ncol = f"{model}_n_exemplars"
    df = freq.merge(disp[["category", gcol, ncol]], on="category", how="inner")
    xcol = _frequency_x_column(cfg)
    if cfg.get("renormalize_proportion") and xcol == "proportion":
        total = df["proportion"].sum()
        if total > 0:
            df = df.copy()
            df["proportion"] = df["proportion"] / total
    if xcol == "count_instances" and "count_instances" not in df.columns:
        raise ValueError(f"Frequency table for {cfg['frequency_cohort']} lacks count_instances")
    df = attach_semantic_style(df, semantic_map, ncol)
    return df


def compute_frequency_dispersion_stats(config_key: str, model: str, semantic_map: dict[str, str]) -> dict:
    cfg = FREQUENCY_DISPERSION_CONFIGS[config_key]
    df = build_frequency_dispersion_df(config_key, model, semantic_map)
    gcol = f"{model}_global_dispersion"
    xcol = _frequency_x_column(cfg)
    sub = df.dropna(subset=[xcol, gcol])
    rho, p = spearmanr(sub[xcol], sub[gcol])
    return {
        "config_key": config_key,
        "frequency_cohort": cfg["frequency_cohort"],
        "dispersion_cohort": cfg["dispersion_cohort"],
        "frequency_mode": cfg.get("frequency_mode", "proportion"),
        "renormalize_proportion": bool(cfg.get("renormalize_proportion", False)),
        "model": model,
        "frequency_column": xcol,
        "dispersion_column": gcol,
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_categories": int(len(sub)),
    }


def compute_frequency_definition_sensitivity(semantic_map: dict[str, str]) -> list[dict]:
    """Spearman rho for alternate x-axis definitions (same valid85 dispersion)."""
    v85 = load_frequency_table("valid85")
    ms_prop = load_frequency_table("valid129_filtered")
    ms_counts = load_frequency_table("valid129_counts")
    rows = []
    for model in ("clip", "dinov3"):
        disp = load_dispersion_table("valid85", model)
        gcol = f"{model}_global_dispersion"
        base = v85[["category", "proportion", "count_instances"]].merge(
            disp[["category", gcol]],
            on="category",
            how="inner",
        )
        ms = base.merge(ms_prop[["category", "proportion"]].rename(columns={"proportion": "ms_prop129"}), on="category")
        ms = ms.merge(
            ms_counts[["category", "count_instances"]].rename(columns={"count_instances": "ms_count129"}),
            on="category",
        )
        ms["ms_prop85"] = ms["ms_prop129"] / ms["ms_prop129"].sum()
        variants = [
            ("annotation_pool_proportion", "proportion"),
            ("annotation_pool_count_instances", "count_instances"),
            ("full_dataset_prop129_denom", "ms_prop129"),
            ("full_dataset_prop_renorm85", "ms_prop85"),
            ("full_dataset_raw_count129", "ms_count129"),
        ]
        for label, xcol in variants:
            sub = ms.dropna(subset=[xcol, gcol])
            rho, p = spearmanr(sub[xcol], sub[gcol])
            rows.append(
                {
                    "model": model,
                    "frequency_definition": label,
                    "frequency_column": xcol,
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n_categories": int(len(sub)),
                }
            )
        rho_rank, _ = spearmanr(ms["proportion"], ms["ms_prop129"])
        rows.append(
            {
                "model": model,
                "frequency_definition": "rank_agreement_annotation_pool_vs_full_dataset_prop129",
                "frequency_column": "proportion vs ms_prop129",
                "spearman_rho": float(rho_rank),
                "p_value": None,
                "n_categories": int(len(ms)),
            }
        )
    return rows


def load_dispersion_table(cohort: str, model: str) -> pd.DataFrame:
    path = DISPERSION_TABLE_PATHS.get((cohort, model))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Dispersion table not found for cohort={cohort}, model={model}: {path}")
    df = pd.read_csv(path)
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    return df.rename(columns={"global_dispersion": f"{model}_global_dispersion", "n_exemplars": f"{model}_n_exemplars"})


def _plot_frequency_dispersion_ax(
    ax,
    df: pd.DataFrame,
    model: str,
    config_key: str,
    label_categories: list[str],
) -> dict:
    cfg = FREQUENCY_DISPERSION_CONFIGS[config_key]
    gcol = f"{model}_global_dispersion"
    xcol = _frequency_x_column(cfg)
    sub = df.dropna(subset=[xcol, gcol])
    rho, p = spearmanr(sub[xcol], sub[gcol])
    ax.scatter(
        sub[xcol],
        sub[gcol],
        c=sub["dot_color"],
        s=sub["dot_size"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    _add_spearman_box(ax, rho, p)
    _style_corr_axes(
        ax,
        f"{model.upper()} global dispersion",
        cfg["xlabel"],
        f"Global dispersion ({model.upper()})",
    )
    annotate_frequency_dispersion_decoupling(
        ax,
        sub,
        xcol,
        gcol,
        label_categories,
        n_resid=cfg["n_resid"],
        n_quad=cfg["n_quad"],
        n_extreme=cfg.get("n_extreme", 0),
        exclude_body_parts=cfg["exclude_body_parts"],
    )
    footnote = cfg.get("frequency_note", "")
    if cfg.get("dispersion_note"):
        footnote = f"{footnote}\n{cfg['dispersion_note']}" if footnote else cfg["dispersion_note"]
    ax.text(
        0.02,
        0.02,
        footnote,
        transform=ax.transAxes,
        fontsize=6.8,
        va="bottom",
        ha="left",
        color="#444444",
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88),
    )
    return {
        "config_key": config_key,
        "frequency_cohort": cfg["frequency_cohort"],
        "dispersion_cohort": cfg["dispersion_cohort"],
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_categories": int(len(sub)),
    }


def figure_frequency_vs_global_dispersion(
    semantic_map: dict[str, str],
    stats: dict,
    out_dir: Path,
    *,
    config_key: str = "valid85_detections",
    model: str = "dinov3",
) -> dict:
    cfg = FREQUENCY_DISPERSION_CONFIGS[config_key]
    df = build_frequency_dispersion_df(config_key, model, semantic_map)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    fig.subplots_adjust(left=0.11, right=0.78, top=0.86, bottom=0.22)
    result = _plot_frequency_dispersion_ax(
        ax,
        df,
        model,
        config_key,
        stats.get("montage_categories_low_to_high_global", []),
    )
    add_cdi_legends(fig, df, f"{model}_n_exemplars", size_legend_title=DOT_SIZE_LEGEND_TITLE)
    fig.suptitle(
        f"Visual frequency vs global dispersion ({cfg['panel_title']})",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.11, 0.02, DOT_SIZE_LEGEND_NOTE, fontsize=8, color="#555555")
    stem = f"fig_explore_frequency_vs_{model}_global_{config_key}"
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)
    return {"model": model, **result}


def figure_frequency_vs_global_robustness_2x2(semantic_map: dict[str, str], stats: dict, out_dir: Path) -> list[dict]:
    label_categories = stats.get("montage_categories_low_to_high_global", [])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.8))
    fig.subplots_adjust(left=0.07, right=0.82, top=0.88, bottom=0.14, wspace=0.22, hspace=0.32)
    results = []
    config_keys = list(ABSTRACT_FREQUENCY_CONFIG_KEYS)
    for row, model in enumerate(["clip", "dinov3"]):
        for col, config_key in enumerate(config_keys):
            ax = axes[row, col]
            cfg = FREQUENCY_DISPERSION_CONFIGS[config_key]
            df = build_frequency_dispersion_df(config_key, model, semantic_map)
            result = _plot_frequency_dispersion_ax(ax, df, model, config_key, label_categories)
            results.append({"model": model, **result})
            if row == 0:
                ax.set_title(cfg["panel_title"], fontsize=10, fontweight="bold", pad=6)
    add_cdi_legends(
        fig,
        build_frequency_dispersion_df("full_dataset_frequency", "clip", semantic_map),
        "clip_n_exemplars",
        size_legend_title=DOT_SIZE_LEGEND_TITLE,
    )
    fig.suptitle(
        "Frequency vs global dispersion: full infant-view detections (valid7018 variability)",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.07,
        0.01,
        DOT_SIZE_LEGEND_NOTE,
        fontsize=8.5,
        color="#555555",
    )
    for ext in ("png", "pdf"):
        path = out_dir / f"fig_explore_frequency_vs_global_robustness_2x2.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)
    return results



def _frequency_dispersion_label_candidates(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    selected_categories: list[str],
    *,
    n_resid: int = 10,
    n_quad: int = 3,
    n_extreme: int = 2,
    exclude_body_parts: bool = True,
) -> list[str]:
    """Pick montage categories plus outliers that illustrate freq–disp decoupling."""
    selected = [str(c).strip().lower() for c in selected_categories]
    out: list[str] = []
    seen: set[str] = set()

    def add(cats: list[str]) -> None:
        for cat in cats:
            k = str(cat).strip().lower()
            if k and k not in seen:
                out.append(k)
                seen.add(k)

    add(selected)

    pool = df.loc[~df["category"].astype(str).str.lower().isin(seen)].copy()
    if exclude_body_parts:
        pool = pool.loc[~pool["category"].isin(BODY_PART_CATEGORIES)]

    if len(pool) >= 3 and n_resid > 0:
        xv = pool[xcol].to_numpy(dtype=float)
        yv = pool[ycol].to_numpy(dtype=float)
        coef = np.polyfit(xv, yv, 1)
        pool = pool.assign(_abs_resid=np.abs(yv - (coef[0] * xv + coef[1])))
        add(pool.nlargest(n_resid, "_abs_resid")["category"].tolist())

    if n_extreme > 0 and len(pool) >= 3:
        add(pool.nlargest(n_extreme, ycol)["category"].tolist())
        add(pool.nsmallest(n_extreme, ycol)["category"].tolist())
        add(pool.nlargest(n_extreme, xcol)["category"].tolist())

    if n_quad > 0 and len(pool) >= 4:
        x_hi = pool[xcol].quantile(0.65)
        x_lo = pool[xcol].quantile(0.35)
        high_freq = pool.loc[pool[xcol] >= x_hi]
        low_freq = pool.loc[pool[xcol] <= x_lo]
        add(high_freq.nlargest(n_quad, ycol)["category"].tolist())  # frequent + variable
        add(high_freq.nsmallest(n_quad, ycol)["category"].tolist())  # frequent + stable
        add(low_freq.nlargest(n_quad, ycol)["category"].tolist())  # rare + variable
        add(low_freq.nsmallest(n_quad, ycol)["category"].tolist())  # rare + stable

    return out


def annotate_frequency_dispersion_decoupling(
    ax,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    selected_categories: list[str],
    *,
    n_resid: int = 10,
    n_quad: int = 3,
    n_extreme: int = 2,
    exclude_body_parts: bool = True,
) -> list[str]:
    label_cats = _frequency_dispersion_label_candidates(
        df,
        xcol,
        ycol,
        selected_categories,
        n_resid=n_resid,
        n_quad=n_quad,
        n_extreme=n_extreme,
        exclude_body_parts=exclude_body_parts,
    )
    outlier_font = 7 if len(label_cats) > 20 else 8
    for j, cat in enumerate(label_cats):
        row = df.loc[df["category"].astype(str).str.lower() == cat]
        if row.empty:
            continue
        row = row.iloc[0]
        ox, oy = _ANNOTATE_OFFSETS_SELECTED[j % len(_ANNOTATE_OFFSETS_SELECTED)]
        if j >= len(selected_categories):
            ox, oy = _ANNOTATE_OFFSETS_OUTLIER[j % len(_ANNOTATE_OFFSETS_OUTLIER)]
        fontsize = 9 if j < len(selected_categories) else outlier_font
        ax.annotate(
            row.category,
            (float(row[xcol]), float(row[ycol])),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold",
            color="#141414" if j < len(selected_categories) else "#222222",
            bbox=dict(
                boxstyle="round,pad=0.25" if j < len(selected_categories) else "round,pad=0.2",
                facecolor="white",
                edgecolor="#555555" if j < len(selected_categories) else "#999999",
                alpha=0.93 if j < len(selected_categories) else 0.9,
            ),
            arrowprops=dict(arrowstyle="-", color="#666666" if j < len(selected_categories) else "#888888", lw=0.8),
            clip_on=True,
            zorder=12 if j < len(selected_categories) else 11,
        )
    return label_cats



ABSTRACT_FIGURE_FILES = (
    "fig1A_valid7018_montages_low_to_high_global",
    "fig1B_valid7018_tsne_dinov3",
    "fig1B_valid7018_tsne_dinov3_semantic_diverse",
    "fig1C_valid7018_cross_model_k5",
    "fig_explore_frequency_vs_global_robustness_2x2",
)


def publish_abstract_figures(out_dir: Path, abstract_dir: Path) -> list[Path]:
    """Copy selected montage + frequency panels for CCN abstract submission."""
    import shutil

    abstract_dir.mkdir(parents=True, exist_ok=True)
    selection_src = out_dir / "valid7018_figure_category_selection.csv"
    if selection_src.is_file():
        shutil.copy2(selection_src, abstract_dir / selection_src.name)

    written: list[Path] = []
    for stem in ABSTRACT_FIGURE_FILES:
        for ext in ("png", "pdf"):
            src = out_dir / f"{stem}.{ext}"
            if not src.is_file():
                if ext == "png":
                    print(f"  Skip missing abstract asset: {src.name}")
                continue
            dst = abstract_dir / src.name
            shutil.copy2(src, dst)
            written.append(dst)
    try:
        dest_label = abstract_dir.relative_to(REPO_ROOT)
    except ValueError:
        dest_label = abstract_dir
    print(f"Published {len(written)} abstract figure files to {dest_label}")
    return written


def figure_1a_montages(
    clip: pd.DataFrame,
    crop_index: dict[str, list[tuple[Path, float]]] | None,
    semantic_map: dict[str, str],
    montage_cats: list[str],
    out_dir: Path,
    montage_zip_crops: dict[str, list[Image.Image]] | None = None,
    n_exemplars: int = 25,
    n_cols: int = 5,
    cell_size: tuple[int, int] = (128, 128),
) -> None:
    clip_by_cat = clip.set_index("category")
    montage_images: list[tuple[str, Image.Image, str, float]] = []

    for cat in montage_cats:
        if montage_zip_crops and cat in montage_zip_crops:
            imgs = montage_zip_crops[cat][:n_exemplars]
        elif crop_index is not None:
            imgs = load_crop_images(crop_index, cat, n_exemplars)
        else:
            imgs = []
        if len(imgs) < n_cols:
            print(f"  Skip montage {cat}: only {len(imgs)} readable crops")
            continue
        montage = make_montage(imgs[: n_cols * n_cols], n_cols, cell_size)
        sem = semantic_map.get(cat, "other")
        g = float(clip_by_cat.loc[cat, "global_dispersion"])
        montage_images.append((cat, montage, sem, g))
        montage.save(out_dir / f"fig1A_montage_{cat}_global={g:.3f}.png")
        montage.save(out_dir / f"fig1A_montage_{cat}_global={g:.3f}.pdf", "PDF", resolution=200)

    if not montage_images:
        print("No montages written (crop paths missing?)")
        return

    n = len(montage_images)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.8), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (cat, montage, sem, g) in zip(axes, montage_images):
        ax.imshow(montage)
        ax.axis("off")
        color = CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS["other"])
        ax.set_title(f"{cat}\n(global={g:.2f})", fontsize=9, color=color, fontweight="bold")
    fig.suptitle("Low → high global dispersion (valid7018; non-body-part categories)", fontsize=11)
    for ext in ("png", "pdf"):
        path = out_dir / f"fig1A_valid7018_montages_low_to_high_global.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def figure_1b_tsne(
    zip_path: Path,
    tsne_cats: list[str],
    semantic_map: dict[str, str],
    out_dir: Path,
    output_stem: str = "fig1B_valid7018_tsne_dinov3",
    title: str = "DINOv3 t-SNE (valid7018; selected categories)",
    max_per_cat: int = 80,
    seed: int = 42,
) -> None:
    import matplotlib.colors as mcolors
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from load_valid7018_embeddings import load_valid7018_from_zip

    _, dino_by_cat = load_valid7018_from_zip(zip_path)
    rng = np.random.default_rng(seed)
    specs: list[dict] = []
    for cat in tsne_cats:
        X = dino_by_cat.get(cat)
        if X is None or len(X) < 2:
            continue
        n = min(max_per_cat, len(X))
        idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
        Xs = np.asarray(X[idx], dtype=np.float64)
        centroid = Xs.mean(axis=0)
        dist = np.linalg.norm(Xs - centroid, axis=1)
        specs.append({"name": cat, "X": Xs, "dist": dist})

    if len(specs) < 2:
        print("Skip t-SNE: not enough categories/embeddings")
        return

    X_all = np.vstack([s["X"] for s in specs])
    mean, std = X_all.mean(axis=0), X_all.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    X_all = (X_all - mean) / std

    perplexity = min(30.0, max(5.0, (len(X_all) - 1) // 3))
    xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
    ).fit_transform(X_all)

    row = 0
    coords_by_name: dict[str, np.ndarray] = {}
    for spec in specs:
        n_i = spec["X"].shape[0]
        coords_by_name[spec["name"]] = xy[row : row + n_i]
        row += n_i

    all_d = np.concatenate([np.asarray(s["dist"], dtype=np.float64) for s in specs])
    p_lo, p_hi = np.percentile(all_d, [6.0, 94.0])
    if p_hi - p_lo < 1e-12:
        p_lo, p_hi = float(all_d.min()), float(all_d.max()) + 1e-6
    sat_gamma, sat_floor = 0.42, 0.12
    gray = np.array([0.90, 0.90, 0.91], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.subplots_adjust(right=0.84, top=0.90, bottom=0.14)
    cbar_drawn = False
    label_offsets = [(12, 10), (-14, 10), (-12, -12), (12, -12)]

    for i, spec in enumerate(specs):
        cat = spec["name"]
        ex = coords_by_name[cat]
        d = np.asarray(spec["dist"], dtype=np.float64)
        t = np.clip((d - p_lo) / (p_hi - p_lo + 1e-12), 0, 1)
        wmix = sat_floor + (1.0 - sat_floor) * (t**sat_gamma)
        sem = semantic_map.get(cat, "other")
        base = np.array(mcolors.to_rgb(CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS["other"])))
        rgb = (1.0 - wmix)[:, np.newaxis] * gray + wmix[:, np.newaxis] * base
        rgb = np.clip(rgb, 0, 1)
        ax.scatter(
            ex[:, 0],
            ex[:, 1],
            c=rgb,
            s=52,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        if not cbar_drawn:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4.8%", pad=0.12)
            sm = cm.ScalarMappable(norm=Normalize(vmin=p_lo, vmax=p_hi), cmap=plt.colormaps["gray"])
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Distance to category centroid", fontsize=10, fontweight="bold")
            cbar.ax.tick_params(labelsize=9)
            cbar_drawn = True

        cx, cy = ex.mean(axis=0)
        ax.scatter([cx], [cy], s=180, marker="o", c="#1958d1", edgecolors="white", linewidths=1.2, zorder=6)
        ax.scatter([cx], [cy], s=260, marker="o", facecolors="none", edgecolors="#1958d1", linewidths=1.2, alpha=0.45, zorder=5)
        ox, oy = label_offsets[i % len(label_offsets)]
        ax.annotate(
            cat,
            (cx, cy),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="#141414",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#555555", alpha=0.93),
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8),
            zorder=12,
        )

    sem_present = []
    for spec in specs:
        sem = semantic_map.get(spec["name"], "other")
        if sem not in sem_present:
            sem_present.append(sem)
    sem_present = [s for s in CDI_SEMANTIC_ORDER if s in sem_present] + [s for s in sem_present if s not in CDI_SEMANTIC_ORDER]
    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]),
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=s.replace("_", " "),
        )
        for s in sem_present
    ]
    ax.legend(handles=legend_elems, title="CDI semantic", loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=len(legend_elems), frameon=True, fontsize=9, title_fontsize=10)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("t-SNE 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for ext in ("png", "pdf"):
        path = out_dir / f"{output_stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def _load_crop_sources(from_zip: bool) -> tuple[dict[str, list[tuple[Path, float]]] | None, dict[str, list[Image.Image]] | None]:
    montage_zip_crops: dict[str, list[Image.Image]] | None = None
    crop_index: dict[str, list[tuple[Path, float]]] | None = None

    if from_zip:
        from load_valid7018_montage_crops import load_montage_crops_from_zip

        montage_zip_crops = load_montage_crops_from_zip(DEFAULT_MONTAGE_ZIP)
        print(f"Loaded montage crops from zip ({len(montage_zip_crops)} categories)")
        return crop_index, montage_zip_crops

    if (
        PER_FILE_PRECISION_CSV.is_file()
        and SAMPLED_EXEMPLAR_CSV.is_file()
        and CATEGORY_FILES["valid85"].is_file()
    ):
        cfg = load_config()
        exemplar_df = build_valid85_sampled_exemplar_table(
            CATEGORY_FILES["valid85"],
            PER_CLASS_PRECISION_CSV,
            PER_FILE_PRECISION_CSV,
            SAMPLED_EXEMPLAR_CSV,
            cfg["precision_threshold"],
        )
        crop_index = build_exemplar_crop_index(exemplar_df)
        return crop_index, montage_zip_crops

    from load_valid7018_montage_crops import load_montage_crops_from_zip

    montage_zip_crops = load_montage_crops_from_zip(DEFAULT_MONTAGE_ZIP)
    print("Annotation tables missing; using montage crop zip")
    return crop_index, montage_zip_crops


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-zip",
        action="store_true",
        help="Load montage JPEGs from data/shared_data_ccn_2026/montages/ (clone-safe)",
    )
    p.add_argument(
        "--abstract-dir",
        type=Path,
        default=DEFAULT_ABSTRACT_DIR,
        help="Publish selected montage + frequency figures here (default: abstract_figures/)",
    )
    args = p.parse_args()

    metrics_dir = DEFAULT_METRICS_DIR
    scratch_dir = DEFAULT_SCRATCH_DIR
    scratch_dir.mkdir(parents=True, exist_ok=True)

    clip = pd.read_csv(metrics_dir / "bv_valid7018_clip_local_global_k5.csv")
    dino = pd.read_csv(metrics_dir / "bv_valid7018_dinov3_local_global_k5.csv")
    merged = pd.read_csv(metrics_dir / "bv_valid7018_clip_vs_dinov3_local_global_k5.csv")

    crop_index, montage_zip_crops = _load_crop_sources(args.from_zip)
    semantic_map = load_semantic_map()

    stats = write_paper_stats(
        clip,
        dino,
        merged,
        semantic_map,
        metrics_dir / "valid7018_paper_stats.json",
        metrics_dir / "valid7018_cdi_semantic_summary.csv",
    )
    plot_df = build_plot_df(merged, semantic_map)
    montage_cats = stats["montage_categories_low_to_high_global"]
    semantic_tsne_cats = stats["tsne_categories_semantic_diverse"]
    print(f"Montage + dispersion t-SNE categories: {montage_cats}")
    print(f"Semantic-diverse t-SNE categories: {semantic_tsne_cats}")

    figure_1a_montages(
        clip,
        crop_index,
        semantic_map,
        montage_cats,
        scratch_dir,
        montage_zip_crops=montage_zip_crops,
    )
    figure_1b_tsne(
        DEFAULT_EMBEDDING_ZIP,
        montage_cats,
        semantic_map,
        scratch_dir,
        output_stem="fig1B_valid7018_tsne_dinov3",
        title="DINOv3 t-SNE (valid7018; low→high global dispersion)",
    )
    figure_1b_tsne(
        DEFAULT_EMBEDDING_ZIP,
        semantic_tsne_cats,
        semantic_map,
        scratch_dir,
        output_stem="fig1B_valid7018_tsne_dinov3_semantic_diverse",
        title="DINOv3 t-SNE (valid7018; one category per CDI semantic group)",
    )
    figure_1c_cross_model(plot_df, stats, scratch_dir)
    for model in ("clip", "dinov3"):
        for config_key in FREQUENCY_DISPERSION_CONFIGS:
            if config_key == "valid85_manuscript_frequency":
                continue  # alias of full_dataset_frequency_renorm85
            res = figure_frequency_vs_global_dispersion(
                semantic_map,
                stats,
                scratch_dir,
                config_key=config_key,
                model=model,
            )
            print(
                f"Frequency vs {model.upper()} global ({config_key}): "
                f"rho={res['spearman_rho']:.3f}, p={res['p_value']:.3e}, n={res['n_categories']}"
            )
    robustness = figure_frequency_vs_global_robustness_2x2(semantic_map, stats, scratch_dir)
    for res in robustness:
        print(
            f"Comparison {res['config_key']} {res['model'].upper()}: "
            f"rho={res['spearman_rho']:.3f}, p={res['p_value']:.3e}, n={res['n_categories']}"
        )

    selection_csv = scratch_dir / "valid7018_figure_category_selection.csv"
    rows = []
    for i, cat in enumerate(montage_cats, start=1):
        row = clip[clip.category == cat].iloc[0]
        rows.append(
            {
                "figure": "1A",
                "panel": i,
                "category": cat,
                "cdi_semantic": semantic_map.get(cat, "other"),
                "global_dispersion_clip": row.global_dispersion,
                "mean_knn_clip": row.mean_knn_dist,
            }
        )
    for i, cat in enumerate(montage_cats, start=1):
        row = clip[clip.category == cat].iloc[0]
        rows.append(
            {
                "figure": "1B_dispersion",
                "panel": i,
                "category": cat,
                "cdi_semantic": semantic_map.get(cat, "other"),
                "global_dispersion_clip": row.global_dispersion,
                "mean_knn_clip": row.mean_knn_dist,
            }
        )
    for i, cat in enumerate(semantic_tsne_cats, start=1):
        row = clip[clip.category == cat].iloc[0]
        rows.append(
            {
                "figure": "1B_semantic_diverse",
                "panel": i,
                "category": cat,
                "cdi_semantic": semantic_map.get(cat, "other"),
                "global_dispersion_clip": row.global_dispersion,
                "mean_knn_clip": row.mean_knn_dist,
            }
        )
    pd.DataFrame(rows).to_csv(selection_csv, index=False)
    print(f"Wrote {selection_csv}")
    publish_abstract_figures(scratch_dir, args.abstract_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
