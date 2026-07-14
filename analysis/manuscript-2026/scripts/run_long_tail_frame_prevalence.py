#!/usr/bin/env python3
"""Regenerate long-tail frame-prevalence tables, power-law fits, and figures (notebook 01)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PREPRINT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PREPRINT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from frame_prevalence import (  # noqa: E402
    build_frame_prevalence_table,
    load_detections,
    sync_public_frequency_tables,
)

DATA_DIR = REPO_ROOT / "data"
FRAME_DATA_CSV = REPO_ROOT / "frame_data" / "merged_frame_detections_with_metadata_filtered-0.27.csv"
CDI_CSV = DATA_DIR / "cdi_words.csv"
THRESHOLD_TOKEN = "0.27"

CATEGORY_FILES = {
    "valid85": DATA_DIR / "included_categories_valid85.txt",
    "valid129": DATA_DIR / "included_categories_valid129.txt",
}
RUN_ROOTS = {
    "valid129": PREPRINT_DIR / "main_results_valid129s_04302026",
    "valid85": PREPRINT_DIR / "supplemental_results_valid85cats_04302026",
}

FRAME_PREVALENCE_YLABEL = "Frame prevalence"

CDI_SEMANTIC_ORDER = [
    "animals", "body_parts", "clothing", "food_drink", "furniture_rooms",
    "household", "outside", "people", "toys", "vehicles", "other",
]
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


def _apply_axis_style(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", labelsize=13, width=1.2)


def _semantic_present(df_plot: pd.DataFrame) -> list[str]:
    semantic_present = [s for s in CDI_SEMANTIC_ORDER if s in set(df_plot["cdi_semantic"])]
    if len(semantic_present) == 0:
        semantic_present = sorted(df_plot["cdi_semantic"].dropna().unique().tolist())
    semantic_rank = (
        df_plot[df_plot["cdi_semantic"].isin(semantic_present)]
        .groupby("cdi_semantic", as_index=False)["proportion"]
        .sum()
        .sort_values("proportion", ascending=False)
    )
    semantic_present = semantic_rank["cdi_semantic"].tolist()
    if "furniture_rooms" in semantic_present and "household" in semantic_present:
        semantic_present.remove("furniture_rooms")
        household_idx = semantic_present.index("household")
        semantic_present.insert(household_idx, "furniture_rooms")
    return semantic_present


def _save_distribution_figures(df_plot: pd.DataFrame, figures_dir: Path, file_suffix: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    semantic_present = _semantic_present(df_plot)

    n_sem = len(semantic_present)
    if n_sem <= 4:
        n_cols = max(1, n_sem)
    elif n_sem <= 6:
        n_cols = 3
    elif n_sem <= 8:
        n_cols = 4
    else:
        n_cols = 5
    n_rows_sub = int(np.ceil(n_sem / n_cols)) if n_sem > 0 else 1

    fig = plt.figure(figsize=(18, 4 + 3.2 * n_rows_sub), constrained_layout=True)
    gs = GridSpec(1 + n_rows_sub, n_cols, figure=fig, height_ratios=[1.4] + [1] * n_rows_sub)

    ax_top = fig.add_subplot(gs[0, :])
    top50 = df_plot.head(50).copy()
    colors_50 = [CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]) for s in top50["cdi_semantic"]]
    x50 = np.arange(len(top50))
    ax_top.bar(x50, top50["proportion"], color=colors_50, edgecolor="none", width=0.8)
    ax_top.set_xticks(x50)
    ax_top.set_xticklabels(top50["category"], rotation=45, ha="right", fontsize=10)
    ax_top.set_ylabel(FRAME_PREVALENCE_YLABEL, fontsize=14)
    ax_top.set_title("Top 50 categories overall (colored by CDI semantic category)")
    _apply_axis_style(ax_top)

    for idx, sem in enumerate(semantic_present):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        sub = (
            df_plot[df_plot["cdi_semantic"] == sem]
            .sort_values("proportion", ascending=False)
            .head(10)
        )
        x = np.arange(len(sub))
        color = CDI_SEMANTIC_COLORS.get(sem, CDI_SEMANTIC_COLORS["other"])
        ax.bar(x, sub["proportion"], color=color, edgecolor="none", width=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["category"], rotation=45, ha="right", fontsize=10)
        ax.set_xlim(-0.5, 9.5)
        if idx == 0:
            ax.set_ylabel(FRAME_PREVALENCE_YLABEL, fontsize=14)
        else:
            ax.set_ylabel("")
        ax.set_title(sem.replace("_", " "), color=color, fontsize=12, fontweight="bold")
        _apply_axis_style(ax)

    total_slots = n_rows_sub * n_cols
    for j in range(n_sem, total_slots):
        ax_unused = fig.add_subplot(gs[1 + j // n_cols, j % n_cols])
        ax_unused.axis("off")

    out_png = figures_dir / f"long_tailed_top50_plus_semantic_subplots_{file_suffix}.png"
    out_pdf = figures_dir / f"long_tailed_top50_plus_semantic_subplots_{file_suffix}.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")

    fig_all, ax_all = plt.subplots(figsize=(26, 8), constrained_layout=True)
    colors_all = [CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]) for s in df_plot["cdi_semantic"]]
    x_all = np.arange(len(df_plot))
    ax_all.bar(x_all, df_plot["proportion"], color=colors_all, edgecolor="none", width=0.8)
    ax_all.set_xticks(x_all)
    ax_all.set_xticklabels(df_plot["category"], rotation=70, ha="right", fontsize=10)
    ax_all.set_ylabel(FRAME_PREVALENCE_YLABEL, fontsize=14)
    ax_all.set_title("All included categories (colored by CDI semantic category)")
    _apply_axis_style(ax_all)
    legend_handles_all = [
        Patch(facecolor=CDI_SEMANTIC_COLORS[k], label=k.replace("_", " "))
        for k in semantic_present
    ]
    ax_all.legend(handles=legend_handles_all, ncol=min(6, len(legend_handles_all)), frameon=False, fontsize=12, loc="upper right")

    out_all_png = figures_dir / f"long_tailed_all_included_categories_{file_suffix}.png"
    out_all_pdf = figures_dir / f"long_tailed_all_included_categories_{file_suffix}.pdf"
    fig_all.savefig(out_all_png, dpi=150, bbox_inches="tight")
    fig_all.savefig(out_all_pdf, bbox_inches="tight")
    plt.close(fig_all)
    print(f"  wrote {out_all_png}")


def _plot_empirical_and_fit(ax, values: np.ndarray, title: str, color: str = "#4C78A8") -> None:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    vals = np.sort(vals)[::-1]
    n = vals.size
    if n == 0:
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        return

    ranks = np.arange(1, n + 1, dtype=np.float64)
    fit = fit_power_law_rank(vals)
    ax.plot(ranks, vals, "o", ms=4, alpha=0.8, color=color, label="empirical")
    if np.isfinite(fit["alpha"]) and np.isfinite(fit["coef_c"]):
        fit_curve = fit["coef_c"] * (ranks ** (-fit["alpha"]))
        ax.plot(ranks, fit_curve, "-", lw=2, color="black", label="power-law fit")
        ax.set_title(f"{title}\nalpha={fit['alpha']:.3f}, R2(log)={fit['r2_log']:.3f}, n={fit['n_points']}")
    else:
        ax.set_title(f"{title} (insufficient points, n={fit['n_points']})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel(FRAME_PREVALENCE_YLABEL)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=9)


def _save_powerlaw_figures(df_cat: pd.DataFrame, figures_dir: Path, file_suffix: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    overall_vals = df_cat["proportion"].astype(float).sort_values(ascending=False).to_numpy()
    semantic_order_for_fit = [s for s in CDI_SEMANTIC_ORDER if s in set(df_cat["cdi_semantic"].astype(str))]

    fig_overall, ax_overall = plt.subplots(figsize=(7, 5), constrained_layout=True)
    _plot_empirical_and_fit(ax_overall, overall_vals, "Overall included categories", color="#4C78A8")
    out_overall_png = figures_dir / f"long_tailed_powerlaw_overall_{file_suffix}.png"
    out_overall_pdf = figures_dir / f"long_tailed_powerlaw_overall_{file_suffix}.pdf"
    fig_overall.savefig(out_overall_png, dpi=150, bbox_inches="tight")
    fig_overall.savefig(out_overall_pdf, bbox_inches="tight")
    plt.close(fig_overall)
    print(f"  wrote {out_overall_png}")

    n_sem = len(semantic_order_for_fit)
    if n_sem == 0:
        return

    n_cols = 4 if n_sem >= 8 else 3
    n_rows = int(np.ceil(n_sem / n_cols))
    fig_sem, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for i, sem in enumerate(semantic_order_for_fit):
        sem_vals = (
            df_cat.loc[df_cat["cdi_semantic"] == sem, "proportion"]
            .astype(float)
            .sort_values(ascending=False)
            .to_numpy()
        )
        color = CDI_SEMANTIC_COLORS.get(sem, "#4C78A8")
        _plot_empirical_and_fit(axes[i], sem_vals, sem.replace("_", " "), color=color)
    for j in range(n_sem, len(axes)):
        axes[j].axis("off")

    out_sem_png = figures_dir / f"long_tailed_powerlaw_semantic_grid_{file_suffix}.png"
    out_sem_pdf = figures_dir / f"long_tailed_powerlaw_semantic_grid_{file_suffix}.pdf"
    fig_sem.savefig(out_sem_png, dpi=150, bbox_inches="tight")
    fig_sem.savefig(out_sem_pdf, bbox_inches="tight")
    plt.close(fig_sem)
    print(f"  wrote {out_sem_png}")


def fit_power_law_rank(values: np.ndarray, min_points: int = 3) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    vals = np.sort(vals)[::-1]
    n = vals.size
    if n < min_points:
        return {
            "n_points": int(n),
            "alpha": np.nan,
            "intercept_log": np.nan,
            "coef_c": np.nan,
            "r2_log": np.nan,
            "rmse_log": np.nan,
        }

    ranks = np.arange(1, n + 1, dtype=np.float64)
    x = np.log(ranks)
    y = np.log(vals)
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    return {
        "n_points": int(n),
        "alpha": float(-slope),
        "intercept_log": float(intercept),
        "coef_c": float(np.exp(intercept)),
        "r2_log": float(r2),
        "rmse_log": rmse,
    }


def semantic_map_from_cdi() -> dict[str, str]:
    df_cdi = pd.read_csv(CDI_CSV, usecols=["uni_lemma", "category"])
    df_cdi["uni_lemma"] = df_cdi["uni_lemma"].astype(str).str.strip().str.lower()
    df_cdi["category"] = df_cdi["category"].astype(str).str.strip().str.lower()
    return (
        df_cdi.drop_duplicates(subset=["uni_lemma"], keep="first")
        .set_index("uni_lemma")["category"]
        .to_dict()
    )


def run_category_set(category_set: str, semantic_map: dict[str, str], det: pd.DataFrame) -> None:
    included = [
        line.strip().lower()
        for line in CATEGORY_FILES[category_set].read_text().splitlines()
        if line.strip()
    ]
    run_root = RUN_ROOTS[category_set]
    results_dir = run_root / "results"
    figures_dir = run_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = f"filtered-{THRESHOLD_TOKEN}_{category_set}"

    df_cat, pool_label = build_frame_prevalence_table(det, category_set, included, semantic_map)
    df_cat = df_cat.sort_values("proportion", ascending=False).reset_index(drop=True)

    out_intermediate = results_dir / f"long_tailed_dist_prop_included_categories_{file_suffix}.csv"
    df_cat.to_csv(out_intermediate, index=False)
    sync_public_frequency_tables(df_cat, category_set, DATA_DIR, threshold_token=THRESHOLD_TOKEN)

    overall_vals = df_cat["proportion"].astype(float).sort_values(ascending=False).to_numpy()
    rows = [
        {
            "distribution": "overall_included_categories",
            "semantic_category": "all",
            **fit_power_law_rank(overall_vals),
        }
    ]
    for sem in sorted(df_cat["cdi_semantic"].dropna().unique()):
        vals = (
            df_cat.loc[df_cat["cdi_semantic"] == sem, "proportion"]
            .astype(float)
            .sort_values(ascending=False)
            .to_numpy()
        )
        rows.append(
            {
                "distribution": f"semantic_{sem}",
                "semantic_category": sem,
                **fit_power_law_rank(vals),
            }
        )

    powerlaw_df = pd.DataFrame(rows)
    out_fit_csv = results_dir / f"long_tailed_powerlaw_fits_{file_suffix}.csv"
    out_fit_txt = results_dir / f"long_tailed_powerlaw_fits_{file_suffix}.txt"
    powerlaw_df.to_csv(out_fit_csv, index=False)
    out_fit_txt.write_text(powerlaw_df.to_string(index=False), encoding="utf-8")

    chair = df_cat.loc[df_cat.category == "chair"].iloc[0]
    print(
        f"[{category_set}] pool={pool_label}; total_frames={chair.total_frames}; "
        f"chair frame_prev={chair.proportion:.4f}; alpha={rows[0]['alpha']:.4f}"
    )
    print(f"  wrote {out_intermediate}")
    print(f"  wrote {out_fit_csv}")

    _save_distribution_figures(df_cat, figures_dir, file_suffix)
    _save_powerlaw_figures(df_cat, figures_dir, file_suffix)


def _rescale_sizes(values: pd.Series, min_size: float = 30.0, max_size: float = 220.0) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    if arr.size == 0:
        return arr
    vmin = float(arr.min())
    vmax = float(arr.max())
    if np.isclose(vmax, vmin):
        return np.full(arr.shape, (min_size + max_size) / 2.0)
    return min_size + ((arr - vmin) / (vmax - vmin)) * (max_size - min_size)


def _corr_for_category_set(
    set_name: str,
    included_txt: Path,
    det: pd.DataFrame,
    semantic_map: dict[str, str],
    df_precision: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    included = [
        line.strip().lower()
        for line in included_txt.read_text().splitlines()
        if line.strip()
    ]
    df_prop, pool_label = build_frame_prevalence_table(det, set_name, included, semantic_map)
    merged = df_prop.merge(df_precision, on="category", how="inner")
    merged = merged.dropna(subset=["precision", "proportion"]).copy()

    pearson_r, pearson_p = stats.pearsonr(merged["precision"], merged["proportion"])
    spearman_rho, spearman_p = stats.spearmanr(merged["precision"], merged["proportion"])

    summary = {
        "category_set": set_name,
        "frame_pool": pool_label,
        "n_categories": int(merged.shape[0]),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }
    return merged.sort_values("proportion", ascending=False).reset_index(drop=True), summary


def _save_precision_scatter(
    corr_by_set: dict[str, pd.DataFrame],
    corr_summary_df: pd.DataFrame,
) -> None:
    valid129_results = RUN_ROOTS["valid129"] / "results"
    valid85_results = RUN_ROOTS["valid85"] / "results"
    output_dirs = {valid129_results, valid85_results}

    for set_name, merged_set in corr_by_set.items():
        set_results_dir = RUN_ROOTS[set_name] / "results"
        summary_one = corr_summary_df[corr_summary_df["category_set"] == set_name].copy()
        summary_name = f"precision_vs_frame_prevalence_correlation_filtered-{THRESHOLD_TOKEN}_{set_name}.csv"
        details_name = f"precision_vs_frame_prevalence_by_category_filtered-{THRESHOLD_TOKEN}_{set_name}.csv"

        summary_one.to_csv(set_results_dir / summary_name, index=False)
        merged_set.to_csv(set_results_dir / details_name, index=False)
        summary_one.to_csv(valid129_results / summary_name, index=False)
        merged_set.to_csv(valid129_results / details_name, index=False)
        print(f"  wrote {set_results_dir / summary_name}")
        print(f"  wrote {set_results_dir / details_name}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, set_name in zip(axes, ["valid85", "valid129"]):
        df_s = corr_by_set[set_name].copy()
        point_sizes = _rescale_sizes(df_s["count_frames"])
        point_colors = [
            CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"])
            for s in df_s["cdi_semantic"]
        ]
        ax.scatter(
            df_s["precision"],
            df_s["proportion"],
            s=point_sizes,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            c=point_colors,
        )
        x = df_s["precision"].to_numpy(dtype=float)
        y = df_s["proportion"].to_numpy(dtype=float)
        m, b = np.polyfit(x, y, deg=1)
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, m * xx + b, color="#444444", linewidth=2)

        row = corr_summary_df[corr_summary_df["category_set"] == set_name].iloc[0]
        ax.set_title(
            f"{set_name} (n={int(row['n_categories'])})\n"
            f"Pearson r={row['pearson_r']:.3f}, p={row['pearson_p']:.2g}\n"
            f"Spearman rho={row['spearman_rho']:.3f}, p={row['spearman_p']:.2g}\n"
            f"Pool: {row['frame_pool']}; color=CDI semantic; size=frames with category",
            fontsize=11,
        )
        ax.set_xlabel("Category precision")
        ax.set_ylabel(FRAME_PREVALENCE_YLABEL)
        _apply_axis_style(ax)

        sem_present = [s for s in CDI_SEMANTIC_ORDER if s in set(df_s["cdi_semantic"])]
        sem_handles = [
            Line2D(
                [0], [0], marker="o", markersize=7,
                markerfacecolor=CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]),
                markeredgecolor="none", linestyle="", label=s.replace("_", " "),
            )
            for s in sem_present
        ]
        counts_unique = np.unique(df_s["count_frames"].to_numpy(dtype=float))
        if counts_unique.size >= 3:
            count_examples = np.quantile(counts_unique, [0.1, 0.5, 0.9]).astype(int)
        else:
            count_examples = counts_unique.astype(int)
        count_examples = np.unique(count_examples)
        size_examples = _rescale_sizes(pd.Series(count_examples.astype(float)))
        size_handles = [
            Line2D(
                [0], [0], marker="o", markersize=np.sqrt(sz),
                markerfacecolor="#777777", markeredgecolor="none", linestyle="",
                label=f"{int(ct)} frames with category",
            )
            for ct, sz in zip(count_examples, size_examples)
        ]
        legend_sem = ax.legend(
            handles=sem_handles, title="CDI semantic", loc="upper left",
            bbox_to_anchor=(1.02, 1.0), fontsize=8, title_fontsize=9, frameon=True,
        )
        ax.add_artist(legend_sem)
        ax.legend(
            handles=size_handles, title="Marker size (frames with category)",
            loc="upper left", bbox_to_anchor=(1.02, 0.55), fontsize=8, title_fontsize=9, frameon=True,
        )

    plot_png_name = "precision_vs_frame_prevalence_scatter_valid85_valid129.png"
    plot_pdf_name = "precision_vs_frame_prevalence_scatter_valid85_valid129.pdf"
    for out_dir in output_dirs:
        fig.savefig(out_dir / plot_png_name, dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / plot_pdf_name, bbox_inches="tight")
        print(f"  wrote {out_dir / plot_png_name}")
    plt.close(fig)


def _save_precision_sensitivity(corr_by_set: dict[str, pd.DataFrame]) -> None:
    top_k = 10
    low_prec_q = 0.10
    low_prec_abs = 0.70
    valid129_results = RUN_ROOTS["valid129"] / "results"
    valid85_results = RUN_ROOTS["valid85"] / "results"

    sensitivity_rows = []
    ranking_tables: dict[str, dict[str, pd.DataFrame]] = {}

    for set_name in ("valid85", "valid129"):
        df_s = corr_by_set[set_name].copy()
        df_s = df_s.sort_values("count_frames", ascending=False).reset_index(drop=True)
        df_s["raw_rank"] = np.arange(1, len(df_s) + 1)

        df_s["weighted_count"] = df_s["count_frames"] * df_s["precision"]
        df_s["weighted_prop"] = df_s["weighted_count"] / df_s["weighted_count"].sum()
        df_s = df_s.sort_values("weighted_count", ascending=False).reset_index(drop=True)
        df_s["weighted_rank"] = np.arange(1, len(df_s) + 1)
        df_s = df_s.sort_values("raw_rank").reset_index(drop=True)

        q_threshold = float(df_s["precision"].quantile(low_prec_q))

        top_raw = (
            df_s.sort_values("count_frames", ascending=False)
            .head(top_k)[["category", "count_frames", "precision", "cdi_semantic"]]
            .rename(columns={"count_frames": "raw_count_frames"})
            .reset_index(drop=True)
        )
        top_weighted = (
            df_s.sort_values("weighted_count", ascending=False)
            .head(top_k)[["category", "weighted_count", "precision", "cdi_semantic"]]
            .reset_index(drop=True)
        )
        trimmed_q = df_s[df_s["precision"] >= q_threshold].copy()
        top_trimmed_q = (
            trimmed_q.sort_values("count_frames", ascending=False)
            .head(top_k)[["category", "count_frames", "precision", "cdi_semantic"]]
            .rename(columns={"count_frames": "trimmed_q_count_frames"})
            .reset_index(drop=True)
        )
        trimmed_abs = df_s[df_s["precision"] >= low_prec_abs].copy()
        top_trimmed_abs = (
            trimmed_abs.sort_values("count_frames", ascending=False)
            .head(top_k)[["category", "count_frames", "precision", "cdi_semantic"]]
            .rename(columns={"count_frames": "trimmed_abs_count_frames"})
            .reset_index(drop=True)
        )

        raw_top_set = set(top_raw["category"])
        weighted_top_set = set(top_weighted["category"])
        trimmed_q_top_set = set(top_trimmed_q["category"])
        trimmed_abs_top_set = set(top_trimmed_abs["category"])

        sensitivity_rows.append(
            {
                "category_set": set_name,
                "top_k": top_k,
                "q_threshold_precision": q_threshold,
                "abs_threshold_precision": low_prec_abs,
                "n_excluded_q": int((df_s["precision"] < q_threshold).sum()),
                "n_excluded_abs": int((df_s["precision"] < low_prec_abs).sum()),
                "top1_raw": top_raw.iloc[0]["category"] if len(top_raw) else None,
                "top1_weighted": top_weighted.iloc[0]["category"] if len(top_weighted) else None,
                "top1_trimmed_q": top_trimmed_q.iloc[0]["category"] if len(top_trimmed_q) else None,
                "top1_trimmed_abs": top_trimmed_abs.iloc[0]["category"] if len(top_trimmed_abs) else None,
                "overlap_raw_vs_weighted_topk": len(raw_top_set & weighted_top_set),
                "overlap_raw_vs_trimmed_q_topk": len(raw_top_set & trimmed_q_top_set),
                "overlap_raw_vs_trimmed_abs_topk": len(raw_top_set & trimmed_abs_top_set),
            }
        )
        ranking_tables[set_name] = {
            "full": df_s.copy(),
            "top_raw": top_raw.copy(),
            "top_weighted": top_weighted.copy(),
            "top_trimmed_q": top_trimmed_q.copy(),
            "top_trimmed_abs": top_trimmed_abs.copy(),
        }

    sensitivity_summary_df = pd.DataFrame(sensitivity_rows)
    out_names = {
        "full": "precision_sensitivity_full_table_{set_name}.csv",
        "top_raw": f"precision_sensitivity_top{top_k}_raw_{{set_name}}.csv",
        "top_weighted": f"precision_sensitivity_top{top_k}_weighted_{{set_name}}.csv",
        "top_trimmed_q": f"precision_sensitivity_top{top_k}_trimmed_q{low_prec_q:.2f}_{{set_name}}.csv",
        "top_trimmed_abs": f"precision_sensitivity_top{top_k}_trimmed_abs{low_prec_abs:.2f}_{{set_name}}.csv",
    }

    for set_name, tables in ranking_tables.items():
        set_results_dir = RUN_ROOTS[set_name] / "results"
        for key, pattern in out_names.items():
            filename = pattern.format(set_name=set_name)
            tables[key].to_csv(set_results_dir / filename, index=False)
            tables[key].to_csv(valid129_results / filename, index=False)
            print(f"  wrote {set_results_dir / filename}")

    summary_name = f"precision_sensitivity_summary_top{top_k}.csv"
    sensitivity_summary_df.to_csv(valid85_results / summary_name, index=False)
    sensitivity_summary_df.to_csv(valid129_results / summary_name, index=False)
    print(f"  wrote {valid129_results / summary_name}")

    for _, row in sensitivity_summary_df.iterrows():
        print(
            f"  [{row['category_set']}] top-1 raw={row['top1_raw']}, weighted={row['top1_weighted']}; "
            f"Top-{int(row['top_k'])} overlap raw-vs-weighted="
            f"{int(row['overlap_raw_vs_weighted_topk'])}/{int(row['top_k'])}"
        )


def main() -> int:
    semantic_map = semantic_map_from_cdi()
    det = load_detections(FRAME_DATA_CSV)
    for category_set in ("valid129", "valid85"):
        run_category_set(category_set, semantic_map, det)

    precision_csv = REPO_ROOT / "annotation" / "per_class_validation_data.csv"
    df_precision = pd.read_csv(precision_csv, usecols=["class", "precision"])
    df_precision["category"] = df_precision["class"].astype(str).str.strip().str.lower()
    df_precision["precision"] = pd.to_numeric(df_precision["precision"], errors="coerce")
    df_precision = df_precision[["category", "precision"]].dropna(subset=["precision"]).copy()

    corr_by_set = {}
    corr_summary_rows = []
    for set_name, included_txt in CATEGORY_FILES.items():
        merged_set, summary_set = _corr_for_category_set(
            set_name, included_txt, det, semantic_map, df_precision
        )
        corr_by_set[set_name] = merged_set
        corr_summary_rows.append(summary_set)

    corr_summary_df = pd.DataFrame(corr_summary_rows).sort_values("category_set").reset_index(drop=True)
    print("Precision vs frame prevalence:")
    _save_precision_scatter(corr_by_set, corr_summary_df)
    print("Precision sensitivity (frame-prevalence rankings):")
    _save_precision_sensitivity(corr_by_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
