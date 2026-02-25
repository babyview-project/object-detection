"""
Long-tailed object detection distribution: skewness, power-law fits, and plots for the 163 shared categories.

- Y-axis: proportion detected (not raw count).
- Fits power-law to (rank, proportion) and reports exponent; reports empirical skewness.
- Plots all 163 and top 50 with Konkle colors; separate plots per group (small_obj, big_obj, animated, others).
- Fits power-law to each group's distribution.
- Saves all plots as .png and .pdf.

Uses preprocessed_object_detections.csv and the 163-category list; Konkle categories from CDI metadata.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

# Default paths (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_PREPROCESSED_CSV = PROJECT_ROOT / "frame_data" / "preprocessed_object_detections.csv"
DEFAULT_163_CATS_FILE = PROJECT_ROOT / "data" / "things_bv_overlap_categories_exclude_zero_precisions.txt"
DEFAULT_CDI_WORDS = PROJECT_ROOT / "data" / "cdi_words.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "long_tailed_163cats"


# Konkle colors (same as ccn_analyses.qmd)
KONKLE_COLORS = {
    "animated": "purple",
    "small_obj": "orange",
    "big_obj": "blue",
    "others": "grey",
}


def load_163_categories(path: Path) -> list[str]:
    """Load category names from file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def load_konkle_mapping(cdi_path: Path) -> dict[str, str]:
    """
    Build uni_lemma -> konkle_category from CDI.
    konkle_category: animated (is_animate==1), small_obj (is_small==1), big_obj (is_big==1), others.
    """
    df = pd.read_csv(cdi_path)
    # One row per uni_lemma: take first (or aggregate) for is_animate, is_small, is_big
    agg = (
        df.groupby("uni_lemma", as_index=False)
        .agg({"is_animate": "max", "is_small": "max", "is_big": "max"})
    )
    out = {}
    for _, row in agg.iterrows():
        u = str(row["uni_lemma"]).strip().lower()
        if row["is_animate"] == 1:
            out[u] = "animated"
        elif row["is_small"] == 1:
            out[u] = "small_obj"
        elif row["is_big"] == 1:
            out[u] = "big_obj"
        else:
            out[u] = "others"
    return out


def fit_power_law(proportions: np.ndarray, eps: float = 1e-10) -> tuple[float, float, np.ndarray]:
    """
    Fit power-law proportion_k ~ rank_k^(-alpha) via log-log linear regression.
    proportions: array of proportions, already sorted descending (rank 1 = first element).
    Returns (alpha, intercept, fitted_proportions).
    """
    p = np.asarray(proportions, dtype=float)
    p = np.maximum(p, eps)
    rank = np.arange(1, len(p) + 1, dtype=float)
    log_rank = np.log(rank)
    log_p = np.log(p)
    # log(p) = c - alpha * log(rank)  =>  slope = -alpha
    slope, intercept, *_ = np.polyfit(log_rank, log_p, 1)
    alpha = -slope
    fitted_log_p = intercept + slope * log_rank
    fitted_p = np.exp(fitted_log_p)
    return alpha, intercept, fitted_p


def save_fig_both(fig: plt.Figure, base_path: Path, dpi: int = 150) -> None:
    """Save figure to both .png and .pdf (base_path has no extension)."""
    for ext in (".png", ".pdf"):
        fig.savefig(base_path.with_suffix(ext), dpi=dpi if ext == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def get_category_counts(preprocessed_csv: Path, allowed_categories: set[str]) -> pd.DataFrame:
    """
    Aggregate preprocessed_object_detections by class_name; keep only allowed categories.
    Returns DataFrame with columns: Category, Count, Total_frames, Proportion.
    """
    df = pd.read_csv(preprocessed_csv)
    df["class_name"] = df["class_name"].str.strip().str.lower()
    df = df[df["class_name"].isin(allowed_categories)]
    agg = (
        df.groupby("class_name", as_index=False)
        .agg(Count=("num_detected", "sum"), Total_frames=("num_frames", "sum"))
    )
    agg["Proportion"] = agg["Count"] / agg["Total_frames"].replace(0, np.nan)
    agg = agg.rename(columns={"class_name": "Category"})
    agg = agg.sort_values("Count", ascending=False).reset_index(drop=True)
    return agg


def plot_bar_with_konkle(
    df: pd.DataFrame,
    konkle_map: dict[str, str],
    title: str,
    out_base: Path,
    top_n: int | None = None,
    power_law_alpha: float | None = None,
    skewness: float | None = None,
) -> None:
    """
    Bar plot of proportion detected per category, colored by Konkle category.
    out_base: path without extension (saved as .png and .pdf).
    """
    plot_df = df.head(top_n).copy() if top_n is not None else df.copy()
    plot_df["konkle_category"] = plot_df["Category"].str.lower().map(
        lambda c: konkle_map.get(c, "others")
    )
    colors = [KONKLE_COLORS[k] for k in plot_df["konkle_category"]]

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["Proportion"], color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Category"], rotation=45, ha="right")
    ax.set_ylabel("Proportion detected")
    ax.set_xlabel("Category")
    if power_law_alpha is not None or skewness is not None:
        parts = [title]
        if power_law_alpha is not None:
            parts.append(f"power-law α={power_law_alpha:.3f}")
        if skewness is not None:
            parts.append(f"skewness={skewness:.3f}")
        ax.set_title(" | ".join(parts))
    else:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    used = plot_df["konkle_category"].unique()
    legend_elements = [
        Patch(facecolor=KONKLE_COLORS[k], label=k.replace("_", " ")) for k in ["animated", "small_obj", "big_obj", "others"] if k in used
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    save_fig_both(fig, out_base)


def plot_group_distribution(
    group_df: pd.DataFrame,
    group_name: str,
    out_base: Path,
    alpha: float | None,
    skewness_val: float | None,
) -> None:
    """Bar plot of proportion for one Konkle group; title includes power-law α and skewness."""
    fig, ax = plt.subplots(figsize=(max(10, len(group_df) * 0.2), 6))
    x = np.arange(len(group_df))
    ax.bar(x, group_df["Proportion"], color=KONKLE_COLORS.get(group_name, "grey"), edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(group_df["Category"], rotation=45, ha="right")
    ax.set_ylabel("Proportion detected")
    ax.set_xlabel("Category")
    parts = [f"Object detection distribution: {group_name.replace('_', ' ')} (n={len(group_df)})"]
    if alpha is not None:
        parts.append(f"power-law α={alpha:.3f}")
    if skewness_val is not None:
        parts.append(f"skewness={skewness_val:.3f}")
    ax.set_title(" | ".join(parts))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_fig_both(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-tailed distribution: skewness and 163/top-50 category plots with Konkle colors."
    )
    parser.add_argument(
        "--preprocessed_csv",
        type=Path,
        default=DEFAULT_PREPROCESSED_CSV,
        help="Preprocessed object detections CSV (class_name, num_detected, num_frames)",
    )
    parser.add_argument(
        "--categories_163",
        type=Path,
        default=DEFAULT_163_CATS_FILE,
        help="Text file with 163 category names (one per line)",
    )
    parser.add_argument(
        "--cdi_words",
        type=Path,
        default=DEFAULT_CDI_WORDS,
        help="CDI words CSV for Konkle category (is_animate, is_small, is_big)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for outputs",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 163 categories
    allowed = set(c.strip().lower() for c in load_163_categories(args.categories_163))
    print(f"Loaded {len(allowed)} allowed categories")

    # Category counts (only for allowed)
    count_df = get_category_counts(args.preprocessed_csv, allowed)
    print(f"Categories with data in preprocessed CSV: {len(count_df)}")
    if len(count_df) == 0:
        print("No data for the 163 categories; check paths and CSV columns.")
        return

    # Normalize category names to lowercase for consistency
    count_df["Category"] = count_df["Category"].str.strip().str.lower()
    # Ensure we have exactly the 163: add missing with 0 count so the plot has all 163
    missing = allowed - set(count_df["Category"])
    if missing:
        count_df = pd.concat([
            count_df,
            pd.DataFrame({"Category": list(missing), "Count": 0, "Total_frames": 0, "Proportion": 0.0}),
        ], ignore_index=True)
    count_df = count_df.sort_values("Proportion", ascending=False).reset_index(drop=True)
    assert len(count_df) == len(allowed), f"Expected {len(allowed)} rows, got {len(count_df)}"

    # Konkle mapping from CDI (needed before plots)
    konkle_map = load_konkle_mapping(args.cdi_words)
    print(f"Konkle mapping: {len(konkle_map)} uni_lemmas from CDI")
    count_df["konkle_category"] = count_df["Category"].str.lower().map(
        lambda c: konkle_map.get(c, "others")
    )

    # Proportion distribution: skewness and power-law fit (all 163)
    proportions = count_df["Proportion"].values
    proportions = np.maximum(proportions, 1e-15)  # avoid zeros for log
    skew_all = stats.skew(proportions)
    alpha_all, _, _ = fit_power_law(count_df["Proportion"].values)
    print(f"All 163: skewness (proportion) = {skew_all:.4f}, power-law α = {alpha_all:.4f}")

    with open(out_dir / "skewness_powerlaw.txt", "w") as f:
        f.write("Object detection proportion distribution (163 categories)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Skewness (proportion):     {skew_all:.4f}\n")
        f.write(f"Power-law exponent α:      {alpha_all:.4f}\n")
        f.write(f"Number of categories:      {len(count_df)}\n\n")

    # Save distribution table
    count_df.to_csv(out_dir / "category_counts_163.csv", index=False)

    # Plot 1: All 163 categories (proportion on y-axis)
    plot_bar_with_konkle(
        count_df,
        konkle_map,
        title="Object detection distribution (163 shared categories)",
        out_base=out_dir / "long_tailed_all_163_categories",
        top_n=None,
        power_law_alpha=alpha_all,
        skewness=skew_all,
    )
    print(f"Saved: {out_dir / 'long_tailed_all_163_categories.png'}, .pdf")

    # Plot 2: Top 50 categories
    plot_bar_with_konkle(
        count_df,
        konkle_map,
        title="Top 50 object categories by proportion detected (163 shared categories)",
        out_base=out_dir / "long_tailed_top50_categories",
        top_n=50,
        power_law_alpha=None,
        skewness=None,
    )
    print(f"Saved: {out_dir / 'long_tailed_top50_categories.png'}, .pdf")

    # Per-group plots: one per Konkle group, with power-law fit and skewness
    for group_name in ["animated", "small_obj", "big_obj", "others"]:
        group_df = count_df[count_df["konkle_category"] == group_name].copy()
        if len(group_df) == 0:
            continue
        group_df = group_df.sort_values("Proportion", ascending=False).reset_index(drop=True)
        p_vals = group_df["Proportion"].values
        p_vals = np.maximum(p_vals, 1e-15)
        skew_grp = stats.skew(p_vals)
        alpha_grp, _, _ = fit_power_law(group_df["Proportion"].values)
        print(f"  {group_name}: n={len(group_df)}, skewness={skew_grp:.4f}, α={alpha_grp:.4f}")
        with open(out_dir / "skewness_powerlaw.txt", "a") as f:
            f.write(f"{group_name}: n={len(group_df)}, skewness={skew_grp:.4f}, power-law α={alpha_grp:.4f}\n")
        plot_group_distribution(
            group_df,
            group_name,
            out_base=out_dir / f"long_tailed_group_{group_name}",
            alpha=alpha_grp,
            skewness_val=skew_grp,
        )
        print(f"Saved: {out_dir / f'long_tailed_group_{group_name}.png'}, .pdf")


if __name__ == "__main__":
    main()
