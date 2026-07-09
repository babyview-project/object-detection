"""
Long-tailed object detection distribution: skewness, power-law fits, and plots for the 163 shared categories.

- Y-axis: proportion detected (not raw count).
- Fits power-law to (rank, proportion) and reports exponent; reports empirical skewness.
- Plots all 163 and top 50 with Konkle colors; separate plots per group (small_obj, big_obj, animated, others).
- Fits power-law to each group's distribution.
- Saves all plots as .png and .pdf.

Uses filtered-embedding metadata CSV (one row per filtered detection) and the 163-category list; Konkle categories from CDI metadata.
Counts only filtered embeddings (e.g. merged_frame_detections_with_metadata_filtered-0.27.csv).
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

# Try Lato font for labels (fallback to default if not available)
def _set_lato_if_available():
    try:
        from matplotlib import font_manager
        lato_paths = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
        lato = [p for p in lato_paths if "lato" in p.lower() and ("regular" in p.lower() or "lato-regular" in p.lower())]
        if lato:
            font_manager.fontManager.addfont(lato[0])
            prop = font_manager.FontProperties(fname=lato[0])
            return {"family": prop.get_name(), "size": 14}
    except Exception:
        pass
    return {}
FONT_KW = _set_lato_if_available()

# Default paths (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_FILTERED_METADATA_CSV = PROJECT_ROOT / "frame_data" / "merged_frame_detections_with_metadata_filtered-0.27.csv"
DEFAULT_163_CATS_FILE = PROJECT_ROOT / "data" / "things_bv_overlap_categories_exclude_zero_precisions.txt"
DEFAULT_CDI_WORDS = PROJECT_ROOT / "data" / "cdi_words.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "long_tailed_163cats"


# Konkle colors for long_tailed_163cats plots
KONKLE_COLORS = {
    "big_obj": "#3A53A4",
    "small_obj": "#FAA41A",
    "animated": "#8250A0",
    "others": "#C0BDBD",
}

# CDI semantic categories: pastel color scheme (category -> hex)
CDI_SEMANTIC_COLORS = {
    "animals": "#B5EAD7",
    "body_parts": "#FFB7B2",
    "clothing": "#C7CEEA",
    "food_drink": "#FFDAC1",
    "furniture_rooms": "#E2F0CB",
    "household": "#F9D5E5",
    "outside": "#95E1D3",
    "people": "#FFEAA7",
    "toys": "#DDA0DD",
    "vehicles": "#81ECEC",
    "other": "#DCDDE1",
}
# Order for legend (consistent across plots)
CDI_SEMANTIC_ORDER = [
    "animals", "body_parts", "clothing", "food_drink", "furniture_rooms",
    "household", "outside", "people", "toys", "vehicles", "other",
]


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


def load_cdi_semantic_mapping(cdi_path: Path) -> dict[str, str]:
    """
    Build uni_lemma -> CDI semantic category from CDI (column 'category').
    Returns dict mapping lowercase uni_lemma to category string; unmapped get 'other'.
    """
    df = pd.read_csv(cdi_path, usecols=["uni_lemma", "category"])
    df["uni_lemma"] = df["uni_lemma"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    # First occurrence per uni_lemma
    out = df.drop_duplicates(subset=["uni_lemma"], keep="first").set_index("uni_lemma")["category"].to_dict()
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


def _apply_axis_style(ax: matplotlib.axes.Axes, use_font: bool = True) -> None:
    """Remove grid and spines; bold, larger tick labels; optional Lato font."""
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")
        if use_font and FONT_KW:
            label.set_fontfamily(FONT_KW.get("family", label.get_fontname()))


def get_category_counts_from_filtered(filtered_metadata_csv: Path, allowed_categories: set[str]) -> pd.DataFrame:
    """
    Count filtered embeddings per class_name from metadata CSV (one row per filtered detection).
    Keeps only allowed categories. Total_frames = total number of filtered embeddings (denominator for proportion).
    Returns DataFrame with columns: Category, Count, Total_frames, Proportion.
    """
    # Read only class_name to keep memory low for large CSVs
    df = pd.read_csv(filtered_metadata_csv, usecols=["class_name"])
    df["class_name"] = df["class_name"].str.strip().str.lower()
    df = df[df["class_name"].isin(allowed_categories)]
    total_filtered = len(df)
    agg = df.groupby("class_name").size().reset_index(name="Count")
    agg["Total_frames"] = total_filtered
    agg["Proportion"] = agg["Count"] / total_filtered if total_filtered else np.nan
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
    bold_labels: bool = False,
    y_max: float | None = None,
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
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")
    if bold_labels:
        ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight="bold")
        ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight="bold")
    if y_max is not None:
        ax.set_ylim(0, y_max)

    used = plot_df["konkle_category"].unique()
    legend_elements = [
        Patch(facecolor=KONKLE_COLORS[k], label=k.replace("_", " ")) for k in ["animated", "small_obj", "big_obj", "others"] if k in used
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    save_fig_both(fig, out_base)


def plot_bar_with_cdi_semantic(
    df: pd.DataFrame,
    cdi_semantic_map: dict[str, str],
    title: str,
    out_base: Path,
    top_n: int | None = None,
    power_law_alpha: float | None = None,
    skewness: float | None = None,
    bold_axis_labels: bool = False,
    y_max: float | None = None,
) -> None:
    """
    Bar plot of proportion per category, colored by CDI semantic category (pastel).
    out_base: path without extension (saved as .png and .pdf).
    """
    plot_df = df.head(top_n).copy() if top_n is not None else df.copy()
    plot_df["cdi_semantic"] = plot_df["Category"].str.lower().map(
        lambda c: cdi_semantic_map.get(c, "other")
    )
    colors = [CDI_SEMANTIC_COLORS.get(k, CDI_SEMANTIC_COLORS["other"]) for k in plot_df["cdi_semantic"]]

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
    _apply_axis_style(ax)
    if bold_axis_labels:
        ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight="bold")
        ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight="bold")
    if y_max is not None:
        ax.set_ylim(0, y_max)

    used = set(plot_df["cdi_semantic"].unique())
    legend_elements = [
        Patch(facecolor=CDI_SEMANTIC_COLORS[k], label=k.replace("_", " "))
        for k in CDI_SEMANTIC_ORDER
        if k in used
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
    y_max: float | None = None,
) -> None:
    """Bar plot of proportion for one Konkle group; title includes power-law α and skewness."""
    fig, ax = plt.subplots(figsize=(max(10, len(group_df) * 0.2), 6))
    x = np.arange(len(group_df))
    ax.bar(x, group_df["Proportion"], color=KONKLE_COLORS.get(group_name, "#C0BDBD"), edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(group_df["Category"], rotation=45, ha="right")
    ax.set_ylabel("Proportion detected")
    ax.set_xlabel("Category")
    parts = [f"Object detection distribution: {group_name.replace('_', ' ')} (top 12)"]
    if alpha is not None:
        parts.append(f"power-law α={alpha:.3f}")
    if skewness_val is not None:
        parts.append(f"skewness={skewness_val:.3f}")
    ax.set_title(" | ".join(parts))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", labelsize=14, width=1.2)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")
    if y_max is not None:
        ax.set_ylim(0, y_max)
    fig.tight_layout()
    save_fig_both(fig, out_base)


def plot_semantic_group_distribution(
    group_df: pd.DataFrame,
    semantic_category: str,
    out_base: Path,
    alpha: float | None = None,
    skewness_val: float | None = None,
    y_max: float | None = None,
) -> None:
    """Bar plot of proportion for one CDI semantic category (top N); pastel color, same style as Konkle group plots."""
    fig, ax = plt.subplots(figsize=(max(10, len(group_df) * 0.2), 6))
    x = np.arange(len(group_df))
    color = CDI_SEMANTIC_COLORS.get(semantic_category, CDI_SEMANTIC_COLORS["other"])
    ax.bar(x, group_df["Proportion"], color=color, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(group_df["Category"], rotation=45, ha="right")
    ax.set_ylabel("Proportion detected")
    ax.set_xlabel("Category")
    parts = [f"Object detection distribution: {semantic_category.replace('_', ' ')} (top {len(group_df)})"]
    if alpha is not None:
        parts.append(f"power-law α={alpha:.3f}")
    if skewness_val is not None:
        parts.append(f"skewness={skewness_val:.3f}")
    ax.set_title(" | ".join(parts))
    _apply_axis_style(ax)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    fig.tight_layout()
    save_fig_both(fig, out_base)


def plot_top50_plus_konkle_2x2(
    count_df: pd.DataFrame,
    konkle_map: dict[str, str],
    out_base: Path,
    y_max: float,
    top_n_per_konkle: int = 12,
) -> None:
    """
    Combined figure: Panel A = top 50 (Konkle colors: big/small/animate/others);
    Panels B–E = 2×2 grid of the four Konkle groups (top 12 each). All y-axes same scale.
    """
    konkle_groups = ["animated", "small_obj", "big_obj", "others"]
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1, 1], hspace=0.12, wspace=0.12)

    # Panel A: top 50 with Konkle colors (full width of first row)
    ax_top = fig.add_subplot(gs[0, :])
    top50 = count_df.head(50).copy()
    top50["konkle_category"] = top50["Category"].str.lower().map(
        lambda c: konkle_map.get(c, "others")
    )
    colors_50 = [KONKLE_COLORS[k] for k in top50["konkle_category"]]
    x50 = np.arange(50)
    ax_top.bar(x50, top50["Proportion"], color=colors_50, edgecolor="none")
    ax_top.set_xticks(x50)
    ax_top.set_xticklabels(top50["Category"], rotation=45, ha="right")
    ax_top.set_ylabel("Proportion detected")
    ax_top.set_title("(A) Top 50 categories (by size/animacy)")
    _apply_axis_style(ax_top)
    ax_top.set_ylim(0, y_max)

    # Panels B–E: 2×2 Konkle groups (rows 1–2, cols 0–1)
    panel_labels = ["B", "C", "D", "E"]
    for idx, group_name in enumerate(konkle_groups):
        row, col = 1 + (idx // 2), idx % 2
        ax = fig.add_subplot(gs[row, col])
        group_df = (
            count_df[count_df["konkle_category"] == group_name]
            .sort_values("Proportion", ascending=False)
            .head(top_n_per_konkle)
        )
        n_bars = len(group_df)
        if n_bars == 0:
            ax.set_visible(False)
            continue
        x = np.arange(n_bars)
        color = KONKLE_COLORS.get(group_name, "#C0BDBD")
        ax.bar(x, group_df["Proportion"], color=color, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(group_df["Category"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Proportion detected")
        _apply_axis_style(ax)
        ax.set_ylim(0, y_max)
        ax.set_title(f"({panel_labels[idx]}) {group_name.replace('_', ' ')}")

    save_fig_both(fig, out_base)


def plot_top50_plus_top10_per_semantic(
    count_df: pd.DataFrame,
    cdi_semantic_map: dict[str, str],
    out_base: Path,
    y_max: float,
    top_n_per_cat: int = 10,
) -> None:
    """
    Wide combined figure: Panel A = top 50 (full width); Panels B–I = top 10 per semantic
    category in a 4×2 grid. Excludes 'outside'. Panel A uses y_max; subplots B–I each
    scale to their own data (no shared y-axis) so low-frequency categories are readable.
    Semantic pastel colors.
    """
    # Semantic categories for the 4×2 grid: all present in data except 'outside'
    # Order by total proportion (frequency) descending so first row = most frequent
    candidates = [
        c for c in CDI_SEMANTIC_ORDER
        if c != "outside"
        and c in count_df["cdi_semantic"].values
        and (count_df["cdi_semantic"] == c).any()
    ]
    total_prop = {c: count_df.loc[count_df["cdi_semantic"] == c, "Proportion"].sum() for c in candidates}
    semantic_for_grid = sorted(candidates, key=lambda c: total_prop[c], reverse=True)[:8]
    if len(semantic_for_grid) == 0:
        return

    # Wide figure: full-width top row, then 2 rows × 4 cols; minimal spacing
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.04, wspace=0.04)

    # Panel A: top 50 (full width)
    ax_top = fig.add_subplot(gs[0, :])
    top50 = count_df.head(50).copy()
    top50["cdi_semantic"] = top50["Category"].str.lower().map(
        lambda c: cdi_semantic_map.get(c, "other")
    )
    colors_50 = [CDI_SEMANTIC_COLORS.get(k, CDI_SEMANTIC_COLORS["other"]) for k in top50["cdi_semantic"]]
    x50 = np.arange(50)
    ax_top.bar(x50, top50["Proportion"], color=colors_50, edgecolor="none")
    ax_top.set_xticks(x50)
    ax_top.set_xticklabels(top50["Category"], rotation=45, ha="right")
    ax_top.set_ylabel("Proportion detected")
    ax_top.set_title("(A) Top 50 categories overall")
    _apply_axis_style(ax_top)
    ax_top.set_ylim(0, y_max)

    # Panels B–I: 4×2 grid, top 10 per semantic category; label inside plot, color-coded
    panel_labels = ["B", "C", "D", "E", "F", "G", "H", "I"]
    for idx, sem_cat in enumerate(semantic_for_grid):
        row, col = 1 + idx // 4, idx % 4
        ax = fig.add_subplot(gs[row, col])
        group_df = (
            count_df[count_df["cdi_semantic"] == sem_cat]
            .sort_values("Proportion", ascending=False)
            .head(top_n_per_cat)
        )
        n_bars = len(group_df)
        if n_bars == 0:
            ax.set_visible(False)
            continue
        x = np.arange(n_bars)
        color = CDI_SEMANTIC_COLORS.get(sem_cat, CDI_SEMANTIC_COLORS["other"])
        ax.bar(x, group_df["Proportion"], color=color, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(group_df["Category"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Proportion detected")
        _apply_axis_style(ax)
        # No shared y_max: each semantic subplot scales to its own data (some categories have much lower frequency)
        label = f"({panel_labels[idx]}) {sem_cat.replace('_', ' ')}"
        ax.set_title(label, fontsize=10, color=color, fontweight="bold", pad=4)

    save_fig_both(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-tailed distribution: skewness and 163/top-50 category plots with Konkle colors."
    )
    parser.add_argument(
        "--filtered_metadata_csv",
        type=Path,
        default=DEFAULT_FILTERED_METADATA_CSV,
        help="Filtered-embedding metadata CSV (one row per filtered detection; must have class_name)",
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

    # Category counts from filtered embeddings only (only for allowed categories)
    count_df = get_category_counts_from_filtered(args.filtered_metadata_csv, allowed)
    print(f"Categories with data in filtered metadata CSV: {len(count_df)}")
    if len(count_df) == 0:
        print("No data for the 163 categories in filtered metadata; check paths and CSV columns.")
        return

    # Normalize category names to lowercase for consistency
    count_df["Category"] = count_df["Category"].str.strip().str.lower()
    # Ensure we have exactly the 163: add missing with 0 count so the plot has all 163
    missing = allowed - set(count_df["Category"])
    total_filtered = count_df["Total_frames"].iloc[0] if len(count_df) else 0
    if missing:
        count_df = pd.concat([
            count_df,
            pd.DataFrame({"Category": list(missing), "Count": 0, "Total_frames": total_filtered, "Proportion": 0.0}),
        ], ignore_index=True)
    count_df = count_df.sort_values("Proportion", ascending=False).reset_index(drop=True)
    assert len(count_df) == len(allowed), f"Expected {len(allowed)} rows, got {len(count_df)}"

    # Konkle mapping from CDI (needed before plots)
    konkle_map = load_konkle_mapping(args.cdi_words)
    print(f"Konkle mapping: {len(konkle_map)} uni_lemmas from CDI")
    count_df["konkle_category"] = count_df["Category"].str.lower().map(
        lambda c: konkle_map.get(c, "others")
    )

    # CDI semantic category mapping (for semantic-colored plots)
    cdi_semantic_map = load_cdi_semantic_mapping(args.cdi_words)
    count_df["cdi_semantic"] = count_df["Category"].str.lower().map(
        lambda c: cdi_semantic_map.get(c, "other")
    )
    print(f"CDI semantic mapping: {len(cdi_semantic_map)} uni_lemmas; categories in data: {sorted(count_df['cdi_semantic'].unique())}")

    # Proportion distribution: skewness and power-law fit (all 163)
    proportions = count_df["Proportion"].values
    proportions = np.maximum(proportions, 1e-15)  # avoid zeros for log
    skew_all = stats.skew(proportions)
    alpha_all, _, _ = fit_power_law(count_df["Proportion"].values)
    print(f"All 163: skewness (proportion) = {skew_all:.4f}, power-law α = {alpha_all:.4f}")

    with open(out_dir / "skewness_powerlaw.txt", "w") as f:
        f.write("Object detection proportion distribution (163 categories, filtered embeddings only)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Skewness (proportion):     {skew_all:.4f}\n")
        f.write(f"Power-law exponent α:      {alpha_all:.4f}\n")
        f.write(f"Number of categories:      {len(count_df)}\n\n")

    # Save distribution table (filtered embeddings only)
    count_df.to_csv(out_dir / "category_counts_163_filtered.csv", index=False)

    # Shared y-axis scale for all plots (max proportion + small padding)
    global_y_max = count_df["Proportion"].max() * 1.02

    # Plot 1: All 163 categories (proportion on y-axis, filtered embeddings only)
    plot_bar_with_konkle(
        count_df,
        konkle_map,
        title="Object detection distribution (163 shared categories, filtered embeddings)",
        out_base=out_dir / "long_tailed_all_163_categories",
        top_n=None,
        power_law_alpha=alpha_all,
        skewness=skew_all,
        y_max=global_y_max,
    )
    print(f"Saved: {out_dir / 'long_tailed_all_163_categories.png'}, .pdf")

    # Plot 2: Top 50 categories
    plot_bar_with_konkle(
        count_df,
        konkle_map,
        title="Top 50 object categories by proportion detected (163 shared, filtered embeddings)",
        out_base=out_dir / "long_tailed_top50_categories",
        top_n=50,
        power_law_alpha=None,
        skewness=None,
        bold_labels=True,
        y_max=global_y_max,
    )
    print(f"Saved: {out_dir / 'long_tailed_top50_categories.png'}, .pdf")

    # Per-group plots: one per Konkle group, top 12 by proportion, with power-law fit and skewness (full group)
    TOP_N_PER_GROUP = 12
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
            group_df.head(TOP_N_PER_GROUP),
            group_name,
            out_base=out_dir / f"long_tailed_group_{group_name}",
            alpha=alpha_grp,
            skewness_val=skew_grp,
            y_max=global_y_max,
        )
        print(f"Saved: {out_dir / f'long_tailed_group_{group_name}.png'}, .pdf")

    # Combined: top 50 (Konkle) + 2×2 Konkle groups, same y-axis scale
    plot_top50_plus_konkle_2x2(
        count_df,
        konkle_map,
        out_base=out_dir / "long_tailed_top50_plus_konkle_2x2",
        y_max=global_y_max,
        top_n_per_konkle=TOP_N_PER_GROUP,
    )
    print(f"Saved: {out_dir / 'long_tailed_top50_plus_konkle_2x2.png'}, .pdf")

    # --- CDI semantic category plots (pastel colors, Lato if available) ---
    # All 163 and top 50 colored by CDI semantic
    plot_bar_with_cdi_semantic(
        count_df,
        cdi_semantic_map,
        title="Object detection distribution (163 shared categories, CDI semantic)",
        out_base=out_dir / "long_tailed_all_163_categories_semantic",
        top_n=None,
        power_law_alpha=alpha_all,
        skewness=skew_all,
        y_max=global_y_max,
    )
    print(f"Saved: {out_dir / 'long_tailed_all_163_categories_semantic.png'}, .pdf")
    plot_bar_with_cdi_semantic(
        count_df,
        cdi_semantic_map,
        title="Top 50 object categories by proportion (CDI semantic)",
        out_base=out_dir / "long_tailed_top50_categories_semantic",
        top_n=50,
        bold_axis_labels=True,
        y_max=global_y_max,
    )
    print(f"Saved: {out_dir / 'long_tailed_top50_categories_semantic.png'}, .pdf")

    # Combined: top 50 overall + top 10 per CDI semantic category (two panels)
    plot_top50_plus_top10_per_semantic(
        count_df,
        cdi_semantic_map,
        out_base=out_dir / "long_tailed_top50_plus_top10_per_semantic",
        y_max=global_y_max,
        top_n_per_cat=10,
    )
    print(f"Saved: {out_dir / 'long_tailed_top50_plus_top10_per_semantic.png'}, .pdf")

    # Per–CDI semantic category: top 12 each, one plot per category
    semantic_categories = [
        c for c in CDI_SEMANTIC_ORDER
        if c in count_df["cdi_semantic"].values and (count_df["cdi_semantic"] == c).any()
    ]
    for sem_cat in semantic_categories:
        group_df = count_df[count_df["cdi_semantic"] == sem_cat].copy()
        group_df = group_df.sort_values("Proportion", ascending=False).reset_index(drop=True)
        if len(group_df) == 0:
            continue
        p_vals = np.maximum(group_df["Proportion"].values, 1e-15)
        skew_grp = stats.skew(p_vals)
        alpha_grp, _, _ = fit_power_law(group_df["Proportion"].values)
        plot_semantic_group_distribution(
            group_df.head(TOP_N_PER_GROUP),
            sem_cat,
            out_base=out_dir / f"long_tailed_semantic_{sem_cat}",
            alpha=alpha_grp,
            skewness_val=skew_grp,
            # No shared y_max: each semantic category scales to its own data (some have much lower frequency)
        )
        print(f"Saved: {out_dir / f'long_tailed_semantic_{sem_cat}.png'}, .pdf")


if __name__ == "__main__":
    main()
