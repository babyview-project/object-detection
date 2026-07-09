#!/usr/bin/env python3
"""
Generate one figure: BV vs THINGS category correlation variability (CLIP + DINOv3).

- Montage rows: fixed high/low categories with BV and THINGS images; each row shows
  Spearman r for both CLIP and DINOv3.
- Bottom: one horizontal histogram plotting both CLIP and DINOv3 correlation
  distributions.

Usage:
  cd analysis/vss-2026
  python plot_bv_things_correlation_montages.py
  python plot_bv_things_correlation_montages.py --cropped-dir /path/to/bv/crops --things-images-dir /path/to/things/images
"""
from pathlib import Path
import argparse
import random
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
COMP_DIR = SCRIPT_DIR / "bv_things_comp_12252025"
BV_CLIP_CSV = COMP_DIR / "bv_clip_filtered_zscored_hierarchical_163cats/normalized_filtered_embeddings_alphabetical.csv"
THINGS_CLIP_CSV = COMP_DIR / "things_clip_filtered_zscored_hierarchical_163cats/normalized_filtered_embeddings_alphabetical.csv"
BV_DINOV3_CSV = COMP_DIR / "bv_dinov3_filtered_zscored_hierarchical_163cats/normalized_filtered_embeddings_alphabetical.csv"
THINGS_DINOV3_CSV = COMP_DIR / "things_dinov3_filtered_zscored_hierarchical_163cats/normalized_filtered_embeddings_alphabetical.csv"
CATEGORY_INCLUDE_FILE = SCRIPT_DIR / "../../data/things_bv_overlap_categories_exclude_zero_precisions.txt"
DEFAULT_CROPPED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class")
THINGS_BASE = Path("/ccn2/dataset/babyview/outputs_20250312")
DEFAULT_THINGS_IMAGES_DIR = THINGS_BASE / "things_bv_overlapping_categories_corrected"
# CLIP-filtered embedding list: one .npy path per line; only BV crops whose (category, stem) appear here are used
CLIP_FILTER_LIST = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/clip_image_embeddings_filtered-by-clip-0.27_exclude-people_exclude-subject-00270001.txt")
# Exclude body-part categories when selecting high/low correlation (no mouth, foot, hand, etc.)
BODY_PARTS = {
    "ankle", "arm", "ear", "eye", "face", "finger", "foot", "hair", "hand", "leg",
    "mouth", "nose", "toe", "tooth", "neck",
}
# Fixed categories to show: high and low Spearman r (order preserved)
HIGH_CORRELATION_CATEGORIES = ["zebra", "giraffe", "cloud", "pasta", "shoe"]
LOW_CORRELATION_CATEGORIES = ["mop", "soap", "elephant", "bear", "slipper"]
N_CROPS = 10
CELL_SIZE = (80, 80)
N_COLS_MONTAGE = 5


def get_category_crop_dir(root_dir, cat_name):
    """Return Path to category subfolder (match by name, case-insensitive)."""
    root_dir = Path(root_dir)
    cat_lower = cat_name.strip().lower()
    direct = root_dir / cat_name
    if direct.exists() and direct.is_dir():
        return direct
    if root_dir.exists():
        for p in root_dir.iterdir():
            if p.is_dir() and p.name.lower() == cat_lower:
                return p
    return root_dir / cat_name


def load_embeddings_csv(csv_path):
    """Load embedding CSV: index = category, columns = dim_*."""
    df = pd.read_csv(csv_path, index_col=0)
    # index might be 'Unnamed: 0' or first column
    if df.index.name is None or str(df.index.name).startswith("Unnamed"):
        if "Unnamed: 0" in df.columns:
            df = df.set_index("Unnamed: 0")
    dim_cols = [c for c in df.columns if re.match(r"dim_\d+", str(c))]
    if not dim_cols:
        dim_cols = [c for c in df.columns if c != "category"]
    X = df[dim_cols].values.astype(np.float64)
    categories = [str(i).strip() for i in df.index]
    return categories, X


def compute_spearman_per_category(bv_csv, things_csv, category_include_file=None):
    """Return (list of dict with category, spearman_r), all_spearman_rs."""
    cats_bv, X_bv = load_embeddings_csv(bv_csv)
    cats_th, X_th = load_embeddings_csv(things_csv)
    # Normalize names
    cats_bv = [c.strip().lower() for c in cats_bv]
    cats_th = [c.strip().lower() for c in cats_th]
    set_bv = set(cats_bv)
    set_th = set(cats_th)
    matching = [c for c in cats_bv if c in set_th]
    if category_include_file and Path(category_include_file).exists():
        with open(category_include_file) as f:
            included = set(line.strip().lower() for line in f if line.strip())
        matching = [c for c in matching if c in included]
    cat_to_idx_bv = {c: i for i, c in enumerate(cats_bv)}
    cat_to_idx_th = {c: i for i, c in enumerate(cats_th)}
    results = []
    all_rs = []
    for cat in matching:
        i_bv = cat_to_idx_bv[cat]
        i_th = cat_to_idx_th[cat]
        v1 = X_bv[i_bv]
        v2 = X_th[i_th]
        mask = np.isfinite(v1) & np.isfinite(v2)
        v1c, v2c = v1[mask], v2[mask]
        if len(v1c) >= 3:
            r, _ = spearmanr(v1c, v2c)
        else:
            r = np.nan
        results.append({"category": cat, "spearman_r": r})
        if not np.isnan(r):
            all_rs.append(r)
    return results, all_rs


def load_clip_filtered_stems_for_categories(filter_list_path, categories_set):
    """
    Stream the CLIP-filtered list (one .npy path per line) and return set of (category_lower, stem)
    only for categories in categories_set. Keeps memory low when file is huge.
    """
    categories_set = {c.strip().lower() for c in categories_set}
    allowed = set()
    path = Path(filter_list_path)
    if not path.exists():
        return allowed
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            cat = p.parent.name.strip().lower()
            if cat not in categories_set:
                continue
            stem = p.stem if p.suffix.lower() == ".npy" else p.name
            allowed.add((cat, stem))
    return allowed


# Stem pattern: {category}_{confidence}_{subject_id}_{gcp_name}_processed_{frame_id} (from embedding/crop pipeline)
_STEM_PATTERN = re.compile(r"^(.+?)_([\d.]+)_(\d+)_(.+?)_processed_(\d+)$")


def _source_key_from_stem(stem):
    """Return (subject_id, gcp_name) for diversification, or (stem, stem) if unparsed."""
    m = _STEM_PATTERN.match(stem)
    if m:
        return (m.group(3), m.group(4))  # subject_id, gcp_name
    return (stem, stem)


def _sample_paths_diversified(paths, n):
    """
    Sample up to n paths, preferring one per (subject, video) so we avoid multiple crops
    from the same subject or same recording. Groups by (subject_id, gcp_name) from stem.
    """
    if len(paths) <= n:
        return list(paths)
    groups = defaultdict(list)
    for p in paths:
        key = _source_key_from_stem(p.stem)
        groups[key].append(p)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    chosen = []
    for key in group_keys:
        if len(chosen) >= n:
            break
        pool = groups[key]
        random.shuffle(pool)
        chosen.append(pool[0])
    if len(chosen) < n:
        remaining = [p for p in paths if p not in chosen]
        random.shuffle(remaining)
        need = n - len(chosen)
        chosen.extend(remaining[:need])
    random.shuffle(chosen)
    return chosen


def get_bv_crop_paths(cropped_dir, cat_name, n=10, allowed_stems=None, random_sample=False):
    """
    Return list of up to n image paths for category from BV cropped dir.
    If allowed_stems is provided (set of (category_lower, stem)), only crops whose (cat, stem) is in it are used (CLIP-filtered).
    If random_sample is True, sample n paths with diversification across subject and video (one per subject+video when possible).
    """
    cat_dir = get_category_crop_dir(cropped_dir, cat_name)
    if not cat_dir.exists():
        return []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(cat_dir.glob(ext))
    cat_lower = cat_name.strip().lower()
    if allowed_stems is not None:
        paths = [p for p in paths if (cat_lower, p.stem) in allowed_stems]
    paths = sorted(paths, key=lambda p: p.name)
    if random_sample and len(paths) > n:
        paths = _sample_paths_diversified(paths, n)
    else:
        paths = paths[:n]
    return paths


def get_things_crop_paths(things_images_dir, cat_name, n=10, random_sample=False):
    """Return list of up to n image paths for category from THINGS dir. If random_sample is True, sample n at random."""
    if things_images_dir is None or not Path(things_images_dir).exists():
        return []
    cat_dir = get_category_crop_dir(things_images_dir, cat_name)
    if not cat_dir.exists():
        return []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(cat_dir.glob(ext))
    paths = sorted(paths, key=lambda p: p.name)
    if random_sample and len(paths) > n:
        paths = random.sample(paths, n)
    else:
        paths = paths[:n]
    return paths


def make_montage(paths, cell_size, n_cols):
    """Tile images into a single PIL Image (no labels)."""
    if not paths:
        return None
    n = len(paths)
    n_rows = (n + n_cols - 1) // n_cols
    out = Image.new("RGB", (n_cols * cell_size[0], n_rows * cell_size[1]), (240, 240, 240))
    for idx, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB")
            if img.size != cell_size:
                img = img.resize(cell_size, Image.Resampling.LANCZOS)
            row, col = idx // n_cols, idx % n_cols
            out.paste(img, (col * cell_size[0], row * cell_size[1]))
        except Exception:
            pass
    return out


def build_figure(
    model_name,
    results_sorted,
    all_spearman_rs,
    cropped_dir,
    things_images_dir,
    n_high=3,
    n_low=3,
    n_crops=N_CROPS,
    cell_size=CELL_SIZE,
    n_cols=N_COLS_MONTAGE,
    allowed_stems=None,
    exclude_body_parts=True,
    fixed_high_categories=None,
    fixed_low_categories=None,
    crop_sample_seed=None,
):
    """Create one figure: montage rows (BV | THINGS), right = histogram. Uses fixed_high/low if provided."""
    random_sample = crop_sample_seed is not None
    if random_sample:
        random.seed(crop_sample_seed)
    cat_to_result = {r["category"].strip().lower(): r for r in results_sorted}

    if fixed_high_categories is not None and fixed_low_categories is not None:
        rows = []
        for cat in fixed_high_categories:
            c = cat.strip().lower()
            rows.append(cat_to_result.get(c, {"category": cat, "spearman_r": np.nan}))
        for cat in fixed_low_categories:
            c = cat.strip().lower()
            rows.append(cat_to_result.get(c, {"category": cat, "spearman_r": np.nan}))
    else:
        valid = [r for r in results_sorted if not np.isnan(r["spearman_r"])]
        if exclude_body_parts:
            valid = [r for r in valid if r["category"].strip().lower() not in BODY_PARTS]
        high = valid[-n_high:][::-1]
        low = valid[:n_low]
        rows = high + low
        if len(rows) != n_high + n_low:
            valid_all = [r for r in results_sorted if not np.isnan(r["spearman_r"]) and (not exclude_body_parts or r["category"].strip().lower() not in BODY_PARTS)]
            if len(valid_all) >= n_high + n_low:
                rows = valid_all[-n_high:][::-1] + valid_all[:n_low]
            else:
                rows = valid_all

    n_rows = len(rows)
    from matplotlib.gridspec import GridSpec
    # Montages on top (2 cols); horizontal histogram in one full-width row below
    hist_height_ratio = 0.45
    height_ratios = [1.0] * n_rows + [hist_height_ratio]
    fig = plt.figure(figsize=(10, 2.2 * n_rows + 1.5))
    gs = GridSpec(n_rows + 1, 2, figure=fig, height_ratios=height_ratios, hspace=0.28, wspace=0.05, top=0.94, bottom=0.06)

    for i, row in enumerate(rows):
        cat = row["category"]
        r_val = row["spearman_r"]
        bv_paths = get_bv_crop_paths(cropped_dir, cat, n_crops, allowed_stems=allowed_stems, random_sample=random_sample)
        th_paths = get_things_crop_paths(things_images_dir, cat, n_crops, random_sample=random_sample)
        ax_bv = fig.add_subplot(gs[i, 0])
        ax_th = fig.add_subplot(gs[i, 1])
        if bv_paths:
            mont_bv = make_montage(bv_paths, cell_size, n_cols)
            if mont_bv is not None:
                ax_bv.imshow(np.array(mont_bv))
        ax_bv.axis("off")
        ax_bv.set_title(f"{cat}, r = {r_val:.2f}", fontsize=9)
        if th_paths:
            mont_th = make_montage(th_paths, cell_size, n_cols)
            if mont_th is not None:
                ax_th.imshow(np.array(mont_th))
        ax_th.axis("off")
    # Column labels above first row
    fig.text(0.25, 0.935, "BV", fontsize=10, ha="center")
    fig.text(0.75, 0.935, "THINGS", fontsize=10, ha="center")

    # Horizontal histogram spanning full width below all montages
    ax_hist = fig.add_subplot(gs[n_rows, :])
    rs = [x for x in all_spearman_rs if not np.isnan(x)]
    if rs:
        ax_hist.hist(rs, bins=20, orientation="vertical", color="steelblue", alpha=0.8, edgecolor="white")
        ax_hist.axvline(x=np.median(rs), color="gray", linestyle="--", linewidth=1, label=f"Median = {np.median(rs):.2f}")
        ax_hist.set_xlabel("Correlation (Spearman r)")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.set_title("Correlation distribution", fontsize=10)

    title_suffix = f"{n_rows // 2} high + {n_rows // 2} low Spearman r" if n_rows else "Spearman r"
    fig.suptitle(f"BV–THINGS category correlation variability ({model_name})\n{title_suffix}", fontsize=12, y=0.96)
    return fig


def _build_one_montage_figure(
    rows,
    all_rs_clip,
    all_rs_dinov3,
    cropped_dir,
    things_images_dir,
    title_label,
    n_crops=N_CROPS,
    cell_size=CELL_SIZE,
    n_cols=N_COLS_MONTAGE,
    allowed_stems=None,
    random_sample=False,
):
    """Build one figure: montage rows (BV | THINGS) for given rows; bottom = histogram. title_label is e.g. 'high' or 'low'."""
    n_rows = len(rows)
    if n_rows == 0:
        return plt.figure(figsize=(10, 3))
    from matplotlib.gridspec import GridSpec
    hist_height_ratio = 0.45
    height_ratios = [1.0] * n_rows + [hist_height_ratio]
    fig = plt.figure(figsize=(10, 2.2 * n_rows + 1.5))
    gs = GridSpec(n_rows + 1, 2, figure=fig, height_ratios=height_ratios, hspace=0.28, wspace=0.05, top=0.94, bottom=0.06)

    for i, row in enumerate(rows):
        cat, r_clip, r_dino = row["category"], row["r_clip"], row["r_dino"]
        bv_paths = get_bv_crop_paths(cropped_dir, cat, n_crops, allowed_stems=allowed_stems, random_sample=random_sample)
        th_paths = get_things_crop_paths(things_images_dir, cat, n_crops, random_sample=random_sample)
        ax_bv = fig.add_subplot(gs[i, 0])
        ax_th = fig.add_subplot(gs[i, 1])
        if bv_paths:
            mont_bv = make_montage(bv_paths, cell_size, n_cols)
            if mont_bv is not None:
                ax_bv.imshow(np.array(mont_bv))
        ax_bv.axis("off")
        ax_bv.set_title(f"{cat}\nr_CLIP={r_clip:.2f}  r_DINOv3={r_dino:.2f}", fontsize=9)
        if th_paths:
            mont_th = make_montage(th_paths, cell_size, n_cols)
            if mont_th is not None:
                ax_th.imshow(np.array(mont_th))
        ax_th.axis("off")
    fig.text(0.25, 0.935, "BV", fontsize=10, ha="center")
    fig.text(0.75, 0.935, "THINGS", fontsize=10, ha="center")

    ax_hist = fig.add_subplot(gs[n_rows, :])
    rs_clip = [x for x in all_rs_clip if not np.isnan(x)]
    rs_dino = [x for x in all_rs_dinov3 if not np.isnan(x)]
    all_vals = rs_clip + rs_dino
    if all_vals:
        bins = np.linspace(min(all_vals), max(all_vals), 21)
        if rs_clip:
            ax_hist.hist(rs_clip, bins=bins, orientation="vertical", color="C0", alpha=0.6, edgecolor="white", label=f"CLIP (n={len(rs_clip)})")
        if rs_dino:
            ax_hist.hist(rs_dino, bins=bins, orientation="vertical", color="C1", alpha=0.6, edgecolor="white", label=f"DINOv3 (n={len(rs_dino)})")
        ax_hist.set_xlabel("Correlation (Spearman r)")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.set_title("Correlation distribution (CLIP & DINOv3)", fontsize=10)

    fig.suptitle(f"BV–THINGS category correlation variability ({title_label})\n{len(rows)} categories", fontsize=12, y=0.96)
    return fig


def build_combined_figure(
    results_sorted_clip,
    all_rs_clip,
    results_sorted_dinov3,
    all_rs_dinov3,
    cropped_dir,
    things_images_dir,
    n_high=5,
    n_low=5,
    n_crops=N_CROPS,
    cell_size=CELL_SIZE,
    n_cols=N_COLS_MONTAGE,
    allowed_stems=None,
    fixed_high_categories=None,
    fixed_low_categories=None,
    crop_sample_seed=None,
):
    """Build two figures: one for high-correlation categories, one for low. Returns (fig_high, fig_low)."""
    random_sample = crop_sample_seed is not None
    if random_sample:
        random.seed(crop_sample_seed)
    cat_to_clip = {r["category"].strip().lower(): r for r in results_sorted_clip}
    cat_to_dino = {r["category"].strip().lower(): r for r in results_sorted_dinov3}
    if fixed_high_categories is not None and fixed_low_categories is not None:
        rows_high = []
        for cat in fixed_high_categories:
            c = cat.strip().lower()
            r_clip = cat_to_clip.get(c, {"category": cat, "spearman_r": np.nan})["spearman_r"]
            r_dino = cat_to_dino.get(c, {"category": cat, "spearman_r": np.nan})["spearman_r"]
            rows_high.append({"category": cat, "r_clip": r_clip, "r_dino": r_dino})
        rows_low = []
        for cat in fixed_low_categories:
            c = cat.strip().lower()
            r_clip = cat_to_clip.get(c, {"category": cat, "spearman_r": np.nan})["spearman_r"]
            r_dino = cat_to_dino.get(c, {"category": cat, "spearman_r": np.nan})["spearman_r"]
            rows_low.append({"category": cat, "r_clip": r_clip, "r_dino": r_dino})
    else:
        valid_clip = [r for r in results_sorted_clip if not np.isnan(r["spearman_r"]) and r["category"].strip().lower() not in BODY_PARTS]
        high = valid_clip[-n_high:][::-1]
        low = valid_clip[:n_low]
        rows_high = [{"category": r["category"], "r_clip": r["spearman_r"], "r_dino": cat_to_dino.get(r["category"].strip().lower(), {"spearman_r": np.nan})["spearman_r"]} for r in high]
        rows_low = [{"category": r["category"], "r_clip": r["spearman_r"], "r_dino": cat_to_dino.get(r["category"].strip().lower(), {"spearman_r": np.nan})["spearman_r"]} for r in low]

    fig_high = _build_one_montage_figure(
        rows_high, all_rs_clip, all_rs_dinov3, cropped_dir, things_images_dir, "high",
        n_crops=n_crops, cell_size=cell_size, n_cols=n_cols, allowed_stems=allowed_stems, random_sample=random_sample,
    )
    fig_low = _build_one_montage_figure(
        rows_low, all_rs_clip, all_rs_dinov3, cropped_dir, things_images_dir, "low",
        n_crops=n_crops, cell_size=cell_size, n_cols=n_cols, allowed_stems=allowed_stems, random_sample=random_sample,
    )
    return fig_high, fig_low


def _get_selected_categories(results_sorted, n_high=3, n_low=3, exclude_body_parts=True, fixed_high=None, fixed_low=None):
    """Return list of result dicts for selected rows. If fixed_high/fixed_low are given, use those and look up r."""
    if fixed_high is not None and fixed_low is not None:
        cat_to_result = {r["category"].strip().lower(): r for r in results_sorted}
        selected = []
        for cat in fixed_high + fixed_low:
            c = cat.strip().lower()
            selected.append(cat_to_result.get(c, {"category": cat, "spearman_r": np.nan}))
        return selected
    valid = [r for r in results_sorted if not np.isnan(r["spearman_r"])]
    if exclude_body_parts:
        valid = [r for r in valid if r["category"].strip().lower() not in BODY_PARTS]
    high = valid[-n_high:][::-1]
    low = valid[:n_low]
    return high + low


def main():
    parser = argparse.ArgumentParser(description="BV–THINGS correlation montages + histogram (CLIP and DINOv3)")
    parser.add_argument("--cropped-dir", type=Path, default=DEFAULT_CROPPED_DIR, help="BV cropped-by-class images")
    parser.add_argument("--things-images-dir", type=Path, default=DEFAULT_THINGS_IMAGES_DIR, help="THINGS images by category")
    parser.add_argument("--clip-filter-list", type=Path, default=CLIP_FILTER_LIST, help="CLIP-filtered embedding paths (one per line) for BV crop selection")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: script dir)")
    parser.add_argument("--no-clip-filter", action="store_true", help="Do not filter BV crops by CLIP list (use all crops)")
    parser.add_argument("--crop-sample-seed", type=int, default=None, metavar="N", help="Random seed for sampling which crops to show per category (default: first n by name). Use e.g. 42 or 123 to sample other crops.")
    args = parser.parse_args()
    out_dir = args.out_dir or SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir = args.cropped_dir
    things_dir = args.things_images_dir
    if not cropped_dir.exists():
        print(f"Warning: BV cropped dir not found: {cropped_dir}. Montages will be empty.")
    if things_dir is None or not things_dir.exists():
        print(f"Warning: THINGS images dir not found: {things_dir}. THINGS montages will be empty.")

    include_file = str(CATEGORY_INCLUDE_FILE) if CATEGORY_INCLUDE_FILE else None

    if not BV_CLIP_CSV.exists() or not THINGS_CLIP_CSV.exists():
        print("Missing CLIP CSVs; cannot build combined figure.")
        return
    if not BV_DINOV3_CSV.exists() or not THINGS_DINOV3_CSV.exists():
        print("Missing DINOv3 CSVs; cannot build combined figure.")
        return

    results_clip, all_rs_clip = compute_spearman_per_category(BV_CLIP_CSV, THINGS_CLIP_CSV, include_file)
    results_dino, all_rs_dino = compute_spearman_per_category(BV_DINOV3_CSV, THINGS_DINOV3_CSV, include_file)
    results_sorted_clip = sorted(results_clip, key=lambda x: x["spearman_r"] if not np.isnan(x["spearman_r"]) else -np.inf)
    results_sorted_dinov3 = sorted(results_dino, key=lambda x: x["spearman_r"] if not np.isnan(x["spearman_r"]) else -np.inf)

    categories_needed = {c.strip().lower() for c in HIGH_CORRELATION_CATEGORIES + LOW_CORRELATION_CATEGORIES}
    # BV montage crops are restricted to (category, stem) pairs in CLIP_FILTER_LIST when not --no-clip-filter
    allowed_stems = None
    if args.no_clip_filter:
        allowed_stems = None  # use all crops
    elif args.clip_filter_list and args.clip_filter_list.exists():
        print(f"Loading CLIP-filtered stems for {len(categories_needed)} categories (BV montages will only use these crops)...")
        allowed_stems = load_clip_filtered_stems_for_categories(args.clip_filter_list, categories_needed)
        print(f"  Found {len(allowed_stems)} allowed (category, stem) pairs.")
        if len(allowed_stems) == 0:
            print("  Warning: no stems found for these categories; using all BV crops for montages.")
            allowed_stems = None
    else:
        if args.clip_filter_list:
            print(f"Warning: CLIP filter list not found: {args.clip_filter_list}. Using all BV crops for montages.")
        allowed_stems = None  # fall back to all crops when filter file is missing

    fig_high, fig_low = build_combined_figure(
        results_sorted_clip,
        all_rs_clip,
        results_sorted_dinov3,
        all_rs_dino,
        cropped_dir,
        things_dir,
        n_high=len(HIGH_CORRELATION_CATEGORIES),
        n_low=len(LOW_CORRELATION_CATEGORIES),
        n_crops=N_CROPS,
        cell_size=CELL_SIZE,
        n_cols=N_COLS_MONTAGE,
        allowed_stems=allowed_stems,
        fixed_high_categories=HIGH_CORRELATION_CATEGORIES,
        fixed_low_categories=LOW_CORRELATION_CATEGORIES,
        crop_sample_seed=args.crop_sample_seed,
    )
    for name, fig in [("high", fig_high), ("low", fig_low)]:
        base_name = f"bv_things_correlation_montages_{name}"
        for ext, opts in [("png", {"dpi": 150}), ("pdf", {})]:
            out_path = out_dir / f"{base_name}.{ext}"
            fig.savefig(out_path, bbox_inches="tight", **opts)
            print(f"Saved {out_path}")
        plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
