#!/usr/bin/env python3
"""
Generate 5x5 (25-exemplar) montages for animal categories from the full BV crop directory.
Only uses crops whose (category, stem) appear in the CLIP-filtered embedding list (each
exemplar used at most once / only filtered stems). Exemplars are sampled to favor
different videos and subjects (one per subject+video when possible).

Usage:
  cd analysis/vss-2026
  python make_animal_category_montages.py
  python make_animal_category_montages.py --cropped-dir /path/to/crops --out-dir ./animal_montages
"""
from pathlib import Path
from typing import Optional
import argparse
import random
import re
from collections import defaultdict
from PIL import Image
import numpy as np

# Stem pattern: {category}_{confidence}_{subject_id}_{gcp_name}_processed_{frame_id}
_STEM_PATTERN = re.compile(r"^(.+?)_([\d.]+)_(\d+)_(.+?)_processed_(\d+)$")

SCRIPT_DIR = Path(__file__).resolve().parent
THRESHOLD = "0.28"
DEFAULT_CROPPED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class")
CLIP_FILTER_LIST = Path(
    "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/"
    f"clip_image_embeddings_filtered-by-clip-{THRESHOLD}_exclude-people_exclude-subject-00270001.txt"
)

# Animal categories to montage (5x5 = 25 exemplars each when available)
DEFAULT_ANIMAL_CATEGORIES = [
    "zebra", "giraffe", "bird", "fish", "cat", "dog", "bear", "elephant",
    "horse", "lion", "tiger", "cow", "sheep", "pig", "duck", "rabbit",
    "mouse", "monkey", "alligator", "turtle", "butterfly", "bee",
]

N_EXEMPLARS = 25  # 5x5
N_COLS = 5
CELL_SIZE = (128, 128)


def get_category_crop_dir(root_dir: Path, cat_name: str) -> Optional[Path]:
    """Return path to category subfolder (case-insensitive), or None if missing."""
    root_dir = Path(root_dir)
    cat_lower = cat_name.strip().lower()
    direct = root_dir / cat_name
    if direct.exists() and direct.is_dir():
        return direct
    if root_dir.exists():
        for p in root_dir.iterdir():
            if p.is_dir() and p.name.lower() == cat_lower:
                return p
    return None


def _source_key_from_stem(stem: str) -> tuple[str, str]:
    """Return (subject_id, gcp_name) for diversification, or (stem, stem) if unparsed."""
    m = _STEM_PATTERN.match(stem)
    if m:
        return (m.group(3), m.group(4))  # subject_id, gcp_name (video)
    return (stem, stem)


def _sample_paths_diversified(paths: list[Path], n: int, seed: Optional[int] = None) -> list[Path]:
    """
    Sample up to n paths, preferring one per (subject, video) to spread across
    different videos/families. Groups by (subject_id, gcp_name) from stem.
    """
    if seed is not None:
        random.seed(seed)
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
        chosen.extend(remaining[: n - len(chosen)])
    random.shuffle(chosen)
    return chosen


def load_filtered_stems_for_categories(filter_list_path: Path, categories: list[str]) -> set[tuple[str, str]]:
    """
    Stream the CLIP-filtered list (one .npy path per line) and return set of (category_lower, stem)
    only for the given categories.
    """
    categories_set = {c.strip().lower() for c in categories}
    allowed = set()
    if not filter_list_path.exists():
        return allowed
    with open(filter_list_path, "r") as f:
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


def make_montage(paths: list[Path], cell_size: tuple[int, int], n_cols: int):
    """Tile images into a single PIL Image. Fills missing cells with gray."""
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


def main():
    parser = argparse.ArgumentParser(
        description="5x5 animal category montages from full BV crops (filtered list only)"
    )
    parser.add_argument(
        "--cropped-dir",
        type=Path,
        default=DEFAULT_CROPPED_DIR,
        help="Root dir of all cropped images: cropped_dir/category/*.jpg",
    )
    parser.add_argument(
        "--filter-list",
        type=Path,
        default=CLIP_FILTER_LIST,
        help="CLIP-filtered embedding paths (one per line); only (category, stem) in this list are used",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {SCRIPT_DIR / f'animal_category_montages_filtered-{THRESHOLD}'})",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Animal categories to montage (default: zebra, giraffe, bird, fish, cat, dog, bear, ...)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_EXEMPLARS,
        help=f"Max exemplars per montage (default {N_EXEMPLARS}, i.e. 5x5)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        nargs=2,
        default=list(CELL_SIZE),
        metavar=("W", "H"),
        help="Cell size in pixels",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Use all crops in each category (ignore filter list)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for diversifying across videos/subjects (default 42)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else SCRIPT_DIR / f"animal_category_montages_filtered-{THRESHOLD}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_size = tuple(args.cell_size)
    n_want = args.n
    categories = args.categories if args.categories is not None else DEFAULT_ANIMAL_CATEGORIES

    if not args.cropped_dir.exists():
        print(f"Error: cropped dir not found: {args.cropped_dir}")
        return 1

    # Load filtered stems only for our categories (so we only use crops that appear in the filter list)
    allowed_stems = None
    if not args.no_filter and args.filter_list.exists():
        print(f"Loading filtered stems for {len(categories)} categories from {args.filter_list}...")
        allowed_stems = load_filtered_stems_for_categories(args.filter_list, categories)
        print(f"  Found {len(allowed_stems)} (category, stem) pairs in filter list.")
    elif not args.no_filter and args.filter_list:
        print(f"Warning: filter list not found: {args.filter_list}. Use --no-filter to use all crops.")
        return 1

    n_cols = int(np.ceil(np.sqrt(n_want)))
    if n_cols * n_cols != n_want:
        n_cols = 5
        n_rows = (n_want + n_cols - 1) // n_cols
    else:
        n_rows = n_cols

    for cat in categories:
        cat_dir = get_category_crop_dir(args.cropped_dir, cat)
        if cat_dir is None:
            print(f"  Skip {cat}: no folder in cropped dir")
            continue

        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(cat_dir.glob(ext))
        cat_lower = cat.strip().lower()
        if allowed_stems is not None:
            paths = [p for p in paths if (cat_lower, p.stem) in allowed_stems]
        paths = sorted(paths, key=lambda p: p.name)
        paths = _sample_paths_diversified(paths, n_want, seed=args.seed)

        if not paths:
            print(f"  Skip {cat}: no images (filtered count 0)")
            continue

        montage = make_montage(paths, cell_size, n_cols)
        if montage is None:
            print(f"  Skip {cat}: failed to build montage")
            continue

        out_path = out_dir / f"{cat}_montage_5x5.png"
        montage.save(out_path)
        print(f"  {cat}: saved {out_path.name} ({len(paths)} exemplars)")

    print(f"Montages saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
