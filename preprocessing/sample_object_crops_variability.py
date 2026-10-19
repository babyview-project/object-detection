#!/usr/bin/env python3
"""
Sample exemplars per category from cropped BV images for crowdsourcing.

Key constraints:
1. Only uses images whose (category, stem) appear in the CLIP-filtered embedding list
   text file (removes false alarm detections).
2. For each category in `data/things_bv_overlap_categories_exclude_zero_precisions.txt`,
   samples `N_PER_CATEGORY` exemplars (default 100).
3. Sampling explicitly maximizes diversity across subject and video
   (subject_id, gcp_name).
4. Assigns a shuffled `bucket_index` in {1,2,3,4} per category for pre-assignment
   into 4 participant buckets (25 crops each when 100 are available).

Outputs:
- A new CSV listing selected exemplars (category, path, and parsed metadata).
- A compact bucket assignment CSV (no participant IDs).
- Optionally copies selected images into a category subfolder structure.
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Collection, Optional

import pandas as pd

# Paths (relative to project root when run from repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CATEGORY_LIST_FILE = PROJECT_ROOT / "data" / "things_bv_overlap_categories_exclude_zero_precisions.txt"

DEFAULT_CROPPED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class")

# CLIP-filtered list: one embedding path per line ( .../category/filename.npy ); only these stems are sampled
DEFAULT_FILTER_LIST = Path(
    "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/"
    "clip_image_embeddings_filtered-by-clip-0.27_exclude-people_exclude-subject-00270001.txt"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "annotation" / "sampled_object_crops_100_diverse_families"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "annotation" / "sampled_object_crops_100_diverse_families.csv"
DEFAULT_PARTICIPANT_CSV = PROJECT_ROOT / "annotation" / "sampled_object_crops_100_bucket_assignments.csv"
DEFAULT_SEED_FILE = PROJECT_ROOT / "annotation" / "sampled_object_crops_100_diverse_families_seed.txt"
DEFAULT_EXAMPLE_OUTPUT_CSV = PROJECT_ROOT / "annotation" / "example_page_correct_detections_table.csv"
DEFAULT_EXAMPLE_OUTPUT_DIR = PROJECT_ROOT / "annotation" / "example_page_correct_detections_table_images"

# Reproducibility
DEFAULT_RANDOM_SEED = 42
DEFAULT_N_PER_CATEGORY = 100

# Image extensions to consider
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Stem pattern from other preprocessing/analysis scripts:
# {category}_{confidence}_{subject_id}_{gcp_name}_processed_{frame_id}
_STEM_PATTERN = re.compile(r"^(.+?)_([\d.]+)_(\d+)_(.+?)_processed_(\d+)$")


def parse_stem(stem: str) -> tuple[str, str, str, str]:
    """
    Return (subject_id, family, gcp_name, confidence_str) for metadata.

    If parsing fails, falls back to (stem, 'unknown', stem, 'unknown').
    """
    m = _STEM_PATTERN.match(stem)
    if not m:
        return stem, "unknown", stem, "unknown"
    # group(3) is subject_id; group(4) is gcp_name.
    # Per your annotation setup, family ID is the full subject_id (e.g., 00510002).
    subject_id = m.group(3)
    family = subject_id if subject_id else "unknown"
    gcp_name = m.group(4)
    confidence_str = m.group(2)
    return subject_id, family, gcp_name, confidence_str


def get_category_crop_dir(cropped_dir: Path, cat_name: str) -> Optional[Path]:
    """Return path to category subfolder (case-insensitive), or None if missing."""
    cat_lower = cat_name.strip().lower()
    direct = cropped_dir / cat_name
    if direct.exists() and direct.is_dir():
        return direct
    for p in cropped_dir.iterdir():
        if p.is_dir() and p.name.lower() == cat_lower:
            return p
    return None


def load_filtered_stems(
    filter_list_path: Path,
    categories_lower: set[str],
) -> set[tuple[str, str]]:
    """
    Stream the CLIP-filtered embedding list and return set of (category_lower, stem).
    Each line is a path like .../category/filename.npy; category and stem (no .npy) are extracted.
    """
    allowed: set[tuple[str, str]] = set()
    with open(filter_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.suffix.lower() == ".npy":
                stem = p.stem
            else:
                stem = p.name
            category = p.parent.name.strip().lower()
            if category in categories_lower:
                allowed.add((category, stem))
    return allowed


def sample_paths_diverse_subject_video(paths: list[Path], n: int, seed: int) -> list[Path]:
    """
    Sample up to n paths while maximizing subject + video diversity.

    Strategy:
    1) Group by subject_id, then by gcp_name (video)
    2) Round-robin across subjects to avoid overconcentration
    3) Within each subject, rotate across videos when possible
    4) If still short, fill from remaining randomized paths
    """
    if n <= 0 or not paths:
        return []

    rng = random.Random(seed)
    if len(paths) <= n:
        out = list(paths)
        rng.shuffle(out)
        return out

    # subject_id -> gcp_name -> list[Path]
    subject_groups: dict[str, dict[str, list[Path]]] = {}
    for p in paths:
        subject_id, _family, gcp_name, _confidence = parse_stem(p.stem)
        subject_groups.setdefault(subject_id, {}).setdefault(gcp_name, []).append(p)

    subject_order = list(subject_groups.keys())
    rng.shuffle(subject_order)

    # Shuffle each per-video pool and video order within family
    subject_video_order: dict[str, list[str]] = {}
    for subject_id in subject_order:
        videos = list(subject_groups[subject_id].keys())
        rng.shuffle(videos)
        subject_video_order[subject_id] = videos
        for v in videos:
            rng.shuffle(subject_groups[subject_id][v])

    chosen: list[Path] = []
    chosen_set: set[Path] = set()

    progressed = True
    while len(chosen) < n and progressed:
        progressed = False
        for subject_id in subject_order:
            if len(chosen) >= n:
                break
            videos = subject_video_order[subject_id]
            if not videos:
                continue

            # Pick next video with available crops (cyclic rotation)
            picked = None
            for _ in range(len(videos)):
                video = videos.pop(0)
                pool = subject_groups[subject_id][video]
                if pool:
                    picked = pool.pop()
                    videos.append(video)  # rotate video to back
                    break
                videos.append(video)

            if picked is not None and picked not in chosen_set:
                chosen.append(picked)
                chosen_set.add(picked)
                progressed = True

    if len(chosen) >= n:
        return chosen[:n]

    leftovers = [p for p in paths if p not in chosen_set]
    rng.shuffle(leftovers)
    chosen.extend(leftovers[: n - len(chosen)])
    return chosen[:n]


def assign_shuffled_buckets(n_items: int, n_buckets: int, seed: int) -> list[int]:
    """
    Return bucket indices (1..n_buckets) in randomized order.
    If n_items is divisible by n_buckets (e.g., 100/4), buckets are balanced.
    """
    if n_items <= 0:
        return []
    base = n_items // n_buckets
    remainder = n_items % n_buckets
    labels: list[int] = []
    for b in range(1, n_buckets + 1):
        count = base + (1 if b <= remainder else 0)
        labels.extend([b] * count)
    rng = random.Random(seed)
    rng.shuffle(labels)
    return labels


def build_bucket_assignment_rows(
    sampled_df: pd.DataFrame,
    n_buckets: int,
    seed: int,
    attention_pool_by_category: dict[str, list[Path]],
) -> pd.DataFrame:
    """
    Build compact bucket-level assignments (no participant IDs, no repeats).
    Adds one real-photo OOD attention check per category page/trial and `bucket_trial_index`.
    The OOD image is sampled from a category different from the target category.
    """
    out_frames: list[pd.DataFrame] = []
    for bucket_idx in range(1, n_buckets + 1):
        bucket_df = sampled_df.loc[
            sampled_df["bucket_index"] == bucket_idx,
            ["bucket_index", "category", "path", "stem"],
        ].copy()
        bucket_df["trial_type"] = "regular"

        # Add one category-specific OOD check per category page/trial.
        ood_rows: list[dict[str, str | int]] = []
        rng = random.Random(seed + 10_000 + bucket_idx)
        categories_in_bucket = sorted(bucket_df["category"].astype(str).unique().tolist())
        for target_category in categories_in_bucket:
            candidate_source_categories = [
                cat
                for cat, paths in attention_pool_by_category.items()
                if cat != target_category.strip().lower() and paths
            ]
            if not candidate_source_categories:
                raise RuntimeError(
                    f"No attention-check source categories available for target category '{target_category}' "
                    "after excluding same-category images."
                )
            source_category = rng.choice(candidate_source_categories)
            p = rng.choice(attention_pool_by_category[source_category])
            ood_rows.append(
                {
                    "bucket_index": bucket_idx,
                    "category": target_category,
                    "path": str(p.resolve()),
                    "stem": p.stem,
                    "trial_type": "ood_attention_check_real_photo",
                    "attention_check_source_category": source_category,
                }
            )

        if ood_rows:
            bucket_df = pd.concat([bucket_df, pd.DataFrame(ood_rows)], ignore_index=True)

        # Randomized within-bucket ordering for downstream assignment/use.
        bucket_df = bucket_df.sample(frac=1, random_state=seed + bucket_idx).reset_index(drop=True)
        bucket_df["bucket_trial_index"] = bucket_df.index + 1
        out_frames.append(bucket_df)
    if not out_frames:
        return pd.DataFrame(
            columns=[
                "bucket_index",
                "category",
                "path",
                "stem",
                "trial_type",
                "attention_check_source_category",
                "bucket_trial_index",
            ]
        )
    return pd.concat(out_frames, ignore_index=True)


def collect_real_attention_pool_by_category(
    cropped_dir: Path,
    things_images_dir: Optional[Path],
    seed: int,
    source_mode: str,
    overlap_categories_lower: set[str],
) -> dict[str, list[Path]]:
    """
    Collect real-photo attention-check candidates grouped by category.
    Sources depend on `source_mode`:
    - babyview_and_things: Babyview + THINGS (legacy behavior)
    - things_nonoverlap: THINGS categories excluding overlap/Babyview categories
    - things_only: all THINGS categories
    - babyview_only: Babyview only
    """
    rng = random.Random(seed)
    out: dict[str, list[Path]] = {}

    def _add_from_root(root_dir: Path, recursive: bool, category_filter=None) -> None:
        if not root_dir.exists():
            raise FileNotFoundError(f"Attention-check source dir not found: {root_dir}")
        for category_dir in sorted(root_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            cat_lower = category_dir.name.strip().lower()
            if category_filter is not None and not category_filter(cat_lower):
                continue
            if recursive:
                paths = [
                    p
                    for ext in IMAGE_EXTENSIONS
                    for p in category_dir.rglob(f"*{ext}")
                ]
            else:
                paths = [
                    p
                    for ext in IMAGE_EXTENSIONS
                    for p in category_dir.glob(f"*{ext}")
                ]
            if not paths:
                continue
            rng.shuffle(paths)
            out.setdefault(cat_lower, []).extend(paths)

    def _add_from_flat_things_root(root_dir: Path, category_filter=None) -> None:
        """
        Handle THINGS layout where files live directly under root:
        root_dir/<category>.jpg
        """
        if not root_dir.exists():
            raise FileNotFoundError(f"Attention-check source dir not found: {root_dir}")
        for ext in IMAGE_EXTENSIONS:
            for p in root_dir.glob(f"*{ext}"):
                if not p.is_file():
                    continue
                cat_lower = p.stem.strip().lower()
                if not cat_lower:
                    continue
                if category_filter is not None and not category_filter(cat_lower):
                    continue
                out.setdefault(cat_lower, []).append(p)

    if source_mode in {"babyview_and_things", "babyview_only"}:
        _add_from_root(cropped_dir, recursive=False)
    if source_mode in {"babyview_and_things", "things_nonoverlap", "things_only"}:
        if things_images_dir is None:
            raise ValueError(
                f"attention source mode '{source_mode}' requires --things-images-dir"
            )
        # THINGS can be organized either as category subfolders or as flat files.
        has_category_dirs = any(p.is_dir() for p in things_images_dir.iterdir())
        if source_mode == "things_nonoverlap":
            category_filter = lambda cat: cat not in overlap_categories_lower
            if has_category_dirs:
                _add_from_root(
                    things_images_dir,
                    recursive=True,
                    category_filter=category_filter,
                )
            else:
                _add_from_flat_things_root(
                    things_images_dir,
                    category_filter=category_filter,
                )
        else:
            if has_category_dirs:
                _add_from_root(things_images_dir, recursive=True)
            else:
                _add_from_flat_things_root(things_images_dir)

    for cat in list(out.keys()):
        rng.shuffle(out[cat])

    if not out:
        raise RuntimeError("No real-photo attention-check candidates found in Babyview/THINGS sources.")
    return out


def collect_filtered_crop_paths_for_category(
    cropped_dir: Path,
    filter_list_path: Path,
    category: str,
) -> list[Path]:
    """
    All on-disk crops for ``category`` whose (category, stem) appears in the CLIP filter list.
    """
    category_lower = category.strip().lower()
    allowed = load_filtered_stems(filter_list_path, {category_lower})
    cat_dir = get_category_crop_dir(cropped_dir, category)
    if not cat_dir:
        return []
    paths: list[Path] = []
    for cat, stem in allowed:
        if cat != category_lower:
            continue
        for ext in IMAGE_EXTENSIONS:
            p = cat_dir / f"{stem}{ext}"
            if p.is_file():
                paths.append(p)
                break
    return paths


def write_example_page_correct_detections(
    sampled_df: Optional[pd.DataFrame],
    category: str,
    n_examples: int,
    output_csv: Path,
    copy_images: bool,
    output_dir: Path,
    *,
    excluded_stems: Optional[Collection[str]] = None,
    cropped_dir: Optional[Path] = None,
    filter_list_path: Optional[Path] = None,
) -> int:
    """
    Write high-confidence Babyview examples for the annotation example page.

    If ``cropped_dir`` and ``filter_list_path`` are set, candidates come from the full
    CLIP-filtered pool (not only ``sampled_df``), so exemplars can be disjoint from
    trial crops when ``excluded_stems`` lists trial stems.

    When those paths are omitted, rows are taken only from ``sampled_df`` for the category
    (legacy behavior), still honoring ``excluded_stems`` when provided.
    """
    category_lower = category.strip().lower()
    exclude = frozenset(str(s) for s in excluded_stems) if excluded_stems else frozenset()

    if cropped_dir is not None and filter_list_path is not None:
        pool_paths = collect_filtered_crop_paths_for_category(
            cropped_dir, filter_list_path, category_lower
        )
        rows: list[dict[str, str]] = []
        for p in pool_paths:
            if p.stem in exclude:
                continue
            subject_id, _family, gcp_name, confidence_str = parse_stem(p.stem)
            rows.append(
                {
                    "category": category_lower,
                    "path": str(p),
                    "stem": p.stem,
                    "confidence_str": confidence_str,
                    "subject_id": subject_id,
                    "gcp_name": gcp_name,
                }
            )
        cat_df = pd.DataFrame(rows)
    else:
        if sampled_df is None:
            raise RuntimeError(
                "sampled_df is required unless cropped_dir and filter_list_path are both set."
            )
        cat_df = sampled_df.loc[
            sampled_df["category"].astype(str).str.strip().str.lower() == category_lower
        ].copy()
        if cat_df.empty:
            raise RuntimeError(
                f"No sampled rows found for example category '{category}'. "
                "Try a category present in sampled output, or pass cropped_dir + filter_list_path."
            )
        if exclude:
            cat_df = cat_df.loc[~cat_df["stem"].astype(str).isin(exclude)].copy()

    if cat_df.empty:
        raise RuntimeError(
            f"No candidate crops for example category '{category}' after exclusions. "
            "Check excluded_stems and filter list."
        )

    cat_df["confidence_num"] = pd.to_numeric(cat_df["confidence_str"], errors="coerce")
    cat_df = cat_df.sort_values(
        by=["confidence_num", "subject_id", "gcp_name", "stem"],
        ascending=[False, True, True, True],
    ).head(max(0, n_examples))
    if len(cat_df) < n_examples:
        raise RuntimeError(
            f"Only {len(cat_df)} example crops available for '{category}' "
            f"after exclusions (need {n_examples})."
        )

    cat_df["trial_type"] = "example_page_correct_detection"
    cat_df["expected_correct_detection"] = 1
    cat_df["expected_correct_cdi"] = 1
    cat_df["example_index"] = range(1, len(cat_df) + 1)
    out = cat_df[
        [
            "example_index",
            "category",
            "path",
            "stem",
            "confidence_str",
            "subject_id",
            "gcp_name",
            "trial_type",
            "expected_correct_detection",
            "expected_correct_cdi",
        ]
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    if copy_images:
        output_dir.mkdir(parents=True, exist_ok=True)
        for src in cat_df["path"].astype(str).tolist():
            src_path = Path(src)
            if not src_path.exists():
                continue
            shutil.copy2(src_path, output_dir / src_path.name)
    return len(out)


def validate_sampled_in_filter_list(sampled_rows: list[dict[str, str]], allowed_stems: set[tuple[str, str]]) -> None:
    """Raise if any sampled (category, stem) is missing from the CLIP-filtered list."""
    bad: list[tuple[str, str]] = []
    for row in sampled_rows:
        cat_lower = str(row["category"]).strip().lower()
        stem = str(row["stem"])
        if (cat_lower, stem) not in allowed_stems:
            bad.append((cat_lower, stem))
    if bad:
        preview = ", ".join([f"({c}, {s})" for c, s in bad[:5]])
        raise RuntimeError(
            f"Found {len(bad)} sampled crops missing from CLIP-filtered list. First few: {preview}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Sample diverse exemplars per category from BV crops (CLIP-filtered only)"
    )
    parser.add_argument(
        "--cropped-dir",
        type=Path,
        default=DEFAULT_CROPPED_DIR,
        help="Root dir of cropped images: cropped_dir/category/*.jpg",
    )
    parser.add_argument(
        "--filter-list",
        type=Path,
        default=DEFAULT_FILTER_LIST,
        help="CLIP-filtered embedding paths (one per line); only (category, stem) in this list are used",
    )
    parser.add_argument(
        "--category-list-file",
        type=Path,
        default=DEFAULT_CATEGORY_LIST_FILE,
        help="Text file with categories to sample (one per line).",
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=DEFAULT_N_PER_CATEGORY,
        help="How many exemplars to sample per category (default 100, strict).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for sampling (default 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to copy sampled images when --copy-images is enabled.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV output path.",
    )
    parser.add_argument(
        "--participant-csv",
        type=Path,
        default=DEFAULT_PARTICIPANT_CSV,
        help="Bucket assignment CSV path (compact; no participant IDs).",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=DEFAULT_SEED_FILE,
        help="Where to save the seed value.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="If set, copy sampled images into --output-dir/category_name/.",
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=4,
        help="Number of preassignment buckets (default 4).",
    )
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Allow categories with fewer than --n-per-category filtered crops (default: disabled).",
    )
    parser.add_argument(
        "--things-images-dir",
        type=Path,
        default=None,
        help="Optional THINGS images root dir with category subfolders; used as additional real-photo attention-check source.",
    )
    parser.add_argument(
        "--n-attention-checks-per-category",
        type=int,
        default=1,
        help="Number of real-photo OOD attention checks per category page/trial (default 1, currently only 1 supported).",
    )
    parser.add_argument(
        "--attention-source-mode",
        type=str,
        default="babyview_and_things",
        choices=["babyview_and_things", "things_nonoverlap", "things_only", "babyview_only"],
        help=(
            "Source pool for real-photo attention checks. "
            "Use 'things_nonoverlap' to sample only THINGS categories that do not overlap Babyview categories."
        ),
    )
    parser.add_argument(
        "--example-category",
        type=str,
        default="table",
        help="Category for Babyview correct-detection examples CSV (set empty to disable).",
    )
    parser.add_argument(
        "--n-example-correct-detections",
        type=int,
        default=12,
        help="Number of high-confidence correct detections to export for example page.",
    )
    parser.add_argument(
        "--example-output-csv",
        type=Path,
        default=DEFAULT_EXAMPLE_OUTPUT_CSV,
        help="Output CSV path for example-page correct detections.",
    )
    parser.add_argument(
        "--copy-example-images",
        action="store_true",
        help="If set, copy example-page correct-detection images into --example-output-dir.",
    )
    parser.add_argument(
        "--example-output-dir",
        type=Path,
        default=DEFAULT_EXAMPLE_OUTPUT_DIR,
        help="Output dir for copied example-page correct-detection images.",
    )
    args = parser.parse_args()

    cropped_dir: Path = args.cropped_dir
    filter_list_path: Path = args.filter_list
    category_list_file: Path = args.category_list_file

    if not category_list_file.exists():
        raise FileNotFoundError(f"Category list file not found: {category_list_file}")
    if not filter_list_path.exists():
        raise FileNotFoundError(f"Filter list not found: {filter_list_path}")
    if not cropped_dir.exists():
        raise FileNotFoundError(f"Cropped dir not found: {cropped_dir}")
    if args.n_attention_checks_per_category != 1:
        raise ValueError("Only 1 attention check per category is currently supported. Use --n-attention-checks-per-category 1.")
    if args.things_images_dir is not None and not args.things_images_dir.exists():
        raise FileNotFoundError(f"THINGS images dir not found: {args.things_images_dir}")

    categories: list[str] = []
    with open(category_list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            categories.append(line)
    categories_lower = {c.strip().lower() for c in categories}

    print(f"Sampling {args.n_per_category} exemplars per category from:")
    print(f"  categories: {len(categories)} from {category_list_file}")
    print(f"  cropped dir: {cropped_dir}")
    print(f"  filter list: {filter_list_path}")

    print("Loading filtered stems (category, stem) pairs...")
    allowed_stems = load_filtered_stems(filter_list_path, categories_lower)
    print(f"  Loaded {len(allowed_stems)} (category, stem) pairs from filter list (filtered to requested categories)")

    args.seed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.seed_file, "w") as f:
        f.write(str(args.seed))

    if args.copy_images:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    shortages: list[tuple[str, int]] = []
    for cat in categories:
        cat_dir = get_category_crop_dir(cropped_dir, cat)
        if cat_dir is None:
            print(f"  Skip {cat}: no folder in cropped dir")
            continue

        all_paths: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            all_paths.extend(cat_dir.glob(f"*{ext}"))

        cat_lower = cat.strip().lower()
        filtered_paths = [p for p in all_paths if (cat_lower, p.stem) in allowed_stems]
        filtered_paths = sorted(filtered_paths, key=lambda p: p.name)

        if not filtered_paths:
            print(f"  Skip {cat}: no filtered images (total in folder: {len(all_paths)})")
            continue

        if len(filtered_paths) < args.n_per_category:
            if not args.allow_fewer:
                shortages.append((cat, len(filtered_paths)))
                print(
                    f"  Insufficient {cat}: filtered {len(filtered_paths)} < required {args.n_per_category}"
                )
                continue
            n_sample = len(filtered_paths)
        else:
            n_sample = args.n_per_category
        sampled = sample_paths_diverse_subject_video(filtered_paths, n_sample, seed=args.seed)
        bucket_indices = assign_shuffled_buckets(
            len(sampled), n_buckets=args.n_buckets, seed=args.seed + len(cat)
        )

        if args.copy_images:
            out_cat_dir = args.output_dir / cat
            out_cat_dir.mkdir(parents=True, exist_ok=True)

        for src, bucket_index in zip(sampled, bucket_indices):
            subject_id, family, gcp_name, confidence_str = parse_stem(src.stem)

            if args.copy_images:
                dst = out_cat_dir / src.name
                shutil.copy2(src, dst)

            rows.append(
                {
                    "category": cat,
                    "path": str(src.resolve()),
                    "stem": src.stem,
                    "family": family,
                    "subject_id": subject_id,
                    "gcp_name": gcp_name,
                    "confidence_str": confidence_str,
                    "bucket_index": bucket_index,
                }
            )

        unique_subjects = len({parse_stem(p.stem)[0] for p in sampled})
        unique_videos = len({parse_stem(p.stem)[2] for p in sampled})
        print(
            f"  {cat}: sampled {n_sample} (subjects {unique_subjects}, videos {unique_videos}, "
            f"filtered {len(filtered_paths)}, total in folder {len(all_paths)})"
        )

    if not rows:
        raise RuntimeError("No rows were sampled. Check that filter-list and cropped-dir paths are correct.")
    if shortages and not args.allow_fewer:
        detail = ", ".join([f"{c}={n}" for c, n in shortages[:10]])
        if len(shortages) > 10:
            detail += f", ... (+{len(shortages) - 10} more)"
        raise RuntimeError(
            "Strict sampling failed: some categories do not have enough CLIP-filtered crops "
            f"for {args.n_per_category}/category. Details: {detail}. "
            "Use --allow-fewer to proceed anyway."
        )

    validate_sampled_in_filter_list(rows, allowed_stems)

    df = pd.DataFrame(
        rows,
        columns=[
            "category",
            "path",
            "stem",
            "family",
            "subject_id",
            "gcp_name",
            "confidence_str",
            "bucket_index",
        ],
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    attention_pool_by_category = collect_real_attention_pool_by_category(
        cropped_dir=cropped_dir,
        things_images_dir=args.things_images_dir,
        seed=args.seed,
        source_mode=args.attention_source_mode,
        overlap_categories_lower=categories_lower,
    )
    participants_df = build_bucket_assignment_rows(
        sampled_df=df,
        n_buckets=args.n_buckets,
        seed=args.seed,
        attention_pool_by_category=attention_pool_by_category,
    )
    args.participant_csv.parent.mkdir(parents=True, exist_ok=True)
    participants_df.to_csv(args.participant_csv, index=False)

    print(f"Wrote {len(df)} rows to {args.output_csv}")
    print(
        f"Wrote {len(participants_df)} bucket-assignment rows to {args.participant_csv} "
        f"({args.n_buckets} buckets total; no participant duplication)"
    )
    print("Included 1 real-photo attention check per category page/trial.")
    print(
        "Attention checks are sampled from source mode "
        f"'{args.attention_source_mode}' and always from categories different from the target category."
    )
    print(f"Attention-check source category pool size: {len(attention_pool_by_category)}")
    example_category = str(args.example_category).strip()
    if example_category:
        trial_stems = set(df["stem"].astype(str))
        n_written = write_example_page_correct_detections(
            sampled_df=df,
            category=example_category,
            n_examples=args.n_example_correct_detections,
            output_csv=args.example_output_csv,
            copy_images=args.copy_example_images,
            output_dir=args.example_output_dir,
            excluded_stems=trial_stems,
            cropped_dir=cropped_dir,
            filter_list_path=filter_list_path,
        )
        print(
            f"Wrote example-page correct detections for '{example_category}' "
            f"to {args.example_output_csv}"
        )
        if args.copy_example_images:
            print(
                f"Copied {n_written} example images to {args.example_output_dir}"
            )
    print(f"Seed saved to {args.seed_file}")


if __name__ == "__main__":
    main()
