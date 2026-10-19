#!/usr/bin/env python3
"""
Build static assets for a Brady-style perceptual free-sort experiment using BabyView crops.

Pipeline (matches analysis conventions in analysis/manuscript-2026/01_long_tailed_distribution.ipynb; optional CLI: build_long_tailed_included_categories.py):
- Keep categories in included_categories.txt whose per-class precision > threshold.
- Keep exemplars with per-file precision > threshold for those classes.
- Require crops on disk under --cropped-dir/<category>/.
- Optionally intersect with CLIP-filtered (category, stem) pairs (same as sample_object_crops_variability.py).

For each eligible category, samples up to --n-per-category crops (default 100) with diverse
(subject, video) selection, or all available when fewer and --allow-fewer is set.
Assigns shuffled bucket_index in 1..N_BUCKETS (default 4). Each stream lists one trial per
category: up to --exemplars-per-trial images from that category in that bucket (subsampled if
the bucket holds more).

Writes:
- variability_exp/bv_validated_crops/media/<category>/<filename>  (copy or symlink)
- variability_exp/bv_validated_crops/stream{b}_images.csv  (b = 1..N_BUCKETS)
- variability_exp/bv_validated_crops/practice_images.csv
- variability_exp/bv_validated_crops/build_manifest.json

Stream CSV: header ``image_path`` then paths ``category/filename.jpg``. Each category appears
as one contiguous block forming a single trial; block length is
``min(exemplars_per_trial, images in that bucket for that category)`` (default cap 20).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = PROJECT_ROOT / "variability_exp" / "bv_validated_crops"
DEFAULT_INCLUDED_CATEGORIES = PROJECT_ROOT / "data" / "included_categories.txt"
DEFAULT_PER_CLASS_CSV = PROJECT_ROOT / "annotation" / "per_class_validation_data.csv"
DEFAULT_PER_FILE_CSV = PROJECT_ROOT / "annotation" / "per_file_precision_data.csv"
DEFAULT_CROPPED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _load_socv_module():
    path = PROJECT_ROOT / "preprocessing" / "sample_object_crops_variability.py"
    spec = importlib.util.spec_from_file_location("sample_object_crops_variability", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_category_list(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def load_valid_classes(per_class_csv: Path, threshold: float) -> set[str]:
    df = pd.read_csv(per_class_csv, usecols=["class", "precision"])
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
    return set(df.loc[df["precision"] > threshold, "class"])


def validated_stems_by_category(
    per_file_csv: Path,
    valid_classes: set[str],
    threshold: float,
    categories_filter: frozenset[str],
) -> dict[str, set[str]]:
    df = pd.read_csv(per_file_csv, usecols=["filename", "class", "precision"])
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
    df = df[(df["precision"] > threshold) & (df["class"].isin(valid_classes))].copy()
    df = df[df["class"].isin(categories_filter)]
    stems: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        base = Path(str(row["filename"]).strip()).name
        stem = Path(base).stem
        stems[row["class"]].add(stem)
    return dict(stems)


def resolve_crop_paths(
    socv,
    cropped_dir: Path,
    category: str,
    stems: set[str],
    clip_allowed: Optional[set[tuple[str, str]]],
) -> list[Path]:
    cat_lower = category.strip().lower()
    cat_dir = socv.get_category_crop_dir(cropped_dir, category)
    if cat_dir is None:
        return []
    paths: list[Path] = []
    for stem in stems:
        for ext in IMAGE_EXTENSIONS:
            p = cat_dir / f"{stem}{ext}"
            if p.is_file():
                if clip_allowed is not None and (cat_lower, stem) not in clip_allowed:
                    break
                paths.append(p)
                break
    return paths


def install_media_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(mode)


def write_stream_csv(paths_in_order: list[tuple[str, Path]], out_csv: Path) -> dict:
    lines = ["image_path"]
    for cat, src in paths_in_order:
        lines.append(f"{cat}/{src.name}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines) + "\n")
    return {"csv": str(out_csv), "n_images": len(lines) - 1}


def stable_cat_seed(global_seed: int, cat: str) -> int:
    return global_seed + sum(ord(c) for c in cat) % 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cropped-dir", type=Path, default=DEFAULT_CROPPED_DIR)
    parser.add_argument("--included-categories", type=Path, default=DEFAULT_INCLUDED_CATEGORIES)
    parser.add_argument("--per-class-csv", type=Path, default=DEFAULT_PER_CLASS_CSV)
    parser.add_argument("--per-file-csv", type=Path, default=DEFAULT_PER_FILE_CSV)
    parser.add_argument("--precision-threshold", type=float, default=0.6)
    parser.add_argument("--clip-filter-list", type=Path, default=None, help="Optional; intersect with CLIP-filtered stems")
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=100,
        help="Max exemplars sampled per category (diverse); min(pool, this value).",
    )
    parser.add_argument("--n-buckets", type=int, default=4)
    parser.add_argument(
        "--exemplars-per-trial",
        type=int,
        default=20,
        help="Cap on images shown per trial; if bucket has fewer, all are shown.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--media-mode", choices=("copy", "symlink"), default="symlink")
    parser.add_argument(
        "--no-allow-fewer",
        action="store_true",
        help="Skip categories with fewer than --n-per-category validated crops (default: include all).",
    )
    parser.add_argument("--practice-categories", type=int, default=2)
    parser.add_argument("--practice-exemplars", type=int, default=12)
    parser.add_argument("--max-categories", type=int, default=None, help="Debug: cap categories")
    args = parser.parse_args()
    allow_fewer = not args.no_allow_fewer

    socv = _load_socv_module()
    included = [c.strip().lower() for c in load_category_list(args.included_categories)]
    if args.max_categories:
        included = included[: args.max_categories]

    valid_classes = load_valid_classes(args.per_class_csv, args.precision_threshold)
    eligible_cats = [c for c in included if c in valid_classes]
    stems_by_cat = validated_stems_by_category(
        args.per_file_csv,
        valid_classes,
        args.precision_threshold,
        frozenset(eligible_cats),
    )

    clip_allowed: Optional[set[tuple[str, str]]] = None
    if args.clip_filter_list is not None:
        if not args.clip_filter_list.exists():
            raise SystemExit(f"CLIP filter list not found: {args.clip_filter_list}")
        clip_allowed = socv.load_filtered_stems(args.clip_filter_list, set(eligible_cats))
        print(f"CLIP filter: {len(clip_allowed)} (category, stem) pairs")

    rows: list[dict] = []
    shortages: list[tuple[str, int]] = []

    for cat in eligible_cats:
        stems = stems_by_cat.get(cat, set())
        paths = resolve_crop_paths(socv, args.cropped_dir, cat, stems, clip_allowed)
        paths = sorted(paths, key=lambda p: p.name)
        if not paths:
            print(f"  skip {cat}: no on-disk crops for validated stems")
            continue

        cap = min(args.n_per_category, len(paths))
        if len(paths) < args.n_per_category and not allow_fewer:
            shortages.append((cat, len(paths)))
            print(f"  skip {cat}: {len(paths)} validated crops < cap {args.n_per_category}")
            continue

        sampled = socv.sample_paths_diverse_subject_video(paths, cap, seed=stable_cat_seed(args.seed, cat))
        bucket_indices = socv.assign_shuffled_buckets(
            len(sampled), n_buckets=args.n_buckets, seed=args.seed + len(cat) * 31
        )
        for src, bi in zip(sampled, bucket_indices):
            subject_id, _family, gcp_name, confidence_str = socv.parse_stem(src.stem)
            rows.append(
                {
                    "category": cat,
                    "path": str(src.resolve()),
                    "stem": src.stem,
                    "bucket_index": int(bi),
                    "subject_id": subject_id,
                    "gcp_name": gcp_name,
                    "confidence_str": confidence_str,
                }
            )
        print(f"  {cat}: sampled {len(sampled)} / {len(paths)} validated on-disk crops")

    if not rows:
        raise SystemExit("No rows sampled; check paths and filters.")

    if shortages and not allow_fewer:
        prev = ", ".join(f"{c}={n}" for c, n in shortages[:15])
        raise SystemExit(f"Strict mode failures: {prev}")

    df = pd.DataFrame(rows)
    args.out_root.mkdir(parents=True, exist_ok=True)
    media_root = args.out_root / "media"
    media_root.mkdir(parents=True, exist_ok=True)

    rel_by_src: dict[str, str] = {}
    for src_str in df["path"].unique():
        src = Path(src_str)
        dst = media_root / src.parent.name / src.name
        install_media_file(src, dst, args.media_mode)
        rel_by_src[src_str] = f"{src.parent.name}/{src.name}"

    df["rel_path"] = df["path"].map(rel_by_src)

    rng = random.Random(args.seed)
    manifest: dict = {"streams": [], "precision_threshold": args.precision_threshold, "seed": args.seed}

    for b in range(1, args.n_buckets + 1):
        sub = df.loc[df["bucket_index"] == b].copy()
        cats = sorted(sub["category"].unique().tolist())
        rng.shuffle(cats)
        block: list[tuple[str, Path]] = []
        for cat in cats:
            cat_df = sub.loc[sub["category"] == cat]
            plist = [Path(p) for p in cat_df["path"].tolist()]
            rng.shuffle(plist)
            take = plist[: args.exemplars_per_trial]
            for p in take:
                block.append((cat, p))

        out_csv = args.out_root / f"stream{b}_images.csv"
        info = write_stream_csv(block, out_csv)
        info["bucket"] = b
        info["n_trials"] = len(cats)
        manifest["streams"].append(info)

    practice_lines = ["trial_type,concept,image_path"]
    rng_p = random.Random(args.seed + 999)
    practice_candidates = [
        c
        for c in eligible_cats
        if len(resolve_crop_paths(socv, args.cropped_dir, c, stems_by_cat.get(c, set()), clip_allowed))
        >= args.practice_exemplars
    ]
    rng_p.shuffle(practice_candidates)
    practice_cats = practice_candidates[: args.practice_categories]
    for pc in practice_cats:
        p_stems = stems_by_cat.get(pc, set())
        pool = resolve_crop_paths(socv, args.cropped_dir, pc, p_stems, clip_allowed)
        sampled_p = socv.sample_paths_diverse_subject_video(
            pool, args.practice_exemplars, seed=stable_cat_seed(args.seed + 404, pc)
        )
        for src in sampled_p:
            dst = media_root / src.parent.name / src.name
            if not dst.exists() and not dst.is_symlink():
                install_media_file(src, dst, args.media_mode)
            rel = f"{src.parent.name}/{src.name}"
            practice_lines.append(f"perceptual,{pc},{rel}")

    practice_csv = args.out_root / "practice_images.csv"
    practice_csv.write_text("\n".join(practice_lines) + "\n")

    manifest.update(
        {
            "n_categories": int(df["category"].nunique()),
            "n_rows": int(len(df)),
            "exemplars_per_trial_cap": args.exemplars_per_trial,
            "n_buckets": args.n_buckets,
        }
    )
    manifest_path = args.out_root / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    print(f"Wrote practice CSV: {practice_csv} ({len(practice_lines) - 1} images)")
    print(f"Media root: {media_root} ({args.media_mode})")


if __name__ == "__main__":
    main()
