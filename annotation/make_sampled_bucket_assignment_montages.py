#!/usr/bin/env python3
"""
Build image montages from sampled bucket-assignment CSV for manual QC.

Writes two layouts under --out-dir:
  per_category/          — all unique crop paths for each category (across buckets)
  per_bucket_category/   — unique paths for each (bucket_index, category) pair

Usage:
  cd annotation && python strip_ood_attention_checks_bucket_csv.py   # build babyview-only CSV
  python make_sampled_bucket_assignment_montages.py
  python make_sampled_bucket_assignment_montages.py --csv sampled_object_crops_..._per_video_cap.csv --out-dir ./montages_all_trials
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def safe_filename_part(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()) or "unknown"


def ordered_unique(paths: list[str]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(Path(p))
    return out


def load_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            images.append(Image.open(p).convert("RGB"))
        except OSError:
            continue
    return images


def make_montage(images: list[Image.Image], n_cols: int, cell_size: tuple[int, int]) -> Image.Image | None:
    if not images:
        return None
    n = len(images)
    cw, ch = cell_size
    n_rows = (n + n_cols - 1) // n_cols
    out = np.zeros((n_rows * ch, n_cols * cw, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        row, col = idx // n_cols, idx % n_cols
        if img.size != (cw, ch):
            img = img.resize((cw, ch), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[:, :, :3]
        out[row * ch : row * ch + ch, col * cw : col * cw + cw, :] = arr
    return Image.fromarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Montages from sampled bucket assignment CSV")
    here = Path(__file__).resolve().parent
    ap.add_argument(
        "--csv",
        type=Path,
        default=here / "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_babyview_only.csv",
        help="Input CSV with bucket_index, category, path, trial_type (default: BabyView-only, no ood_* rows)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=here / "sampled_object_crops_100ex_8subj_montages_babyview_only",
        help="Output root (creates per_category/ and per_bucket_category/)",
    )
    ap.add_argument("--n-cols", type=int, default=10, help="Grid columns")
    ap.add_argument(
        "--cell-size",
        type=int,
        nargs=2,
        default=[128, 128],
        metavar=("W", "H"),
        help="Thumbnail size in pixels",
    )
    args = ap.parse_args()
    cell_size = (args.cell_size[0], args.cell_size[1])

    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}")
        return 1

    per_category: dict[str, list[str]] = defaultdict(list)
    per_bucket_cat: dict[tuple[int, str], list[str]] = defaultdict(list)

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"].strip()
            path = row["path"].strip()
            try:
                bi = int(row["bucket_index"])
            except (KeyError, ValueError):
                continue
            per_category[cat].append(path)
            per_bucket_cat[(bi, cat)].append(path)

    out_root = args.out_dir
    cat_dir = out_root / "per_category"
    bc_dir = out_root / "per_bucket_category"
    cat_dir.mkdir(parents=True, exist_ok=True)
    bc_dir.mkdir(parents=True, exist_ok=True)

    n_cat_ok = n_cat_skip = 0
    for cat, plist in sorted(per_category.items(), key=lambda x: x[0].lower()):
        paths = ordered_unique(plist)
        imgs = load_images(paths)
        montage = make_montage(imgs, args.n_cols, cell_size)
        if montage is None:
            n_cat_skip += 1
            print(f"  skip category {cat!r}: no readable images ({len(paths)} paths)")
            continue
        out_path = cat_dir / f"{safe_filename_part(cat)}.png"
        montage.save(out_path)
        n_cat_ok += 1
        if n_cat_ok <= 3 or n_cat_ok % 40 == 0:
            print(f"  category {cat!r}: {len(imgs)}/{len(paths)} images -> {out_path.name}")

    n_bc_ok = n_bc_skip = 0
    for (bi, cat), plist in sorted(per_bucket_cat.items(), key=lambda x: (x[0][0], x[0][1].lower())):
        paths = ordered_unique(plist)
        imgs = load_images(paths)
        montage = make_montage(imgs, args.n_cols, cell_size)
        if montage is None:
            n_bc_skip += 1
            continue
        out_path = bc_dir / f"bucket_{bi:03d}_{safe_filename_part(cat)}.png"
        montage.save(out_path)
        n_bc_ok += 1

    print(
        f"Done. per_category: {n_cat_ok} saved, {n_cat_skip} skipped. "
        f"per_bucket_category: {n_bc_ok} saved, {n_bc_skip} skipped.\nOutput: {out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
