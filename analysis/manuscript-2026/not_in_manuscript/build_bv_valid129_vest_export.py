#!/usr/bin/env python3
"""
Build VEST (vision-embedding-space-travelling) inputs for BabyView exemplars:
valid129 categories, CLIP-filter list + paired CLIP/DINO .npy on disk (same rules as
06_exemplar_set_zscore_embeddings.ipynb).

Writes:
  - exemplar_manifest_valid129.csv — every paired exemplar with resolved crop path
  - vest_data.csv — filename, x, y, z, category (+ optional numeric columns for VEST coloring)
  - images/<category>/<stem>.<ext> — symlinks to crops (default) or copies
  - vest_export_run.json — paths, counts, UMAP params

Environment (aligned with notebook 06):
  BV_EMBEDDINGS_BASE, BV_CLIP_EMBEDDINGS_DIR, BV_DINOV3_EMBEDDINGS_DIR,
  BV_CLIP_FILTER_LIST, BV_CLIP_FILTER_LIST_THRESHOLD,
  BV_CROP_PATH_PREFIX, BV_CROP_PATH_PREFIX_NEW,
  BV_CROPPED_DIR — root of yoloe_cdi_all_cropped_by_class (category subfolders, JPG/PNG)

Usage:
  pip install umap-learn  # once
  python build_bv_valid129_vest_export.py --out-dir ./vest_export_bv_valid129

Then:
  cd vest_export_bv_valid129 && vest vest_data.csv --image-path ./images

See https://github.com/ScaDS/vest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

SCRIPT_PATH = Path(__file__).resolve()
NOT_IN_MS_DIR = SCRIPT_PATH.parent
PREPRINT_DIR = NOT_IN_MS_DIR.parent
PROJECT_ROOT = PREPRINT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INCLUDED = DATA_DIR / "included_categories_valid129.txt"
DEFAULT_CROPPED_DIR = Path(
    os.environ.get(
        "BV_CROPPED_DIR",
        "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class",
    )
).expanduser()

_EMB_BASE = Path(
    os.environ.get(
        "BV_EMBEDDINGS_BASE",
        "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings",
    )
).expanduser()
DEFAULT_CLIP_EMB = Path(
    os.environ.get("BV_CLIP_EMBEDDINGS_DIR", str(_EMB_BASE / "clip_embeddings_new"))
).expanduser()
DEFAULT_DINO_EMB = Path(
    os.environ.get(
        "BV_DINOV3_EMBEDDINGS_DIR",
        str(_EMB_BASE / "facebook_dinov3-vitb16-pretrain-lvd1689m"),
    )
).expanduser()
DEFAULT_CLIP_FILTER_LIST = Path(
    os.environ.get(
        "BV_CLIP_FILTER_LIST",
        str(
            _EMB_BASE
            / f"clip_image_embeddings_filtered-by-clip-{os.environ.get('BV_CLIP_FILTER_LIST_THRESHOLD', '0.27').strip()}_exclude-people_exclude-subject-00270001.txt"
        ),
    )
).expanduser()

CROP_PREFIX = os.environ.get("BV_CROP_PATH_PREFIX", "").strip()
CROP_PREFIX_NEW = os.environ.get("BV_CROP_PATH_PREFIX_NEW", "").strip()


def remap_absolute_path(p: Path) -> Path:
    s = str(p)
    if CROP_PREFIX and CROP_PREFIX_NEW and s.startswith(CROP_PREFIX):
        return Path(CROP_PREFIX_NEW + s[len(CROP_PREFIX) :])
    return p


def resolve_category_subdir(embed_root: Path, cat: str) -> Path | None:
    direct = embed_root / cat
    if direct.is_dir():
        return direct
    for p in embed_root.iterdir():
        if p.is_dir() and p.name.lower() == cat.lower():
            return p
    return None


def get_category_crop_dir(cropped_dir: Path, cat_name: str) -> Path | None:
    cat_lower = cat_name.strip().lower()
    direct = cropped_dir / cat_name
    if direct.exists() and direct.is_dir():
        return direct
    for p in cropped_dir.iterdir():
        if p.is_dir() and p.name.lower() == cat_lower:
            return p
    return None


def resolve_crop_image_path(cropped_dir: Path, category: str, stem: str) -> tuple[Path | None, str]:
    cat_dir = get_category_crop_dir(cropped_dir, category)
    if cat_dir is None:
        return None, ""
    for ext in IMAGE_EXTENSIONS:
        p = cat_dir / f"{stem}{ext}"
        if p.is_file():
            return p, ext
    return None, ""


def load_included_categories(txt_path: Path) -> list[str]:
    return [line.strip().lower() for line in txt_path.read_text().splitlines() if line.strip()]


def load_normalized_npy_vector(path: Path) -> np.ndarray | None:
    p = remap_absolute_path(path)
    if not p.is_file():
        return None
    v = np.load(p, mmap_mode="r")
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def _load_clip_row(row: dict[str, str]) -> np.ndarray | None:
    return load_normalized_npy_vector(Path(row["clip_npy"]))


def _install_one_vest_image(
    row: dict[str, str],
    images_root: Path,
    mode: str,
) -> tuple[str, str] | None:
    cat = row["category"]
    src = Path(row["image_path"])
    if not src.is_file():
        return None
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    dst_name = f"{row['stem']}{ext}"
    rel = f"{cat}/{dst_name}"
    dst = images_root / cat / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)
    return (rel.replace("\\", "/"), cat)


def allocate_stratified_quotas(counts: Counter, max_total: int) -> dict[str, int]:
    """Largest-remainder quotas proportional to counts, capped per category, summing to max_total."""
    total = int(sum(counts.values()))
    if total == 0:
        return {}
    if max_total <= 0:
        return {c: 0 for c in counts}
    if total <= max_total:
        return dict(counts)
    cats = sorted(counts.keys())
    raw = {c: max_total * counts[c] / total for c in cats}
    floors = {c: min(counts[c], int(math.floor(raw[c]))) for c in cats}
    remainder = max_total - sum(floors.values())
    frac_order = sorted(cats, key=lambda c: raw[c] - math.floor(raw[c]), reverse=True)
    i = 0
    stalled = 0
    while remainder > 0 and stalled < len(frac_order) + 3:
        c = frac_order[i % len(frac_order)]
        if floors[c] < counts[c]:
            floors[c] += 1
            remainder -= 1
            stalled = 0
        else:
            stalled += 1
        i += 1
    return floors


def reservoir_add(buf: list, item: Any, k: int, seen_in_cat: int, rng: random.Random) -> None:
    """Reservoir sample of size k from the first `seen_in_cat` items in this category (uniform)."""
    if k <= 0:
        return
    if seen_in_cat <= k:
        buf.append(item)
        return
    j = rng.randrange(seen_in_cat)
    if j < k:
        buf[j] = item


def count_pass_filter_list(
    filter_list_path: Path,
    clip_embed_root: Path,
    dino_embed_root: Path,
    allowed_categories: set[str],
) -> Counter:
    """First pass: paired exemplars per category (image optional); dedupe (cat, fname)."""
    counts_image_ok = Counter()
    seen: dict[str, set[str]] = defaultdict(set)
    with filter_list_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.suffix.lower() != ".npy":
                continue
            fname = p.name
            cat = p.parent.name.strip().lower()
            if cat not in allowed_categories:
                continue
            key = fname.lower()
            if key in seen[cat]:
                continue
            seen[cat].add(key)
            cdir_c = resolve_category_subdir(clip_embed_root, cat)
            cdir_d = resolve_category_subdir(dino_embed_root, cat)
            if cdir_c is None or cdir_d is None:
                continue
            pc = remap_absolute_path(cdir_c / fname)
            pd_ = remap_absolute_path(cdir_d / fname)
            if not pc.is_file() or not pd_.is_file():
                continue
            img_path, _ = resolve_crop_image_path(DEFAULT_CROPPED_DIR, cat, pc.stem)
            if img_path is not None:
                counts_image_ok[cat] += 1
    return counts_image_ok


def second_pass_write_manifest_and_sample(
    filter_list_path: Path,
    clip_embed_root: Path,
    dino_embed_root: Path,
    allowed_categories: set[str],
    cropped_dir: Path,
    manifest_path: Path,
    quotas: dict[str, int],
    rng: random.Random,
) -> list[dict[str, str]]:
    """Write full manifest; return stratified sample rows for UMAP (same stream)."""
    per_cat_seen: Counter = Counter()
    sample_rows: list[dict[str, str]] = []
    sample_bufs: dict[str, list[dict[str, str]]] = {c: [] for c in quotas} if quotas else {}
    seen: dict[str, set[str]] = defaultdict(set)

    fieldnames = [
        "category",
        "stem",
        "clip_npy",
        "dinov3_npy",
        "image_path",
        "image_found",
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as out_f:
        w = csv.DictWriter(out_f, fieldnames=fieldnames)
        w.writeheader()
        with filter_list_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if p.suffix.lower() != ".npy":
                    continue
                fname = p.name
                cat = p.parent.name.strip().lower()
                if cat not in allowed_categories:
                    continue
                key = fname.lower()
                if key in seen[cat]:
                    continue
                seen[cat].add(key)
                cdir_c = resolve_category_subdir(clip_embed_root, cat)
                cdir_d = resolve_category_subdir(dino_embed_root, cat)
                if cdir_c is None or cdir_d is None:
                    continue
                pc = remap_absolute_path(cdir_c / fname)
                pd_ = remap_absolute_path(cdir_d / fname)
                if not pc.is_file() or not pd_.is_file():
                    continue
                stem = pc.stem
                img_path, ext_used = resolve_crop_image_path(cropped_dir, cat, stem)
                image_found = img_path is not None
                row = {
                    "category": cat,
                    "stem": stem,
                    "clip_npy": str(pc),
                    "dinov3_npy": str(pd_),
                    "image_path": str(img_path) if img_path else "",
                    "image_found": str(image_found),
                }
                w.writerow(row)

                if not image_found:
                    continue
                per_cat_seen[cat] += 1
                k = quotas.get(cat, 0)
                if k <= 0:
                    continue
                reservoir_add(sample_bufs[cat], row, k, per_cat_seen[cat], rng)

    for c in quotas:
        sample_rows.extend(sample_bufs[c])
    return sample_rows


def install_vest_images(
    sample_rows: list[dict[str, str]],
    images_root: Path,
    mode: str,
    max_workers: int = 1,
) -> list[tuple[str, str]]:
    """
    Returns list of (vest_filename, category) where vest_filename uses / separators,
    relative to images_root parent for VEST --image-path ./images
    """
    images_root.mkdir(parents=True, exist_ok=True)
    if max_workers <= 1 or len(sample_rows) <= 1:
        out: list[tuple[str, str]] = []
        for row in sample_rows:
            t = _install_one_vest_image(row, images_root, mode)
            if t is not None:
                out.append(t)
        return out

    def _one(row: dict[str, str]) -> tuple[str, str] | None:
        return _install_one_vest_image(row, images_root, mode)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_one, sample_rows))
    return [t for t in results if t is not None]


def run_umap(X: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=True,
    )
    return np.asarray(reducer.fit_transform(X), dtype=np.float64)


def main() -> None:
    global DEFAULT_CROPPED_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=NOT_IN_MS_DIR / "vest_export_bv_valid129",
        help="Output directory for manifest, vest_data.csv, images/",
    )
    parser.add_argument("--included-categories", type=Path, default=DEFAULT_INCLUDED)
    parser.add_argument("--clip-embed-dir", type=Path, default=DEFAULT_CLIP_EMB)
    parser.add_argument("--dinov3-embed-dir", type=Path, default=DEFAULT_DINO_EMB)
    parser.add_argument("--clip-filter-list", type=Path, default=DEFAULT_CLIP_FILTER_LIST)
    parser.add_argument("--cropped-dir", type=Path, default=DEFAULT_CROPPED_DIR)
    parser.add_argument(
        "--max-exemplars",
        type=int,
        default=75_000,
        help="Stratified cap for UMAP / vest_data rows (manifest still lists all paired exemplars).",
    )
    parser.add_argument("--umap-seed", type=int, default=42)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to place crops under out-dir/images/",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel threads for loading CLIP .npy files and installing images (default: min(32, CPU+4)).",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Reuse existing exemplar_manifest_valid129.csv in out-dir (skip filter-list scan).",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only build exemplar_manifest_valid129.csv (no UMAP, no vest_data).",
    )
    args = parser.parse_args()

    workers = args.workers
    if workers is None:
        workers = min(32, (os.cpu_count() or 1) + 4)
    workers = max(1, workers)

    cropped_dir = args.cropped_dir.expanduser()
    DEFAULT_CROPPED_DIR = cropped_dir

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "exemplar_manifest_valid129.csv"
    vest_csv = out_dir / "vest_data.csv"
    images_root = out_dir / "images"
    meta_path = out_dir / "vest_export_run.json"

    included = load_included_categories(args.included_categories)
    allowed = set(included)
    if len(allowed) != 129:
        print(f"Warning: expected 129 categories, got {len(allowed)} from {args.included_categories}", file=sys.stderr)

    if not args.clip_filter_list.is_file():
        raise FileNotFoundError(f"CLIP filter list not found: {args.clip_filter_list}")
    if not args.clip_embed_dir.is_dir():
        raise FileNotFoundError(f"CLIP embeddings dir not found: {args.clip_embed_dir}")
    if not args.dinov3_embed_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 embeddings dir not found: {args.dinov3_embed_dir}")

    rng = random.Random(args.umap_seed)

    if args.skip_manifest and manifest_path.is_file():
        print(f"Using existing manifest: {manifest_path}")
        dfm = pd.read_csv(manifest_path)
        dfm = dfm[dfm["image_found"].astype(str).str.lower().isin(("true", "1"))]
        counts = Counter(dfm["category"].astype(str).str.lower())
        quotas = allocate_stratified_quotas(counts, args.max_exemplars)
        sample_rows: list[dict[str, str]] = []
        sample_bufs: dict[str, list[dict[str, str]]] = {c: [] for c in quotas}
        per_cat_seen: Counter = Counter()
        for _, row in dfm.iterrows():
            cat = str(row["category"]).strip().lower()
            per_cat_seen[cat] += 1
            k = quotas.get(cat, 0)
            if k <= 0:
                continue
            d = row.to_dict()
            reservoir_add(sample_bufs[cat], d, k, per_cat_seen[cat], rng)
        for c in quotas:
            sample_rows.extend(sample_bufs[c])
    else:
        print("Pass 1/2: counting paired exemplars with on-disk crops...")
        counts_image_ok = count_pass_filter_list(
            args.clip_filter_list,
            args.clip_embed_dir,
            args.dinov3_embed_dir,
            allowed,
        )
        total_ok = sum(counts_image_ok.values())
        print(f"  exemplars with paired npy + image file: {total_ok:,} across {len(counts_image_ok)} categories")

        quotas = {} if args.manifest_only else allocate_stratified_quotas(counts_image_ok, args.max_exemplars)
        print(
            "Pass 2/2: writing manifest"
            + ("" if args.manifest_only else f" + stratified sample (cap={args.max_exemplars:,})")
            + "..."
        )
        sample_rows = second_pass_write_manifest_and_sample(
            args.clip_filter_list,
            args.clip_embed_dir,
            args.dinov3_embed_dir,
            allowed,
            cropped_dir,
            manifest_path,
            quotas,
            rng,
        )
        print(f"  wrote {manifest_path}")
        if not args.manifest_only:
            print(f"  UMAP subset size: {len(sample_rows):,}")

    if args.manifest_only:
        meta = {
            "manifest_only": True,
            "manifest_csv": str(manifest_path),
            "included_categories_txt": str(args.included_categories),
            "clip_filter_list": str(args.clip_filter_list),
        }
        if manifest_path.is_file():
            meta["manifest_bytes"] = manifest_path.stat().st_size
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Done (--manifest-only). Meta: {meta_path}")
        return

    if not sample_rows:
        raise RuntimeError("No exemplars sampled for UMAP (check image paths / quotas).")

    sample_rows = [r for r in sample_rows if r.get("image_path") and Path(r["image_path"]).is_file()]
    if not sample_rows:
        raise RuntimeError("No sampled rows with a readable image_path.")

    print(f"Loading {len(sample_rows):,} CLIP vectors ({workers} workers)...")
    X_list: list[np.ndarray] = []
    keep_rows: list[dict[str, str]] = []
    if workers <= 1 or len(sample_rows) <= 1:
        for row in sample_rows:
            v = _load_clip_row(row)
            if v is not None:
                X_list.append(v)
                keep_rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vecs = list(ex.map(_load_clip_row, sample_rows))
        for row, v in zip(sample_rows, vecs):
            if v is not None:
                X_list.append(v)
                keep_rows.append(row)
    if not X_list:
        raise RuntimeError("Failed to load any CLIP .npy vectors for the sample.")
    X = np.stack(X_list, axis=0)
    print(f"  matrix shape {X.shape}")

    print("Running UMAP (3D, cosine)...")
    coords = run_umap(X, args.umap_seed, args.umap_n_neighbors, args.umap_min_dist)

    print(f"Linking images under {images_root} ({args.image_mode}, {workers} workers)...")
    vest_files = install_vest_images(keep_rows, images_root, args.image_mode, max_workers=workers)
    if len(vest_files) != len(keep_rows):
        raise RuntimeError(
            f"Symlink/copy dropped {len(keep_rows) - len(vest_files)} rows (unexpected; images were pre-checked)."
        )

    rows_out = []
    for i, (rel_fn, cat) in enumerate(vest_files):
        rows_out.append(
            {
                "filename": rel_fn,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "category": cat,
            }
        )
    pd.DataFrame(rows_out).to_csv(vest_csv, index=False)
    print(f"Wrote {vest_csv} ({len(rows_out):,} rows)")

    meta = {
        "vest_data_csv": str(vest_csv),
        "manifest_csv": str(manifest_path),
        "images_dir": str(images_root),
        "included_categories_txt": str(args.included_categories),
        "clip_filter_list": str(args.clip_filter_list),
        "clip_embeddings_dir": str(args.clip_embed_dir),
        "dinov3_embeddings_dir": str(args.dinov3_embed_dir),
        "cropped_dir": str(cropped_dir),
        "max_exemplars_umap": args.max_exemplars,
        "io_workers": workers,
        "n_vest_rows": len(rows_out),
        "umap": {
            "seed": args.umap_seed,
            "n_neighbors": args.umap_n_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": "cosine",
        },
        "vest_command": f"cd {out_dir} && vest vest_data.csv --image-path ./images",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")
    print("Run VEST:\n ", meta["vest_command"])


if __name__ == "__main__":
    main()
