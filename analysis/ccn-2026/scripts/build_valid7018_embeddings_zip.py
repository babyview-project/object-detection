#!/usr/bin/env python3
"""Package 7,018 rater-validated CLIP + DINOv3 crop embeddings for git sharing.

Per-crop vectors are read from ``clip_embeddings_new`` and
``facebook_dinov3-vitb16-pretrain-lvd1689m`` under ``BV_EMBEDDINGS_BASE``, then
feature-wise globally normalized using mu/sigma from grouped age-month dirs
(notebook 05; see ``valid7018_embedding_normalize.py``).

Default ``BV_EMBEDDINGS_BASE``::
  /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings

Run from repo root (requires cluster embedding paths)::

  export BV_EMBEDDINGS_BASE=/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings
  python analysis/ccn-2026/scripts/build_valid7018_embeddings_zip.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

CCN_DIR = Path(__file__).resolve().parent.parent
CCN_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = CCN_DIR.parent.parent
MANUSCRIPT_SCRIPTS = REPO_ROOT / "analysis" / "manuscript-2026" / "scripts"
DEFAULT_ZIP = REPO_ROOT / "data" / "shared_data_ccn_2026" / "embeddings" / "valid7018_bv_embeddings.zip"
DEFAULT_NORM_STATS = CCN_DIR / "valid7018" / "valid7018_embedding_norm_stats.json"
SHARED_NORM_STATS = (
    REPO_ROOT / "data" / "shared_data_ccn_2026" / "embeddings" / "valid7018_embedding_norm_stats.json"
)

for p in (CCN_SCRIPTS, MANUSCRIPT_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exemplar_set_zscore_embeddings import (  # noqa: E402
    CATEGORY_FILES,
    PER_CLASS_PRECISION_CSV,
    PER_FILE_PRECISION_CSV,
    SAMPLED_EXEMPLAR_CSV,
    build_valid85_sampled_exemplar_table,
    load_clip_filter_npy_fname_map,
    load_clip_filter_pair_set,
    load_config,
    remap_absolute_path,
    valid85_npy_paths_by_category_from_manifest,
)

from valid7018_embedding_normalize import fit_and_save_norm_stats, load_norm_stats  # noqa: E402


README_TXT = """\
BabyView CCN 2026 — valid7018 embedding archive
===============================================

~7,018 human-validated object crops (85 categories), each with paired:
  - CLIP ViT-B/32 512-d vectors
  - DINOv3 ViT-B/16 768-d vectors

Vectors are feature-wise globally normalized (notebook 05 stats from grouped
age-month embeddings under BV_EMBEDDINGS_BASE), then stored as float16 .npy.

Cohort: sampled_object_crops CSV (regular trials) ∩ per_file_precision > 0.6
        ∩ included_categories_valid85 ∩ per-class precision > 0.6
        ∩ CLIP detection filter list (default threshold 0.27).

Sources (maintainer):
  BV_EMBEDDINGS_BASE/clip_embeddings_new/{category}/{stem}.npy
  BV_EMBEDDINGS_BASE/facebook_dinov3-vitb16-pretrain-lvd1689m/{category}/{stem}.npy

Layout:
  manifest.csv — category, stem, clip_npy, dinov3_npy (paths inside this zip)
  clip/{category}/{stem}.npy
  dinov3/{category}/{stem}.npy

Metrics: analysis/ccn-2026/valid7018/
Norm stats: valid7018_embedding_norm_stats.json (same bundle)

Rebuild:
  python analysis/ccn-2026/scripts/build_valid7018_embeddings_zip.py
"""


def archive_member(category: str, stem: str, model: str) -> str:
    return str(PurePosixPath(model) / category / f"{stem}.npy")


def _load_vec(path: Path, crop_prefix: str, crop_prefix_new: str) -> np.ndarray:
    p = remap_absolute_path(path, crop_prefix, crop_prefix_new)
    return np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64).ravel()


def _write_npy_to_zip(zf: zipfile.ZipFile, arcname: str, vec: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, vec.astype(np.float16), allow_pickle=False)
    zf.writestr(arcname, buf.getvalue())


def main() -> int:
    cfg = load_config()
    out_zip = Path(os.environ.get("CCN_VALID7018_ZIP", str(DEFAULT_ZIP))).expanduser()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    emb_base = Path(
        os.environ.get(
            "BV_EMBEDDINGS_BASE",
            "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings",
        )
    ).expanduser()

    norm_stats_path = DEFAULT_NORM_STATS if DEFAULT_NORM_STATS.is_file() else SHARED_NORM_STATS
    if not norm_stats_path.is_file():
        threshold = str(cfg.get("clip_filter_list_threshold", "0.27"))
        print(f"Fitting norm stats → {norm_stats_path}")
        fit_and_save_norm_stats(emb_base, norm_stats_path, threshold=threshold)
        if norm_stats_path != DEFAULT_NORM_STATS:
            fit_and_save_norm_stats(emb_base, DEFAULT_NORM_STATS, threshold=threshold)
        if norm_stats_path != SHARED_NORM_STATS:
            fit_and_save_norm_stats(emb_base, SHARED_NORM_STATS, threshold=threshold)
    norm = load_norm_stats(norm_stats_path)
    mu_c, sig_c = norm["clip"]
    mu_d, sig_d = norm["dinov3"]

    exemplar_df = build_valid85_sampled_exemplar_table(
        CATEGORY_FILES["valid85"],
        PER_CLASS_PRECISION_CSV,
        PER_FILE_PRECISION_CSV,
        SAMPLED_EXEMPLAR_CSV,
        cfg["precision_threshold"],
    )
    elig_cats = set(exemplar_df["category"].astype(str).str.strip().str.lower())
    clip_filter_pairs = load_clip_filter_pair_set(cfg["clip_filter_list_path"], elig_cats)
    npy_fname_map = load_clip_filter_npy_fname_map(cfg["clip_filter_list_path"], elig_cats)

    crop_prefix = cfg["crop_prefix"]
    crop_prefix_new = cfg["crop_prefix_new"]

    clip_paths = valid85_npy_paths_by_category_from_manifest(
        exemplar_df,
        cfg["clip_embeddings_dir"],
        clip_filter_pairs,
        crop_prefix,
        crop_prefix_new,
        npy_fname_map,
    )
    dino_paths = valid85_npy_paths_by_category_from_manifest(
        exemplar_df,
        cfg["dinov3_embeddings_dir"],
        clip_filter_pairs,
        crop_prefix,
        crop_prefix_new,
        npy_fname_map,
    )

    rows: list[dict[str, str]] = []
    missing_clip = 0
    missing_dino = 0
    n_listed = 0
    for cat in sorted(clip_paths.keys()):
        dino_by_stem = {Path(p).stem.lower(): p for p in dino_paths.get(cat, [])}
        for clip_src in clip_paths[cat]:
            n_listed += 1
            clip_src = Path(clip_src)
            stem = clip_src.stem.lower()
            dino_src = dino_by_stem.get(stem)
            if not clip_src.is_file():
                missing_clip += 1
                continue
            if dino_src is None or not Path(dino_src).is_file():
                missing_dino += 1
                continue
            rows.append(
                {
                    "category": cat,
                    "stem": stem,
                    "clip_npy": archive_member(cat, stem, "clip"),
                    "dinov3_npy": archive_member(cat, stem, "dinov3"),
                    "_clip_src": clip_src,
                    "_dino_src": Path(dino_src),
                }
            )

    if not rows:
        raise RuntimeError("No exemplars with both CLIP and DINOv3 files on disk.")

    tmp_zip = out_zip.with_suffix(".zip.part")
    if tmp_zip.exists():
        tmp_zip.unlink()

    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("README.txt", README_TXT)

        buf = io.StringIO()
        fieldnames = ["category", "stem", "clip_npy", "dinov3_npy"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
        zf.writestr("manifest.csv", buf.getvalue())

        for r in rows:
            clip_v = (_load_vec(r["_clip_src"], crop_prefix, crop_prefix_new) - mu_c) / sig_c
            dino_v = (_load_vec(r["_dino_src"], crop_prefix, crop_prefix_new) - mu_d) / sig_d
            _write_npy_to_zip(zf, r["clip_npy"], clip_v)
            _write_npy_to_zip(zf, r["dinov3_npy"], dino_v)

    tmp_zip.replace(out_zip)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "zip_path": str(out_zip.relative_to(REPO_ROOT)),
        "embeddings_base": str(emb_base),
        "clip_embeddings_dir": str(cfg["clip_embeddings_dir"]),
        "dinov3_embeddings_dir": str(cfg["dinov3_embeddings_dir"]),
        "normalization": "featurewise_global_from_grouped_age_month",
        "norm_stats_json": str(norm_stats_path.relative_to(REPO_ROOT)),
        "n_exemplars_in_zip": len(rows),
        "n_paths_listed_cohort": n_listed,
        "n_rows_sampled_validated": int(len(exemplar_df)),
        "missing_clip": missing_clip,
        "missing_dino": missing_dino,
        "zip_size_bytes": out_zip.stat().st_size,
        "precision_threshold": cfg["precision_threshold"],
    }
    meta_path = out_zip.parent / "valid7018_bv_embeddings_zip.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out_zip} ({out_zip.stat().st_size / 1e6:.2f} MB)")
    print(f"  BV_EMBEDDINGS_BASE: {emb_base}")
    print(f"  Exemplars in zip: {len(rows)}")
    print(f"  Skipped (missing clip/dino): {missing_clip}/{missing_dino}")
    print(f"  Norm stats: {norm_stats_path}")
    print(f"  Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
