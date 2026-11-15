#!/usr/bin/env python3
"""Package all 7,018 rater-validated crop JPEGs for git sharing.

Aligns 1:1 with ``data/shared_data_ccn_2026/embeddings/valid7018_bv_embeddings.zip``
(same category/stem rows). Crops are read from ``BV_CROPPED_DIR`` (default:
``yoloe_cdi_all_cropped_by_class``).

Run from repo root (requires cluster crop paths)::

  export BV_CROPPED_DIR=/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class
  python analysis/ccn-2026/scripts/build_valid7018_crops_zip.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

CCN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CCN_DIR.parent.parent
DEFAULT_EMBEDDING_ZIP = (
    REPO_ROOT / "data" / "shared_data_ccn_2026" / "embeddings" / "valid7018_bv_embeddings.zip"
)
DEFAULT_ZIP = REPO_ROOT / "data" / "shared_data_ccn_2026" / "crops" / "valid7018_crops.zip"
DEFAULT_CROPPED_DIR = Path(
    "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class"
)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

README_TXT = """\
BabyView CCN 2026 — valid7018 crop archive
==========================================

All ~7,018 human-validated object crops (85 categories), aligned 1:1 with
embeddings/valid7018_bv_embeddings.zip (same category + stem).

Layout:
  manifest.csv — category, stem, jpeg_path (paths inside this zip)
  crops/{category}/{stem}.jpg

Source (maintainer):
  BV_CROPPED_DIR/{category}/{stem}.jpg

Rebuild:
  python analysis/ccn-2026/scripts/build_valid7018_crops_zip.py
"""


def archive_member(category: str, stem: str, suffix: str = ".jpg") -> str:
    return str(PurePosixPath("crops") / category / f"{stem}{suffix}")


def resolve_crop_path(cropped_dir: Path, category: str, stem: str) -> Path | None:
    cat_dir = cropped_dir / category
    if not cat_dir.is_dir():
        for p in cropped_dir.iterdir():
            if p.is_dir() and p.name.lower() == category.lower():
                cat_dir = p
                break
        else:
            return None

    for ext in IMAGE_EXTS:
        candidate = cat_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate

    stem_l = stem.lower()
    for p in cat_dir.iterdir():
        if p.is_file() and p.stem.lower() == stem_l and p.suffix in IMAGE_EXTS:
            return p
    return None


def load_embedding_manifest(embedding_zip: Path) -> list[dict[str, str]]:
    if not embedding_zip.is_file():
        raise FileNotFoundError(
            f"Embedding zip not found (build embeddings first): {embedding_zip}"
        )
    with zipfile.ZipFile(embedding_zip, "r") as zf:
        with zf.open("manifest.csv") as mf:
            return list(csv.DictReader(io.TextIOWrapper(mf, encoding="utf-8")))


def main() -> int:
    cropped_dir = Path(
        os.environ.get("BV_CROPPED_DIR", str(DEFAULT_CROPPED_DIR))
    ).expanduser()
    embedding_zip = Path(
        os.environ.get("CCN_VALID7018_EMBEDDING_ZIP", str(DEFAULT_EMBEDDING_ZIP))
    ).expanduser()
    out_zip = Path(os.environ.get("CCN_VALID7018_CROPS_ZIP", str(DEFAULT_ZIP))).expanduser()

    if not cropped_dir.is_dir():
        raise FileNotFoundError(f"Crop root not found: {cropped_dir}")

    rows = load_embedding_manifest(embedding_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    resolved: list[tuple[dict[str, str], Path]] = []
    missing: list[tuple[str, str]] = []
    for row in rows:
        cat = row["category"].strip().lower()
        stem = row["stem"].strip().lower()
        src = resolve_crop_path(cropped_dir, cat, stem)
        if src is None:
            missing.append((cat, stem))
            continue
        resolved.append(({"category": cat, "stem": stem}, src))

    if missing:
        eg = ", ".join(f"{c}/{s}" for c, s in missing[:5])
        raise RuntimeError(
            f"Missing {len(missing)}/{len(rows)} crop files under {cropped_dir}. "
            f"Examples: {eg}"
        )

    tmp_zip = out_zip.with_suffix(".zip.part")
    if tmp_zip.exists():
        tmp_zip.unlink()

    manifest_rows: list[dict[str, str]] = []
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("README.txt", README_TXT)
        for meta, src in resolved:
            suffix = src.suffix.lower() if src.suffix.lower() in {".jpg", ".jpeg", ".png"} else ".jpg"
            if suffix == ".jpeg":
                suffix = ".jpg"
            member = archive_member(meta["category"], meta["stem"], suffix)
            zf.write(src, arcname=member)
            manifest_rows.append(
                {
                    "category": meta["category"],
                    "stem": meta["stem"],
                    "jpeg_path": member,
                }
            )

        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=["category", "stem", "jpeg_path"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
        zf.writestr("manifest.csv", buf.getvalue())

    tmp_zip.replace(out_zip)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "zip_path": str(out_zip.relative_to(REPO_ROOT)),
        "cropped_dir": str(cropped_dir),
        "aligned_embedding_zip": str(embedding_zip.relative_to(REPO_ROOT)),
        "n_crops_in_zip": len(manifest_rows),
        "n_categories": len({r["category"] for r in manifest_rows}),
        "missing": 0,
        "zip_size_bytes": out_zip.stat().st_size,
    }
    meta_path = out_zip.with_name(out_zip.stem + "_zip.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Wrote {out_zip.relative_to(REPO_ROOT)} ({out_zip.stat().st_size / 1e6:.2f} MB)")
    print(f"  Crops: {len(manifest_rows)}  categories: {meta['n_categories']}")
    print(f"  Aligned to: {embedding_zip.relative_to(REPO_ROOT)}")
    print(f"  Metadata: {meta_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
