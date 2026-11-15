#!/usr/bin/env python3
"""Build public-release valid7018 crops + embeddings (privacy-filtered).

Excludes ``PUBLIC_EXCLUDE_CATEGORIES`` (body parts + glasses) — the same set
used for montage/figure privacy filtering. Stems are replaced with opaque ids
so subject/file identifiers are not shipped.

Reads the full internal zips and writes::

  data/shared_data_ccn_2026/public/crops/valid7018_public_crops.zip
  data/shared_data_ccn_2026/public/embeddings/valid7018_public_embeddings.zip

Run from repo root::

  python analysis/ccn-2026/scripts/build_valid7018_public_release.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

CCN_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CCN_DIR.parent.parent
SHARED = REPO_ROOT / "data" / "shared_data_ccn_2026"
PUBLIC_ROOT = SHARED / "public"

SRC_CROPS = SHARED / "crops" / "valid7018_crops.zip"
SRC_EMB = SHARED / "embeddings" / "valid7018_bv_embeddings.zip"
SRC_NORM = SHARED / "embeddings" / "valid7018_embedding_norm_stats.json"

OUT_CROPS = PUBLIC_ROOT / "crops" / "valid7018_public_crops.zip"
OUT_EMB = PUBLIC_ROOT / "embeddings" / "valid7018_public_embeddings.zip"

if str(CCN_DIR) not in sys.path:
    sys.path.insert(0, str(CCN_DIR))

from valid7018_category_filters import (  # noqa: E402
    BODY_PART_CATEGORIES,
    MONTAGE_EXCLUDE_CATEGORIES,
    PUBLIC_EXCLUDE_CATEGORIES,
    PUBLIC_EXCLUDE_REASONS,
)


def anonymize_stem(category: str, stem: str) -> str:
    """Opaque, stable id — no subject/date/hash fragments from the original stem."""
    digest = hashlib.sha256(f"{category.lower()}:{stem.lower()}".encode("utf-8")).hexdigest()[:16]
    return f"{category.lower()}_{digest}"


def _read_manifest(zf: zipfile.ZipFile) -> list[dict[str, str]]:
    with zf.open("manifest.csv") as mf:
        return list(csv.DictReader(io.TextIOWrapper(mf, encoding="utf-8")))


def _write_csv(zf: zipfile.ZipFile, name: str, rows: list[dict], fieldnames: list[str]) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    zf.writestr(name, buf.getvalue())


def build_public_crops(src: Path, dst: Path, keep_pairs: set[tuple[str, str]]) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".zip.part")
    if tmp.exists():
        tmp.unlink()

    n_in = n_out = 0
    excluded_counts: dict[str, int] = {}
    manifest_rows: list[dict[str, str]] = []

    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zout:
        zout.writestr(
            "README.txt",
            (
                "BabyView CCN 2026 — valid7018 PUBLIC crop archive\n"
                "=================================================\n\n"
                "Privacy-filtered crop JPEGs for public release.\n"
                "Excluded: body-part categories + glasses (face privacy).\n"
                "Filenames use opaque stems (no subject/file ids).\n\n"
                "Aligned 1:1 with embeddings/valid7018_public_embeddings.zip\n"
                "via manifest.csv (category, stem, jpeg_path).\n"
            ),
        )
        for row in _read_manifest(zin):
            n_in += 1
            cat = row["category"].strip().lower()
            stem = row["stem"].strip().lower()
            if cat in PUBLIC_EXCLUDE_CATEGORIES:
                excluded_counts[cat] = excluded_counts.get(cat, 0) + 1
                continue
            if (cat, stem) not in keep_pairs:
                continue
            anon = anonymize_stem(cat, stem)
            src_member = row["jpeg_path"]
            suffix = Path(src_member).suffix.lower() or ".jpg"
            if suffix == ".jpeg":
                suffix = ".jpg"
            out_member = str(PurePosixPath("crops") / cat / f"{anon}{suffix}")
            zout.writestr(out_member, zin.read(src_member))
            manifest_rows.append(
                {"category": cat, "stem": anon, "jpeg_path": out_member}
            )
            n_out += 1
        _write_csv(zout, "manifest.csv", manifest_rows, ["category", "stem", "jpeg_path"])

    tmp.replace(dst)
    return {
        "n_in": n_in,
        "n_out": n_out,
        "n_categories": len({r["category"] for r in manifest_rows}),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "zip_size_bytes": dst.stat().st_size,
    }


def build_public_embeddings(src: Path, dst: Path) -> tuple[dict, set[tuple[str, str]]]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".zip.part")
    if tmp.exists():
        tmp.unlink()

    n_in = n_out = 0
    excluded_counts: dict[str, int] = {}
    manifest_rows: list[dict[str, str]] = []
    keep_pairs: set[tuple[str, str]] = set()

    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(
        tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zout:
        zout.writestr(
            "README.txt",
            (
                "BabyView CCN 2026 — valid7018 PUBLIC embedding archive\n"
                "=====================================================\n\n"
                "Privacy-filtered CLIP + DINOv3 per-crop vectors.\n"
                "Excluded: body-part categories + glasses (face privacy).\n"
                "Stems are opaque ids aligned with the public crops zip.\n"
                "Vectors keep the same cohort z-score as the full 7,018 archive\n"
                "(mu/sigma fit on all validated crops; see norm stats JSON).\n\n"
                "Layout: manifest.csv, clip/{cat}/{stem}.npy, dinov3/{cat}/{stem}.npy\n"
            ),
        )
        for row in _read_manifest(zin):
            n_in += 1
            cat = row["category"].strip().lower()
            stem = row["stem"].strip().lower()
            if cat in PUBLIC_EXCLUDE_CATEGORIES:
                excluded_counts[cat] = excluded_counts.get(cat, 0) + 1
                continue
            anon = anonymize_stem(cat, stem)
            clip_out = str(PurePosixPath("clip") / cat / f"{anon}.npy")
            dino_out = str(PurePosixPath("dinov3") / cat / f"{anon}.npy")
            zout.writestr(clip_out, zin.read(row["clip_npy"]))
            zout.writestr(dino_out, zin.read(row["dinov3_npy"]))
            manifest_rows.append(
                {
                    "category": cat,
                    "stem": anon,
                    "clip_npy": clip_out,
                    "dinov3_npy": dino_out,
                }
            )
            keep_pairs.add((cat, stem))
            n_out += 1
        _write_csv(
            zout,
            "manifest.csv",
            manifest_rows,
            ["category", "stem", "clip_npy", "dinov3_npy"],
        )

    tmp.replace(dst)
    stats = {
        "n_in": n_in,
        "n_out": n_out,
        "n_categories": len({r["category"] for r in manifest_rows}),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "zip_size_bytes": dst.stat().st_size,
    }
    return stats, keep_pairs


def main() -> int:
    if not SRC_EMB.is_file():
        raise FileNotFoundError(f"Missing embedding zip: {SRC_EMB}")
    if not SRC_CROPS.is_file():
        raise FileNotFoundError(f"Missing crops zip: {SRC_CROPS}")

    emb_stats, keep_pairs = build_public_embeddings(SRC_EMB, OUT_EMB)
    crop_stats = build_public_crops(SRC_CROPS, OUT_CROPS, keep_pairs)

    if emb_stats["n_out"] != crop_stats["n_out"]:
        raise RuntimeError(
            f"Public emb/crop count mismatch: {emb_stats['n_out']} vs {crop_stats['n_out']}"
        )

    if SRC_NORM.is_file():
        norm_dst = PUBLIC_ROOT / "embeddings" / SRC_NORM.name
        norm_dst.write_bytes(SRC_NORM.read_bytes())

    present_excluded = sorted(
        set(emb_stats["excluded_counts"]) | set(crop_stats["excluded_counts"])
    )
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_embeddings_zip": str(SRC_EMB.relative_to(REPO_ROOT)),
        "source_crops_zip": str(SRC_CROPS.relative_to(REPO_ROOT)),
        "public_embeddings_zip": str(OUT_EMB.relative_to(REPO_ROOT)),
        "public_crops_zip": str(OUT_CROPS.relative_to(REPO_ROOT)),
        "n_exemplars_public": emb_stats["n_out"],
        "n_categories_public": emb_stats["n_categories"],
        "n_exemplars_source": emb_stats["n_in"],
        "n_categories_excluded_present": len(present_excluded),
        "exclude_policy": {
            "body_parts": sorted(BODY_PART_CATEGORIES),
            "face_privacy": sorted(MONTAGE_EXCLUDE_CATEGORIES),
            "union": sorted(PUBLIC_EXCLUDE_CATEGORIES),
            "reasons": PUBLIC_EXCLUDE_REASONS,
        },
        "excluded_category_counts": emb_stats["excluded_counts"],
        "stem_anonymization": "sha256(category:stem)[:16] → {category}_{digest}",
        "normalization_note": (
            "Public embedding vectors are a subset of the full valid7018 zip; "
            "feature-wise z-score mu/sigma were fit on all 7,018 crops."
        ),
        "embeddings_zip_size_bytes": emb_stats["zip_size_bytes"],
        "crops_zip_size_bytes": crop_stats["zip_size_bytes"],
    }
    meta_path = PUBLIC_ROOT / "valid7018_public_release.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    readme = f"""\
# valid7018 public release (privacy-filtered)

Paired crop JPEGs + CLIP/DINOv3 embeddings with sensitive categories removed.

## Exclusions

Body parts ({len(BODY_PART_CATEGORIES)} labels): {", ".join(sorted(BODY_PART_CATEGORIES))}

Face privacy: {", ".join(sorted(MONTAGE_EXCLUDE_CATEGORIES))}

Categories present in the source cohort that were dropped:
{", ".join(present_excluded) if present_excluded else "(none)"}

## Contents

- `crops/valid7018_public_crops.zip` — {crop_stats["n_out"]:,} JPEGs, {crop_stats["n_categories"]} categories
- `embeddings/valid7018_public_embeddings.zip` — matching CLIP + DINOv3 `.npy`
- `embeddings/valid7018_embedding_norm_stats.json` — cohort z-score μ/σ (fit on full 7,018)
- `valid7018_public_release.json` — build metadata

Stems are opaque ids (no subject/file identifiers). Manifests align crops ↔ embeddings 1:1.

Rebuild::

  python analysis/ccn-2026/scripts/build_valid7018_public_release.py
"""
    (PUBLIC_ROOT / "README.md").write_text(readme)

    print(f"Public embeddings: {OUT_EMB.relative_to(REPO_ROOT)} ({emb_stats['n_out']} vecs)")
    print(f"Public crops:      {OUT_CROPS.relative_to(REPO_ROOT)} ({crop_stats['n_out']} JPEGs)")
    print(f"Excluded categories present: {present_excluded}")
    print(f"Metadata: {meta_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
