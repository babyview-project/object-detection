#!/usr/bin/env python3
"""Compute global dispersion and local kNN on the same valid85 / ~7018 crop cohort.

Cohort definition (matches manuscript ``exemplar_embedding_run.json`` for valid85):
  - ``data/included_categories_valid85.txt`` ∩ per-class precision > threshold
  - ``annotation/sampled_object_crops_*_babyview_only.csv`` (trial_type == regular)
  - ``annotation/per_file_precision_data.csv`` (per-crop rater validation)
  - Intersection with CLIP detection filter list (default threshold 0.27)
  - Per-image CLIP and DINOv3 ``.npy`` (cluster paths or git zip)

**Counts:** 7,018 validated crop rows; all have paired CLIP + DINOv3 ``.npy`` on disk when
paths are resolved with case-aware lookup from the CLIP filter list (36 stems use ``_NA_``
in the date slot on disk but are lowercased in the annotation table).

Both metrics use the **same** exemplar vectors per category (Euclidean, k=5).

Examples::

  # Cluster embeddings (also writes missing-manifest if any gaps)
  python analysis/ccn-2026/scripts/compute_valid7018_local_global.py

  # From committed zip (clone-only)
  python analysis/ccn-2026/scripts/compute_valid7018_local_global.py --from-zip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

CCN_DIR = Path(__file__).resolve().parent.parent
CCN_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = CCN_DIR.parent.parent
MANUSCRIPT_SCRIPTS = REPO_ROOT / "analysis" / "manuscript-2026" / "scripts"
DEFAULT_OUT_DIR = CCN_DIR / "valid7018"
DEFAULT_ZIP = REPO_ROOT / "data" / "shared_data_ccn_2026" / "embeddings" / "valid7018_bv_embeddings.zip"

for p in (CCN_SCRIPTS, MANUSCRIPT_SCRIPTS, str(CCN_DIR)):
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

from valid7018_category_metrics import compute_category_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--from-zip",
        action="store_true",
        help="Load embeddings from data/shared_data_ccn_2026/embeddings/valid7018_bv_embeddings.zip",
    )
    p.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help=f"Override zip path (default: {DEFAULT_ZIP.relative_to(REPO_ROOT)})",
    )
    p.add_argument("--k", type=int, default=int(os.environ.get("CCN_VALID7018_K", "5")))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("CCN_VALID7018_OUT_DIR", str(DEFAULT_OUT_DIR))),
    )
    return p.parse_args()


def load_raw_category_embeddings(
    paths_by_cat: dict[str, list[Path]],
    model_label: str,
    crop_prefix: str,
    crop_prefix_new: str,
    min_exemplars: int = 2,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cat in tqdm(sorted(paths_by_cat.keys()), desc=f"Load {model_label}"):
        vecs: list[np.ndarray] = []
        for p in paths_by_cat[cat]:
            pn = remap_absolute_path(Path(p), crop_prefix, crop_prefix_new)
            if not pn.is_file():
                continue
            try:
                v = np.asarray(np.load(pn, mmap_mode="r"), dtype=np.float64).ravel()
                vecs.append(v)
            except Exception:
                continue
        if len(vecs) >= min_exemplars:
            out[cat] = np.stack(vecs, axis=0)
    return out


def load_embeddings_from_zip(zip_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    from load_valid7018_embeddings import load_valid7018_from_zip

    clip_emb, dino_emb = load_valid7018_from_zip(zip_path)
    n = sum(x.shape[0] for x in clip_emb.values())
    return clip_emb, dino_emb, n


def build_missing_embeddings_report(
    exemplar_df: pd.DataFrame,
    clip_paths: dict[str, list[Path]],
    dino_paths: dict[str, list[Path]],
    cfg: dict,
) -> pd.DataFrame:
    rows: list[dict] = []
    dino_by_stem = {
        cat: {Path(p).stem.lower(): p for p in paths} for cat, paths in dino_paths.items()
    }
    for cat in sorted(clip_paths.keys()):
        for p in clip_paths[cat]:
            p = Path(p)
            if p.is_file():
                continue
            stem = p.stem.lower()
            sub = exemplar_df[(exemplar_df["category"] == cat) & (exemplar_df["stem"] == stem)]
            crop_path = ""
            crop_exists = False
            if len(sub) == 1:
                crop_path = str(
                    remap_absolute_path(Path(sub.iloc[0]["path"]), cfg["crop_prefix"], cfg["crop_prefix_new"])
                )
                crop_exists = Path(crop_path).is_file()
            dino_p = Path(dino_by_stem.get(cat, {}).get(stem, cfg["dinov3_embeddings_dir"] / cat / f"{stem}.npy"))
            rows.append(
                {
                    "category": cat,
                    "stem": stem,
                    "clip_npy_exists": False,
                    "dinov3_npy_exists": dino_p.is_file(),
                    "crop_exists": crop_exists,
                    "expected_clip_npy": str(p),
                    "expected_dinov3_npy": str(dino_p),
                }
            )
    return pd.DataFrame(rows)


def rename_for_merge(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = {
        "category": "category",
        "n_exemplars": f"{prefix}_n_exemplars",
        "mean_knn_dist": f"{prefix}_mean_knn_dist",
        "global_dispersion": f"{prefix}_global_dispersion",
        "local_coherence": f"{prefix}_local_coherence",
        "local_over_global": f"{prefix}_local_over_global",
    }
    cols = [c for c in keep if c in df.columns]
    return df[cols].rename(columns={k: v for k, v in keep.items() if k in cols})


def run_metrics(
    clip_emb: dict[str, np.ndarray],
    dino_emb: dict[str, np.ndarray],
    k: int,
    out_dir: Path,
    run_meta_base: dict,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_df = compute_category_metrics(clip_emb, k=k)
    dino_df = compute_category_metrics(dino_emb, k=k)

    clip_path = out_dir / f"bv_valid7018_clip_local_global_k{k}.csv"
    dino_path = out_dir / f"bv_valid7018_dinov3_local_global_k{k}.csv"
    clip_df.to_csv(clip_path, index=False)
    dino_df.to_csv(dino_path, index=False)

    clip_r = rename_for_merge(clip_df, "clip")
    dino_r = rename_for_merge(dino_df, "dinov3")
    merged = clip_r.merge(dino_r, on="category", how="inner")

    stats_rows = []
    for label, gcol, lcol in [
        ("clip_within", "clip_global_dispersion", "clip_mean_knn_dist"),
        ("dinov3_within", "dinov3_global_dispersion", "dinov3_mean_knn_dist"),
    ]:
        sub = merged.dropna(subset=[gcol, lcol])
        rho, p = spearmanr(sub[gcol], sub[lcol])
        stats_rows.append(
            {
                "comparison": label,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n_categories": int(len(sub)),
            }
        )

    for label, c1, c2 in [
        ("cross_global", "clip_global_dispersion", "dinov3_global_dispersion"),
        ("cross_local_knn", "clip_mean_knn_dist", "dinov3_mean_knn_dist"),
    ]:
        sub = merged.dropna(subset=[c1, c2])
        rho, p = spearmanr(sub[c1], sub[c2])
        stats_rows.append(
            {
                "comparison": label,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "n_categories": int(len(sub)),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / f"bv_valid7018_correlations_k{k}.csv"
    stats_df.to_csv(stats_path, index=False)

    merged_path = out_dir / f"bv_valid7018_clip_vs_dinov3_local_global_k{k}.csv"
    merged.to_csv(merged_path, index=False)

    cohort_counts = (
        clip_df[["category", "n_exemplars"]]
        .rename(columns={"n_exemplars": "n_exemplars_clip"})
        .merge(
            dino_df[["category", "n_exemplars"]].rename(columns={"n_exemplars": "n_exemplars_dinov3"}),
            on="category",
            how="outer",
        )
    )
    cohort_counts.to_csv(out_dir / "bv_valid7018_n_exemplars_by_category.csv", index=False)

    run_meta = {
        **run_meta_base,
        "n_exemplars_loaded_clip": int(sum(x.shape[0] for x in clip_emb.values())),
        "n_exemplars_loaded_dinov3": int(sum(x.shape[0] for x in dino_emb.values())),
        "n_categories_clip_metrics": int(len(clip_df)),
        "n_categories_dinov3_metrics": int(len(dino_df)),
        "n_categories_merged": int(len(merged)),
        "k": k,
        "embedding_metric": "euclidean_raw_vectors",
        "global_dispersion": "mean L2 distance to category centroid",
        "local_knn": f"mean kNN distance (k={k})",
        "correlations": stats_rows,
        "outputs": {
            "clip_metrics": clip_path.name,
            "dinov3_metrics": dino_path.name,
            "merged": merged_path.name,
            "correlations": stats_path.name,
        },
    }
    (out_dir / "valid7018_run.json").write_text(json.dumps(run_meta, indent=2))

    print(f"\nWrote outputs under: {out_dir}")
    print(f"  Exemplars loaded — CLIP: {run_meta['n_exemplars_loaded_clip']}, DINOv3: {run_meta['n_exemplars_loaded_dinov3']}")
    print(f"  Categories: {len(clip_df)}")
    print("\nCorrelations (same cohort, paired exemplars per model):")
    for row in stats_rows:
        print(f"  {row['comparison']}: rho={row['spearman_rho']:.4f} (n={row['n_categories']})")
    return 0


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser()
    k = args.k

    if args.from_zip:
        zip_path = (args.zip_path or DEFAULT_ZIP).expanduser()
        print(f"Loading from zip: {zip_path}")
        clip_emb, dino_emb, n_zip = load_embeddings_from_zip(zip_path)
        base = {
            "cohort": "valid85_sampled_per_file_validated_clip_filter_list",
            "embedding_source": "valid7018_bv_embeddings.zip",
            "zip_path": str(zip_path.relative_to(REPO_ROOT))
            if str(zip_path).startswith(str(REPO_ROOT))
            else str(zip_path),
            "n_exemplars_in_zip": n_zip,
            "n_rows_sampled_validated": None,
        }
        return run_metrics(clip_emb, dino_emb, k, out_dir, base)

    cfg = load_config()
    inc_path = CATEGORY_FILES["valid85"]
    exemplar_df = build_valid85_sampled_exemplar_table(
        inc_path,
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

    n_clip_listed = sum(len(v) for v in clip_paths.values())
    n_dino_listed = sum(len(v) for v in dino_paths.values())

    missing_df = build_missing_embeddings_report(exemplar_df, clip_paths, dino_paths, cfg)
    if not missing_df.empty:
        miss_path = out_dir / "valid7018_missing_embeddings.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(miss_path, index=False)
        print(f"Wrote {miss_path} ({len(missing_df)} crops without CLIP .npy on disk)")

    clip_emb = load_raw_category_embeddings(clip_paths, "CLIP", crop_prefix, crop_prefix_new)
    dino_emb = load_raw_category_embeddings(dino_paths, "DINOv3", crop_prefix, crop_prefix_new)

    base = {
        "cohort": "valid85_sampled_per_file_validated_clip_filter_list",
        "embedding_source": "cluster_npy_dirs",
        "n_rows_sampled_validated": int(len(exemplar_df)),
        "n_paths_listed_clip": int(n_clip_listed),
        "n_paths_listed_dinov3": int(n_dino_listed),
        "n_missing_both_npy_on_disk": int(len(missing_df)),
        "n_missing_clip_only_on_disk": int((~missing_df["clip_npy_exists"] & missing_df["dinov3_npy_exists"]).sum())
        if not missing_df.empty
        else 0,
        "precision_threshold": cfg["precision_threshold"],
        "clip_filter_list_path": os.environ.get("BV_CLIP_FILTER_LIST", "<BV_CLIP_FILTER_LIST>"),
        "note": (
            "7018 validated annotation rows; paired CLIP+DINOv3 .npy resolved via "
            "case-aware lookup from the CLIP filter list."
        ),
    }
    if not missing_df.empty:
        base["missing_embeddings_csv"] = "valid7018_missing_embeddings.csv"

    print(f"  Cohort rows (sampled+validated): {len(exemplar_df)}")
    print(f"  Paths listed (CLIP / DINO): {n_clip_listed} / {n_dino_listed}")
    if len(missing_df):
        print(f"  Missing both .npy on disk: {len(missing_df)} (see valid7018_missing_embeddings.csv)")

    return run_metrics(clip_emb, dino_emb, k, out_dir, base)


if __name__ == "__main__":
    raise SystemExit(main())
