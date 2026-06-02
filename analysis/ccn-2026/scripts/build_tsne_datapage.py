#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


CCN_DIR = Path(__file__).resolve().parent.parent
MANUSCRIPT_DIR = CCN_DIR.parent / "manuscript-2026"

DEFAULT_TSNE_CSV = MANUSCRIPT_DIR / "tsne_cdi_results_clip" / "tsne_cdi_coordinates.csv"
# Expects dist_to_things_centroid (joint BV–THINGS run); BV-only Plot B CSVs use dist_to_bv_centroid instead.
DEFAULT_PER_EXEMPLAR_CSV = (
    CCN_DIR
    / "old_plots"
    / "plotB_tsne_distance_to_centroid_outputs_20260401"
    / "bv_to_things_centroid_clip_per_exemplar.csv"
)
DEFAULT_CROPPED_DIR = Path(
    os.getenv("BV_CROPS_BASE", "SET_BV_CROPS_BASE")
).expanduser()
DEFAULT_EMBEDDINGS_DIR = Path(
    os.getenv("BV_CLIP_EMBEDDINGS_DIR", "SET_BV_CLIP_EMBEDDINGS_DIR")
).expanduser()


def find_crop_file(cropped_root: Path, category: str, stem: str) -> Path | None:
    cat_dir = cropped_root / category
    if not cat_dir.exists():
        for child in cropped_root.iterdir():
            if child.is_dir() and child.name.lower() == category.lower():
                cat_dir = child
                break
    if not cat_dir.exists():
        return None

    for ext in (".jpg", ".jpeg", ".png"):
        p = cat_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_embedding_file(embeddings_root: Path | None, category: str, stem: str) -> Path | None:
    if embeddings_root is None or not embeddings_root.exists():
        return None
    cat_dir = embeddings_root / category
    if not cat_dir.exists():
        for child in embeddings_root.iterdir():
            if child.is_dir() and child.name.lower() == category.lower():
                cat_dir = child
                break
    if not cat_dir.exists():
        return None
    p = cat_dir / f"{stem}.npy"
    return p if p.exists() else None


def tsne_2d(X: np.ndarray, random_state: int, *, max_iter: int) -> np.ndarray:
    """Return (n, 2) coordinates for rows of X (float, 2+ samples)."""
    n, _d = X.shape
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64)
    if n == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    n_comp = int(min(50, n - 1, X.shape[1]))
    if n_comp >= 2:
        X = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)

    perp = min(30, max(2, n // 4))
    if perp >= n:
        perp = n - 1
    if perp < 1:
        perp = 1
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
    )
    return tsne.fit_transform(X)


def attach_within_category_tsne(
    category: str,
    samples: list[dict],
    embeddings_root: Path | None,
    random_state: int,
    *,
    tsne_max_iter: int,
) -> tuple[int, int]:
    """Add within_tsne_x / within_tsne_y to samples where embeddings exist. Returns (with_emb, with_coords)."""
    if not samples or embeddings_root is None:
        return 0, 0

    emb_list: list[np.ndarray] = []
    valid_idx: list[int] = []
    for j, s in enumerate(samples):
        stem = str(s.get("stem", "")).strip()
        path = find_embedding_file(embeddings_root, category, stem)
        if path is None:
            continue
        vec = np.load(path)
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        if v.size == 0:
            continue
        emb_list.append(v)
        valid_idx.append(j)

    if len(emb_list) < 2:
        return len(emb_list), 0

    X = np.stack(emb_list, axis=0)
    Y = tsne_2d(X, random_state, max_iter=tsne_max_iter)
    for j, yrow in zip(valid_idx, Y):
        samples[j]["within_tsne_x"] = float(yrow[0])
        samples[j]["within_tsne_y"] = float(yrow[1])
    return len(emb_list), len(valid_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive t-SNE datapage manifest and thumbnails.")
    parser.add_argument("--tsne-csv", type=Path, default=DEFAULT_TSNE_CSV)
    parser.add_argument("--per-exemplar-csv", type=Path, default=DEFAULT_PER_EXEMPLAR_CSV)
    parser.add_argument("--cropped-dir", type=Path, default=DEFAULT_CROPPED_DIR)
    parser.add_argument("--out-dir", type=Path, default=CCN_DIR / "tsne_datapage")
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=48,
        help="Exemplars per category in manifest (datapage dropdown can show a subset; increase for even more, e.g. 96).",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Per-exemplar .npy CLIP dirs (category/stem.npy). Omit or set empty to skip within-category t-SNE.",
    )
    parser.add_argument(
        "--no-within-tsne",
        action="store_true",
        help="Do not compute per-category exemplar t-SNE (smaller / faster manifest).",
    )
    parser.add_argument(
        "--within-tsne-max-iter",
        type=int,
        default=400,
        help="sklearn TSNE max_iter per category (lower = faster, rougher).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = args.out_dir.resolve()
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    tsne_df = pd.read_csv(args.tsne_csv)
    ex_df = pd.read_csv(
        args.per_exemplar_csv,
        usecols=["category", "subject_id", "age_mo", "dist_to_things_centroid", "dist_to_bv_centroid", "stem"],
        dtype={"subject_id": str, "category": str, "stem": str},
    )

    ex_df = ex_df.dropna(subset=["category", "stem"])
    ex_df["category"] = ex_df["category"].astype(str).str.strip()
    ex_df["stem"] = ex_df["stem"].astype(str).str.strip()

    records: list[dict] = []
    copied = 0
    missing = 0
    emb_root: Path | None = None
    if not args.no_within_tsne and args.embeddings_dir:
        p = args.embeddings_dir.expanduser().resolve()
        emb_root = p if p.exists() else None

    within_tsne_categories = 0
    within_tsne_exemplars = 0

    for _, row in tsne_df.iterrows():
        cat = str(row["class_name"]).strip()
        subset = ex_df[ex_df["category"].str.lower() == cat.lower()]
        if subset.empty:
            records.append(
                {
                    "category": cat,
                    "cdi_category": str(row.get("cdi_category", "")),
                    "tsne_x": float(row["tsne_x"]),
                    "tsne_y": float(row["tsne_y"]),
                    "exemplar_count": int(row.get("exemplar_count", 0)),
                    "samples": [],
                }
            )
            continue

        if len(subset) > args.samples_per_category:
            sampled_idx = rng.sample(list(subset.index), args.samples_per_category)
            chosen = subset.loc[sampled_idx].copy()
        else:
            chosen = subset.copy()

        samples = []
        for i, exr in chosen.iterrows():
            stem = str(exr["stem"]).strip()
            crop_path = find_crop_file(args.cropped_dir, cat, stem)
            if crop_path is None:
                missing += 1
                continue

            ext = crop_path.suffix.lower()
            thumb_name = f"{cat}_{i}{ext}".replace(" ", "_")
            dest = thumbs_dir / thumb_name
            if not dest.exists():
                shutil.copy2(crop_path, dest)
                copied += 1

            samples.append(
                {
                    "stem": stem,
                    "subject_id": str(exr["subject_id"]),
                    "age_mo": int(exr["age_mo"]),
                    "dist_to_things_centroid": float(exr["dist_to_things_centroid"]),
                    "dist_to_bv_centroid": float(exr["dist_to_bv_centroid"]),
                    "thumb_path": f"thumbs/{thumb_name}",
                }
            )

        if emb_root is not None and samples:
            _n_emb, n_coord = attach_within_category_tsne(
                cat, samples, emb_root, args.seed, tsne_max_iter=args.within_tsne_max_iter
            )
            if n_coord >= 2:
                within_tsne_categories += 1
                within_tsne_exemplars += n_coord

        records.append(
            {
                "category": cat,
                "cdi_category": str(row.get("cdi_category", "")),
                "tsne_x": float(row["tsne_x"]),
                "tsne_y": float(row["tsne_y"]),
                "exemplar_count": int(row.get("exemplar_count", len(subset))),
                "samples": samples,
            }
        )

    payload = {
        "meta": {
            "tsne_csv": str(args.tsne_csv),
            "per_exemplar_csv": str(args.per_exemplar_csv),
            "cropped_dir": str(args.cropped_dir),
            "embeddings_dir": str(emb_root) if emb_root is not None else None,
            "within_tsne": bool(emb_root is not None and not args.no_within_tsne),
            "within_tsne_categories": within_tsne_categories,
            "within_tsne_exemplars": within_tsne_exemplars,
            "samples_per_category": args.samples_per_category,
            "copied_thumbnails": copied,
            "missing_crops": missing,
        },
        "points": records,
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Copied thumbnails: {copied}")
    print(f"Missing crop matches: {missing}")


if __name__ == "__main__":
    main()
