#!/usr/bin/env python3
"""Z-score THINGS category embeddings from BabyDINOv3 per-image .npy trees.

Mirrors the BabyDINOv3 block in ``07_things_exemplar_set_zscore_embeddings.ipynb``.
Run after ``create_babydinov3_things_embeddings.py``.

  BV_CATEGORY_SET=valid129 python scripts/things_exemplar_set_zscore_babydinov3.py
  BV_CATEGORY_SET=valid85 python scripts/things_exemplar_set_zscore_babydinov3.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import MANUSCRIPT_DIR, PREPRINT_DIR, PROJECT_ROOT, SCRIPTS_DIR
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATION_DIR = PROJECT_ROOT / "annotation"
OUT_ROOT = PREPRINT_DIR / "exemplar_set_embeddings"

CATEGORY_FILES = {
    "valid85": DATA_DIR / "included_categories_valid85.txt",
    "valid129": DATA_DIR / "included_categories_valid129.txt",
}
PER_CLASS_PRECISION_CSV = ANNOTATION_DIR / "per_class_validation_data.csv"
PER_FILE_PRECISION_CSV = ANNOTATION_DIR / "per_file_precision_data.csv"
SAMPLED_EXEMPLAR_CSV = ANNOTATION_DIR / (
    "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_babyview_only.csv"
)

BABYDINOV3_CHECKPOINT_STEP = os.environ.get("BV_BABYDINOV3_CHECKPOINT_STEP", "119999").strip()
_default_bd3 = Path(
    "/data2/dataset/babyview/868_hours/outputs/things_babydinov3_grad_accum_1"
) / f"step_{BABYDINOV3_CHECKPOINT_STEP}"
THINGS_BABYDINOV3_DIR = Path(
    os.environ.get("THINGS_BABYDINOV3_EMBEDDINGS_DIR", str(_default_bd3))
).expanduser()

PRECISION_THRESHOLD = float(os.environ.get("BV_PRECISION_THRESHOLD", "0.6"))
CROP_PREFIX = os.environ.get("THINGS_CROP_PATH_PREFIX", "").strip()
CROP_PREFIX_NEW = os.environ.get("THINGS_CROP_PATH_PREFIX_NEW", "").strip()


def resolve_category_subdir(embed_root: Path, cat: str) -> Path | None:
    direct = embed_root / cat
    if direct.is_dir():
        return direct
    for p in embed_root.iterdir():
        if p.is_dir() and p.name.lower() == cat.lower():
            return p
    return None


def load_included_categories(txt_path: Path) -> list[str]:
    return [line.strip().lower() for line in txt_path.read_text().splitlines() if line.strip()]


def load_valid_classes(per_class_csv: Path, threshold: float) -> set[str]:
    df = pd.read_csv(per_class_csv, usecols=["class", "precision"])
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    return set(df.loc[df["precision"] > threshold, "class"])


def load_valid_pairs(per_file_csv: Path, threshold: float) -> set[tuple[str, str]]:
    df = pd.read_csv(per_file_csv, usecols=["filename", "class", "precision"])
    df = df[df["precision"] > threshold].copy()
    df["class_norm"] = df["class"].astype(str).str.strip().str.lower()
    df["stem"] = (
        df["filename"]
        .astype(str)
        .str.strip()
        .str.rsplit("/", n=1)
        .str[-1]
        .str.rsplit(".", n=1)
        .str[0]
        .str.lower()
    )
    return set(zip(df["class_norm"], df["stem"]))


def remap_absolute_path(p: Path) -> Path:
    s = str(p)
    if CROP_PREFIX and CROP_PREFIX_NEW and s.startswith(CROP_PREFIX):
        return Path(CROP_PREFIX_NEW + s[len(CROP_PREFIX) :])
    return p


def build_valid85_sampled_exemplar_table(
    included_txt: Path,
    per_class_csv: Path,
    per_file_csv: Path,
    sampled_csv: Path,
    precision_threshold: float,
) -> pd.DataFrame:
    included = set(load_included_categories(included_txt))
    valid_classes = load_valid_classes(per_class_csv, precision_threshold)
    eligible_cats = included & valid_classes
    valid_pairs = load_valid_pairs(per_file_csv, precision_threshold)

    sampled = pd.read_csv(sampled_csv)
    sampled = sampled[sampled["trial_type"] == "regular"].copy()
    sampled["category"] = sampled["category"].astype(str).str.strip().str.lower()
    sampled["stem"] = sampled["stem"].astype(str).str.strip().str.lower()

    mask = sampled.apply(lambda r: (r["category"], r["stem"]) in valid_pairs, axis=1)
    sampled = sampled.loc[mask].copy()
    sampled = sampled[sampled["category"].isin(eligible_cats)].copy()
    return sampled[["category", "path", "stem"]].reset_index(drop=True)


def collect_things_npy_paths(
    embed_root: Path,
    allowed_categories_lower: set[str],
) -> tuple[dict[str, list[Path]], dict]:
    out: dict[str, list[Path]] = {}
    stats = {
        "model": "babydinov3",
        "n_categories_requested": len(allowed_categories_lower),
        "n_categories_with_subdir": 0,
        "n_npy_files": 0,
    }
    for cat in sorted(allowed_categories_lower):
        cdir = resolve_category_subdir(embed_root, cat)
        if cdir is None:
            out[cat] = []
            continue
        stats["n_categories_with_subdir"] += 1
        paths = sorted(
            (remap_absolute_path(p) for p in cdir.rglob("*.npy") if p.is_file()),
            key=lambda x: str(x).lower(),
        )
        stats["n_npy_files"] += len(paths)
        out[cat] = paths
    return out, stats


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


def average_precomputed_embeddings_for_categories(
    paths_by_cat: dict[str, list[Path]],
) -> tuple[list[str], np.ndarray, dict]:
    categories = sorted(paths_by_cat.keys())
    diag: dict = {"missing_paths": [], "n_paths": {}, "skipped_categories": []}
    cat_vectors: list[np.ndarray] = []
    categories_out: list[str] = []

    for cat in categories:
        paths_in_order = list(dict.fromkeys(paths_by_cat[cat]))
        vecs: list[np.ndarray] = []
        miss: list[str] = []
        for p in paths_in_order:
            v = load_normalized_npy_vector(p)
            if v is None:
                miss.append(str(remap_absolute_path(p)))
            else:
                vecs.append(v)
        diag["missing_paths"].extend(miss[:20])
        diag["n_paths"][cat] = {"total": len(paths_in_order), "found": len(vecs)}
        if not vecs:
            diag["skipped_categories"].append(cat)
            continue
        cat_vectors.append(np.mean(np.stack(vecs, axis=0), axis=0))
        categories_out.append(cat)

    X = np.stack(cat_vectors, axis=0) if cat_vectors else np.zeros((0, 0))
    return categories_out, X, diag


def zscore_rows(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    return (X - mu) / (sig + eps)


def save_cat_emb(csv_path: Path, cats: list[str], X: np.ndarray) -> None:
    dim_cols = [f"dim_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=dim_cols)
    df.insert(0, "category", cats)
    df.to_csv(csv_path, index=False)


def run_category_set(category_set: str) -> None:
    if not THINGS_BABYDINOV3_DIR.is_dir():
        raise FileNotFoundError(
            f"BabyDINOv3 THINGS embeddings dir not found: {THINGS_BABYDINOV3_DIR}"
        )

    inc_path = CATEGORY_FILES[category_set]
    if category_set == "valid129":
        allowed = set(load_included_categories(inc_path))
        exemplar_source = "things_babydinov3_npy_trees_valid129_inclusion"
    elif category_set == "valid85":
        exemplar_df = build_valid85_sampled_exemplar_table(
            inc_path,
            PER_CLASS_PRECISION_CSV,
            PER_FILE_PRECISION_CSV,
            SAMPLED_EXEMPLAR_CSV,
            PRECISION_THRESHOLD,
        )
        allowed = set(exemplar_df["category"].astype(str).str.strip().str.lower())
        exemplar_source = "things_babydinov3_npy_trees_valid85_category_set_same_as_bv06"
    else:
        raise ValueError(category_set)

    paths_by_cat, stats = collect_things_npy_paths(THINGS_BABYDINOV3_DIR, allowed)
    inc_all = sorted(allowed)
    paths_by_cat = {c: paths_by_cat.get(c, []) for c in inc_all}
    cats_with = [c for c in inc_all if paths_by_cat[c]]
    paths_by_cat = {c: paths_by_cat[c] for c in cats_with}

    print(f"\n=== {category_set} (THINGS BabyDINOv3) ===")
    print("embed dir:", THINGS_BABYDINOV3_DIR)
    print("stats:", stats)
    print("categories with >=1 npy:", len(cats_with), "/", len(inc_all))

    cats, X_raw, diag = average_precomputed_embeddings_for_categories(paths_by_cat)
    X_z = zscore_rows(X_raw)

    out_dir = OUT_ROOT / category_set
    out_dir.mkdir(parents=True, exist_ok=True)
    save_cat_emb(
        out_dir / f"things_babydinov3_exemplar_avg_zscore_within_{category_set}.csv",
        cats,
        X_z,
    )
    save_cat_emb(
        out_dir / f"things_babydinov3_exemplar_avg_raw_within_{category_set}.csv",
        cats,
        X_raw,
    )

    meta = {
        "dataset": "things",
        "model": "babydinov3",
        "category_set": category_set,
        "exemplar_source": exemplar_source,
        "included_categories_txt": str(inc_path),
        "n_categories": len(cats),
        "babydinov3_checkpoint_step": BABYDINOV3_CHECKPOINT_STEP,
        "things_babydinov3_embeddings_dir": str(THINGS_BABYDINOV3_DIR),
        "babydinov3_stats": stats,
        "categories_alphabetical": cats,
        "diagnostics_babydinov3": {k: v for k, v in diag.items() if k != "missing_paths"},
        "n_missing_paths_reported": len(diag.get("missing_paths", [])),
    }
    (out_dir / "things_babydinov3_exemplar_embedding_run.json").write_text(
        json.dumps(meta, indent=2)
    )
    print("Wrote:", out_dir / f"things_babydinov3_exemplar_avg_zscore_within_{category_set}.csv")
    print("Wrote:", out_dir / f"things_babydinov3_exemplar_avg_raw_within_{category_set}.csv")


def main() -> None:
    category_set = os.environ.get("BV_CATEGORY_SET", "valid129").strip()
    if category_set == "all":
        for s in ("valid129", "valid85"):
            run_category_set(s)
    else:
        run_category_set(category_set)


if __name__ == "__main__":
    main()
