#!/usr/bin/env python3
"""
Within-category kNN diversity for BV exemplars in CLIP/DinoV3 space.

For each category we have N BV exemplars. For each exemplar we compute the mean
distance to its k nearest neighbors (within that category, excluding self).
Interpretation:
- **Low mean kNN distance** → micro-structure: exemplars form local clusters
  (e.g. every 5–10 exemplars are similar in views, format, etc.), so each point
  has close neighbors.
- **High mean kNN distance** → no consistent local structure; exemplars are
  more uniformly spread, so even the k nearest are relatively far.

So we get a category-level "kNN diversity" (e.g. mean over exemplars of their
mean-kNN-distance). Lower = more micro-structure; higher = more uniform spread.

Outputs:
1. Per-category summary CSV: category, n_exemplars, k, mean_knn_dist, std_knn_dist,
   median_knn_dist, std_of_exemplar_means (optional).
2. Optional per-exemplar CSV (per k): category, subject_id, age_mo, mean_knn_dist, rank_within_cat.

Interpretation (summary metrics):
- mean_knn_dist: Mean over exemplars of (mean distance to k nearest neighbors). Lower = more
  local clustering (micro-structure); higher = more uniform spread within category.
- std_knn_dist: Std of those per-exemplar mean-kNN distances. High std = some exemplars have
  very close neighbors, others don't (e.g. a few tight clusters plus outliers).
- median_knn_dist: Robust center; use with mean to spot skew (e.g. a few far outliers).

Compare with centroid-based spread (mean_bv_to_bv_centroid): a category can have large
overall spread but low mean_knn_dist (micro-structure: clusters of similar exemplars), or
high spread and high mean_knn_dist (no local consistency). See visualize_bv_knn_diversity.py.

Usage:
  cd analysis/manuscript-2026/exemplar_variability_analyses
  python bv_within_category_knn_diversity.py --embedding clip --k 5
  python bv_within_category_knn_diversity.py --embedding clip --k 5 10 --save-exemplar-csv
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse loaders from centroid script (run from same dir)
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    from bv_to_things_centroid_distances import (
        load_allowed_categories,
        load_bv_embeddings,
        EXCLUDED_SUBJECT,
        MIN_EXEMPLARS,
    )
except ImportError:
    # Fallback if run from elsewhere: duplicate minimal config
    GROUPED_EMBEDDINGS_BASE = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings")
    GROUPED_EMBEDDINGS_DIRS = {
        "clip": GROUPED_EMBEDDINGS_BASE / "clip_embeddings_grouped_by_age-mo_normalized",
        "dinov3": GROUPED_EMBEDDINGS_BASE / "facebook_dinov3-vitb16-pretrain-lvd1689m_grouped_by_age-mo_normalized",
    }
    CATEGORIES_FILE = SCRIPT_DIR / "../../../data/things_bv_overlap_categories_exclude_zero_precisions.txt"
    EXCLUDED_SUBJECT = "00270001"
    MIN_EXEMPLARS = 2

    def load_allowed_categories():
        if not CATEGORIES_FILE.exists():
            return None
        with open(CATEGORIES_FILE) as f:
            return set(line.strip() for line in f if line.strip())

    def load_bv_embeddings(embedding_type: str, allowed_categories, excluded_subject, min_exemplars=2):
        grouped_dir = GROUPED_EMBEDDINGS_DIRS[embedding_type]
        if not grouped_dir.exists():
            raise FileNotFoundError(f"BV grouped dir not found: {grouped_dir}")
        category_embeddings = {}
        category_exemplar_ids = {}
        for cat_folder in sorted(grouped_dir.iterdir()):
            if not cat_folder.is_dir():
                continue
            cat_name = cat_folder.name
            if allowed_categories is not None and cat_name not in allowed_categories:
                continue
            embs, ids = [], []
            for f in cat_folder.glob("*.npy"):
                stem = f.stem
                parts = stem.split("_")
                if len(parts) < 2:
                    continue
                subject_id, age_mo = parts[0], None
                try:
                    age_mo = int(parts[1])
                except ValueError:
                    continue
                if excluded_subject and subject_id == excluded_subject:
                    continue
                try:
                    e = np.load(f)
                    e = np.asarray(e, dtype=np.float64).flatten()
                    embs.append(e)
                    ids.append((subject_id, age_mo))
                except Exception:
                    continue
            if len(embs) >= min_exemplars:
                category_embeddings[cat_name] = np.array(embs)
                category_exemplar_ids[cat_name] = ids
        return category_embeddings, category_exemplar_ids


def compute_knn_mean_distances(X: np.ndarray, k: int):
    """
    For each row in X (n, dim), compute mean L2 distance to its k nearest
    neighbors (excluding self). Uses k+1 neighbors and drops the first (self).

    X: (n, dim). Must have n >= k+1.
    Returns: (n,) array of mean kNN distances.
    """
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    n_neighbors = min(k + 1, n)  # self + up to k others
    if n_neighbors < 2:
        return np.full(n, np.nan)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)  # (n, n_neighbors); first column is 0 (self)
    # Mean distance to neighbors excluding self (columns 1:)
    mean_knn = np.mean(distances[:, 1:], axis=1)
    return mean_knn


def run_knn_per_category(bv_embeddings, bv_ids, categories, k: int):
    """
    For each category, compute per-exemplar mean kNN distance, then aggregate.
    Returns (summary_rows, exemplar_rows).
    """
    summary_rows = []
    exemplar_rows = []
    for cat in tqdm(categories, desc="kNN per category"):
        X = bv_embeddings[cat]
        id_list = bv_ids[cat]
        n = X.shape[0]
        effective_k = min(k, n - 1)
        if effective_k < 1:
            summary_rows.append({
                "category": cat,
                "n_exemplars": n,
                "k": k,
                "effective_k": 0,
                "mean_knn_dist": np.nan,
                "std_knn_dist": np.nan,
                "median_knn_dist": np.nan,
            })
            continue
        mean_knn = compute_knn_mean_distances(X, effective_k)
        summary_rows.append({
            "category": cat,
            "n_exemplars": n,
            "k": k,
            "effective_k": effective_k,
            "mean_knn_dist": float(np.nanmean(mean_knn)),
            "std_knn_dist": float(np.nanstd(mean_knn)),
            "median_knn_dist": float(np.nanmedian(mean_knn)),
        })
        for i, (sid, age_mo) in enumerate(id_list):
            exemplar_rows.append({
                "category": cat,
                "subject_id": sid,
                "age_mo": age_mo,
                "mean_knn_dist": float(mean_knn[i]),
            })
    return summary_rows, exemplar_rows


def main():
    parser = argparse.ArgumentParser(
        description="Within-category kNN diversity: mean distance to k nearest neighbors per exemplar"
    )
    parser.add_argument("--embedding", type=str, default="clip", choices=("clip", "dinov3"))
    parser.add_argument("--k", type=int, nargs="+", default=[5],
                        help="k for kNN (e.g. 5 10). Default 5.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: script dir)")
    parser.add_argument("--save-exemplar-csv", action="store_true",
                        help="Save per-exemplar mean kNN distance CSV (for largest k only)")
    args = parser.parse_args()

    out_dir = args.out_dir or SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed = load_allowed_categories()
    print(f"Using {len(allowed) if allowed else 'all'} categories")

    print("Loading BV embeddings...")
    bv_embeddings, bv_ids = load_bv_embeddings(
        args.embedding,
        allowed_categories=allowed,
        excluded_subject=EXCLUDED_SUBJECT,
        min_exemplars=MIN_EXEMPLARS,
    )
    categories = sorted(bv_embeddings.keys())
    print(f"Categories: {len(categories)}")

    all_summary = []
    for k in args.k:
        prefix = f"bv_within_category_knn_{args.embedding}_k{k}"
        summary_rows, exemplar_rows = run_knn_per_category(
            bv_embeddings, bv_ids, categories, k
        )
        summary_df = pd.DataFrame(summary_rows)
        # Sort by mean_knn_dist ascending: lowest first = most micro-structure
        summary_df = summary_df.sort_values("mean_knn_dist", ascending=True).reset_index(drop=True)
        summary_path = out_dir / f"{prefix}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved {prefix}: {summary_path}")

        if args.save_exemplar_csv:
            exemplar_df = pd.DataFrame(exemplar_rows)
            # Optional: add rank within category (1 = lowest mean_knn_dist = most local structure)
            exemplar_df["rank_within_cat"] = exemplar_df.groupby("category")["mean_knn_dist"].rank(
                method="first", ascending=True
            ).astype(int)
            exemplar_path = out_dir / f"{prefix}_per_exemplar.csv"
            exemplar_df.to_csv(exemplar_path, index=False)
            print(f"Saved per-exemplar: {exemplar_path}")

        all_summary.append(summary_df.assign(k_used=k))

    # Single combined summary for multiple k (if more than one)
    if len(args.k) > 1:
        combined = pd.concat(all_summary, ignore_index=True)
        combined_path = out_dir / f"bv_within_category_knn_{args.embedding}_multi_k_summary.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined multi-k summary: {combined_path}")

    print("Done. Lower mean_knn_dist = more micro-structure (local clustering); higher = more uniform spread.")


if __name__ == "__main__":
    main()
