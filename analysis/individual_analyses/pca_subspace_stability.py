"""
PCA subspace stability analysis.

Addresses: How much is across-family similarity of individual subject RDMs
driven by the embedding space vs. the actual visual input? We restrict to the
subspace spanned by the observed object crops (age-month level embeddings), fit
PCA, project all embeddings onto this subspace, then recompute individual
subject RDMs (one per subject, aggregate across age) and across-subject
similarity (mean pairwise RDM correlation) in the projected space.

This compares individual subject RDMs across subjects/families (not developmental
trajectory younger vs older). Same category order and overlapping categories only
per pair.

Usage:
  cd analysis/individual_analyses
  python pca_subspace_stability.py --embeddings_dir <path> [--n_components 0.95] [--output_dir .]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from tqdm import tqdm


def load_embeddings_by_age(embeddings_dir: Path, allowed_categories: set[str] | None,
                          excluded_subject: str | None) -> dict:
    """
    Load pre-normalized age-month level embeddings.
    Returns: subject_age_embeddings[subject_id][age_mo][category] = embedding (1D array).
    """
    embeddings_dir = Path(embeddings_dir)
    subject_age_embeddings = defaultdict(lambda: defaultdict(dict))

    category_folders = sorted([f for f in embeddings_dir.iterdir() if f.is_dir()])
    if allowed_categories is not None:
        category_folders = [f for f in category_folders if f.name in allowed_categories]

    for category_folder in tqdm(category_folders, desc="Loading categories"):
        category = category_folder.name
        for emb_file in category_folder.glob("*.npy"):
            stem = emb_file.stem
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            subject_id = parts[0]
            if excluded_subject and subject_id == excluded_subject:
                continue
            try:
                age_mo = int(parts[1])
            except ValueError:
                continue
            try:
                emb = np.load(emb_file)
                if emb.ndim > 1:
                    emb = emb.flatten()
                subject_age_embeddings[subject_id][age_mo][category] = emb
            except Exception:
                continue

    return dict(subject_age_embeddings)


def stack_all_embeddings(subject_age_embeddings: dict) -> np.ndarray:
    """Stack all age-month level embeddings into (n_observations, n_dims)."""
    rows = []
    for _sid, age_data in subject_age_embeddings.items():
        for _age_mo, cat_dict in age_data.items():
            for _cat, emb in cat_dict.items():
                rows.append(emb)
    return np.array(rows, dtype=np.float64)


def aggregate_embeddings_per_subject(subject_age_embeddings: dict) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate per subject across all age_mo: one embedding per (subject, category). Like notebook 06."""
    out = {}
    for subject_id, age_data in subject_age_embeddings.items():
        category_embeddings = defaultdict(list)
        for _age_mo, cat_dict in age_data.items():
            for cat, emb in cat_dict.items():
                category_embeddings[cat].append(emb)
        out[subject_id] = {
            cat: np.mean(embs, axis=0)
            for cat, embs in category_embeddings.items()
            if embs
        }
    return out


def aggregate_embeddings_by_bin(
    age_data: dict[int, dict[str, np.ndarray]],
    median_age: float,
    bin_name: str,
) -> dict[str, np.ndarray]:
    """Aggregate embeddings for one subject into one bin: younger (age_mo <= median) or older (age_mo > median)."""
    if bin_name == "younger":
        ages_to_use = {a: d for a, d in age_data.items() if a <= median_age}
    else:
        ages_to_use = {a: d for a, d in age_data.items() if a > median_age}
    if not ages_to_use:
        return {}
    category_embeddings = defaultdict(list)
    for _age_mo, cat_dict in ages_to_use.items():
        for cat, emb in cat_dict.items():
            category_embeddings[cat].append(emb)
    return {cat: np.mean(embs, axis=0) for cat, embs in category_embeddings.items() if embs}


def compute_rdm_for_bin(bin_embeddings_dict: dict, ordered_categories: list,
                        min_categories: int) -> tuple[np.ndarray | None, list]:
    """
    Compute RDM for one subject/bin using a fixed category order.
    RDM row/column i always corresponds to ordered_categories[i], so all subjects
    have RDMs in the same exact order (same ordered_categories list must be passed).
    Returns (rdm, available_categories) or (None, []).
    """
    # Preserve canonical order: only include categories that exist, in ordered_categories order
    available = [c for c in ordered_categories if c in bin_embeddings_dict]
    if len(available) < min_categories:
        return None, available
    n_cat = len(ordered_categories)
    matrix = np.array([bin_embeddings_dict[c].flatten() for c in available])
    sim = cosine_similarity(matrix)
    dist = 1 - sim
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    rdm = np.full((n_cat, n_cat), np.nan)
    for i, c_i in enumerate(available):
        idx_i = ordered_categories.index(c_i)
        for j, c_j in enumerate(available):
            idx_j = ordered_categories.index(c_j)
            rdm[idx_i, idx_j] = dist[i, j]
    return rdm, available


def compute_rdm_correlation(rdm1: np.ndarray, rdm2: np.ndarray, ordered_categories: list,
                            cats1: list, cats2: list) -> tuple[float, int]:
    """
    Spearman correlation between two RDMs on common categories (upper triangle).
    Assumes both RDMs were built with the same ordered_categories so that
    rdm[k, l] corresponds to (ordered_categories[k], ordered_categories[l]).
    """
    # Common categories in the same canonical order (order of ordered_categories)
    common = [c for c in ordered_categories if c in cats1 and c in cats2]
    if len(common) < 2:
        return np.nan, len(common)
    indices = [ordered_categories.index(c) for c in common]
    sub1 = rdm1[np.ix_(indices, indices)]
    sub2 = rdm2[np.ix_(indices, indices)]
    mask = np.triu(np.ones_like(sub1, dtype=bool), k=1)
    d1 = sub1[mask]
    d2 = sub2[mask]
    valid = ~(np.isnan(d1) | np.isnan(d2))
    d1, d2 = d1[valid], d2[valid]
    if len(d1) == 0:
        return np.nan, len(common)
    corr, _ = spearmanr(d1, d2)
    return corr, len(common)


def build_subject_rdms(
    subject_embeddings: dict[str, dict[str, np.ndarray]],
    ordered_categories: list,
    min_categories: int,
) -> dict[str, tuple[np.ndarray, list]]:
    """
    Build one RDM per subject using the same ordered_categories for all.
    Every RDM has shape (len(ordered_categories), len(ordered_categories));
    row/column i is ordered_categories[i]. Returns subject_id -> (rdm, available_categories).
    """
    n_cat = len(ordered_categories)
    result = {}
    for subject_id, cat_embeddings in subject_embeddings.items():
        rdm, cats = compute_rdm_for_bin(cat_embeddings, ordered_categories, min_categories)
        if rdm is not None:
            assert rdm.shape == (n_cat, n_cat), f"RDM shape {rdm.shape} != ({n_cat}, {n_cat})"
            result[subject_id] = (rdm, cats)
    return result


def mean_pairwise_rdm_correlation(
    subject_rdms: dict[str, tuple[np.ndarray, list]],
    ordered_categories: list,
) -> tuple[float, list[dict]]:
    """Mean Spearman correlation between all pairs of subject RDMs (common categories per pair)."""
    subject_ids = sorted(subject_rdms.keys())
    pair_results = []
    correlations = []
    for i in range(len(subject_ids)):
        for j in range(i + 1, len(subject_ids)):
            sid_i, sid_j = subject_ids[i], subject_ids[j]
            rdm_i, cats_i = subject_rdms[sid_i]
            rdm_j, cats_j = subject_rdms[sid_j]
            corr, n_common = compute_rdm_correlation(rdm_i, rdm_j, ordered_categories, cats_i, cats_j)
            if not np.isnan(corr):
                correlations.append(corr)
                pair_results.append({"subject_id_1": sid_i, "subject_id_2": sid_j, "correlation": corr, "n_common_categories": n_common})
    mean_corr = float(np.mean(correlations)) if correlations else np.nan
    return mean_corr, pair_results


def project_embeddings(subject_age_embeddings: dict, pca: PCA) -> dict:
    """Project all embeddings using fitted PCA (transform)."""
    projected = defaultdict(lambda: defaultdict(dict))
    for sid, age_data in subject_age_embeddings.items():
        for age_mo, cat_dict in age_data.items():
            for cat, emb in cat_dict.items():
                proj = pca.transform(emb.reshape(1, -1)).flatten()
                projected[sid][age_mo][cat] = proj
    return dict(projected)


def run_pca_stability(
    embeddings_dir: Path,
    categories_file: Path | None,
    category_order_file: Path | None,
    output_dir: Path,
    excluded_subject: str = "00270001",
    min_categories_per_age_bin: int = 8,
    n_components: float | int = 0.95,
) -> pd.DataFrame:
    """
    Run full pipeline: load -> PCA on observed embeddings -> project -> recompute RDMs -> stability.
    Returns trajectory DataFrame with columns including rdm_correlation (in PCA subspace).
    """
    embeddings_dir = Path(embeddings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_categories = None
    if categories_file and categories_file.exists():
        with open(categories_file) as f:
            allowed_categories = set(line.strip() for line in f if line.strip() and not line.strip().startswith("#"))

    ordered_categories = None
    if category_order_file and category_order_file.exists():
        with open(category_order_file) as f:
            ordered_categories = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if allowed_categories:
            ordered_categories = [c for c in ordered_categories if c in allowed_categories]

    if not embeddings_dir.exists():
        raise FileNotFoundError(
            f"Embeddings directory not found: {embeddings_dir}\n"
            "Set --embeddings_dir to the path to your normalized age-month embeddings\n"
            "(e.g. .../clip_embeddings_grouped_by_age-mo_normalized from notebook 05 output)."
        )
    print("Loading embeddings...")
    subject_age_embeddings = load_embeddings_by_age(
        embeddings_dir, allowed_categories, excluded_subject
    )
    if not subject_age_embeddings:
        raise ValueError("No embeddings loaded.")

    # Build ordered category list if not from file (deterministic: sorted)
    if ordered_categories is None:
        all_cats = set()
        for age_data in subject_age_embeddings.values():
            for cat_dict in age_data.values():
                all_cats.update(cat_dict.keys())
        ordered_categories = sorted(all_cats)

    # Save canonical category order so RDMs are reproducible and order is explicit
    with open(output_dir / "ordered_categories.txt", "w") as f:
        for c in ordered_categories:
            f.write(c + "\n")
    print(f"Using {len(ordered_categories)} categories in fixed order (saved to ordered_categories.txt)")

    # Stack all observations and fit PCA
    X = stack_all_embeddings(subject_age_embeddings)
    print(f"Stacked shape: {X.shape} (n_observations, n_dims)")

    if isinstance(n_components, float) and 0 < n_components < 1:
        pca = PCA(n_components=n_components, random_state=42)
    else:
        pca = PCA(n_components=int(n_components), random_state=42)
    pca.fit(X)
    n_comp = pca.n_components_
    print(f"PCA: n_components={n_comp}, explained variance ratio (sum)={pca.explained_variance_ratio_.sum():.4f}")

    # Save PCA summary
    np.save(output_dir / "pca_mean.npy", pca.mean_)
    np.save(output_dir / "pca_components.npy", pca.components_)
    pd.DataFrame({
        "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(output_dir / "pca_variance.csv", index=False)

    # Project all embeddings
    projected = project_embeddings(subject_age_embeddings, pca)

    # Individual subject RDMs: aggregate per subject across all age_mo (like notebook 06)
    subject_embeddings_full = aggregate_embeddings_per_subject(subject_age_embeddings)
    subject_embeddings_pca = aggregate_embeddings_per_subject(projected)

    subject_rdms_full = build_subject_rdms(
        subject_embeddings_full, ordered_categories, min_categories_per_age_bin
    )
    subject_rdms_pca = build_subject_rdms(
        subject_embeddings_pca, ordered_categories, min_categories_per_age_bin
    )

    # Across-subject similarity: mean pairwise RDM correlation (common categories per pair)
    mean_corr_full, pair_results_full = mean_pairwise_rdm_correlation(subject_rdms_full, ordered_categories)
    mean_corr_pca, pair_results_pca = mean_pairwise_rdm_correlation(subject_rdms_pca, ordered_categories)

    summary_df = pd.DataFrame([
        {"space": "full", "mean_pairwise_rdm_correlation": mean_corr_full, "n_subjects": len(subject_rdms_full), "n_pairs": len(pair_results_full)},
        {"space": "pca_subspace", "mean_pairwise_rdm_correlation": mean_corr_pca, "n_subjects": len(subject_rdms_pca), "n_pairs": len(pair_results_pca)},
    ])
    summary_df.to_csv(output_dir / "across_subject_rdm_similarity_summary.csv", index=False)
    pd.DataFrame(pair_results_full).to_csv(output_dir / "pairwise_rdm_correlations_full.csv", index=False)
    pd.DataFrame(pair_results_pca).to_csv(output_dir / "pairwise_rdm_correlations_pca_subspace.csv", index=False)

    # Per-subject trajectory: younger vs older RDM correlation in PCA subspace (median age split)
    all_ages = []
    for age_data in subject_age_embeddings.values():
        all_ages.extend(age_data.keys())
    overall_median_age = float(np.median(all_ages))
    trajectory_rows = []
    for subject_id, age_data in projected.items():
        younger_agg = aggregate_embeddings_by_bin(age_data, overall_median_age, "younger")
        older_agg = aggregate_embeddings_by_bin(age_data, overall_median_age, "older")
        rdm_younger, cats_younger = compute_rdm_for_bin(
            younger_agg, ordered_categories, min_categories_per_age_bin
        )
        rdm_older, cats_older = compute_rdm_for_bin(
            older_agg, ordered_categories, min_categories_per_age_bin
        )
        if rdm_younger is None or rdm_older is None:
            continue
        corr, n_common = compute_rdm_correlation(
            rdm_younger, rdm_older, ordered_categories, cats_younger, cats_older
        )
        trajectory_rows.append({
            "subject_id": subject_id,
            "age_bin_1": "younger",
            "age_bin_2": "older",
            "median_age_threshold": overall_median_age,
            "rdm_correlation": corr,
            "n_common_categories": n_common,
            "n_categories_younger": len(cats_younger),
            "n_categories_older": len(cats_older),
        })
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_df.to_csv(output_dir / "trajectory_correlations_pca_subspace.csv", index=False)
    print(f"Saved {output_dir / 'trajectory_correlations_pca_subspace.csv'} ({len(trajectory_df)} subjects)")

    print(f"Saved {output_dir / 'across_subject_rdm_similarity_summary.csv'}")
    print(f"  Full space: mean pairwise RDM correlation = {mean_corr_full:.4f} (n_subjects={len(subject_rdms_full)}, n_pairs={len(pair_results_full)})")
    print(f"  PCA subspace: mean pairwise RDM correlation = {mean_corr_pca:.4f} (n_subjects={len(subject_rdms_pca)}, n_pairs={len(pair_results_pca)})")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Stability in PCA subspace of observed embeddings")
    parser.add_argument(
        "--embeddings_dir",
        type=Path,
        required=True,
        help="Path to normalized age-month embeddings (notebook 05 output), e.g. .../clip_embeddings_grouped_by_age-mo_normalized",
    )
    parser.add_argument(
        "--categories_file",
        type=Path,
        default=Path("../../data/things_bv_overlap_categories_exclude_zero_precisions.txt"),
        help="Optional: categories to include (one per line)",
    )
    parser.add_argument(
        "--category_order_file",
        type=Path,
        default=Path("../vss-2026/bv_things_comp_12252025/bv_clip_filtered_zscored_hierarchical_163cats/category_order_reorganized.txt"),
        help="Predefined category order for RDM (same as notebook 07)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: individual_subject_rdms_<clip|dinov3>_pca_subspace)",
    )
    parser.add_argument(
        "--excluded_subject",
        default="00270001",
        help="Subject ID to exclude",
    )
    parser.add_argument(
        "--min_categories_per_subject",
        type=int,
        default=8,
        help="Minimum categories per subject to build RDM",
    )
    parser.add_argument(
        "--n_components",
        default="0.95",
        help="PCA: fraction of variance (e.g. 0.95) or integer number of components",
    )
    args = parser.parse_args()

    try:
        n_comp = float(args.n_components)
        if n_comp >= 1:
            n_comp = int(n_comp)
    except ValueError:
        n_comp = int(args.n_components)

    if args.output_dir is None:
        emb_str = str(args.embeddings_dir).lower()
        emb_type = "dinov3" if "dinov" in emb_str else "clip"
        args.output_dir = Path(f"individual_subject_rdms_{emb_type}_pca_subspace")

    script_dir = Path(__file__).resolve().parent
    embeddings_dir = Path(args.embeddings_dir).resolve() if args.embeddings_dir.is_absolute() else (script_dir / args.embeddings_dir).resolve()
    categories_file = Path(args.categories_file) if args.categories_file.is_absolute() else script_dir / args.categories_file
    category_order_file = Path(args.category_order_file) if args.category_order_file.is_absolute() else script_dir / args.category_order_file
    output_dir = Path(args.output_dir).resolve() if args.output_dir.is_absolute() else (script_dir / args.output_dir).resolve()

    run_pca_stability(
        embeddings_dir=embeddings_dir,
        categories_file=categories_file,
        category_order_file=category_order_file,
        output_dir=output_dir,
        excluded_subject=args.excluded_subject,
        min_categories_per_age_bin=args.min_categories_per_subject,
        n_components=n_comp,
    )


if __name__ == "__main__":
    main()
