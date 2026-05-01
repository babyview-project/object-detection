from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

DEFAULT_CATEGORY_SET = "valid85"
DEFAULT_N_TOP = 8

INCLUDED_CATEGORIES_TXT = PROJECT_ROOT / "data" / "included_categories_valid85.txt"
SAMPLED_EXEMPLAR_CSV = (
    PROJECT_ROOT
    / "annotation"
    / "sampled_object_crops_100_bucket_assignments_100ex_8subj_per_video_cap_babyview_only.csv"
)
TRAJECTORY_CSV = (
    ANALYSIS_DIR / "individual_analyses" / "developmental_trajectory_rdms_clip" / "trajectory_correlations.csv"
)

MODEL_CONFIGS = {
    "clip": {
        "embeddings_dir": Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/clip_embeddings_new"),
    },
    "dinov3": {
        "embeddings_dir": Path(
            "/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/facebook_dinov3-vitb16-pretrain-lvd1689m"
        ),
    },
}

DEFAULT_OUT_DIR = (
    SCRIPT_DIR
    / "supplemental_results_valid85cats_04202026"
    / "results"
    / "top8_subject_category_centroid_comparison_valid85_all_regular"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare category centroid representations across the same top-8 subjects used in notebook 09. "
            "For each category/model, computes pairwise subject-to-subject centroid distances."
        )
    )
    parser.add_argument("--category-set", default=DEFAULT_CATEGORY_SET, choices=["valid85"])
    parser.add_argument("--n-top", type=int, default=DEFAULT_N_TOP, help="Number of top-density subjects to include.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--per-category-top-n",
        type=int,
        default=12,
        help="Number of highest-shift categories per model to render as per-category subject heatmaps.",
    )
    return parser.parse_args()


def parse_subject_id_from_stem(stem: str) -> str | None:
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    return parts[2].zfill(8)


def get_top_subjects_from_trajectory(trajectory_csv: Path, n_top: int) -> list[str]:
    traj_df = pd.read_csv(trajectory_csv)
    traj_df["subject_id"] = traj_df["subject_id"].astype(str).str.zfill(8)

    if "density" not in traj_df.columns:
        needed = {"n_categories_younger", "n_categories_older"}
        if needed.issubset(set(traj_df.columns)):
            traj_df["density"] = traj_df["n_categories_younger"] + traj_df["n_categories_older"]
        else:
            raise ValueError("trajectory_correlations.csv must have density or younger/older category count columns.")

    ranked = (
        traj_df[["subject_id", "density"]]
        .drop_duplicates(subset=["subject_id"])
        .sort_values("density", ascending=False)
        .reset_index(drop=True)
    )
    return ranked.head(n_top)["subject_id"].tolist()


def load_valid85_regular_pairs(
    included_categories_txt: Path,
    sampled_exemplar_csv: Path,
) -> tuple[set[str], set[tuple[str, str]]]:
    valid_categories = {
        x.strip().lower()
        for x in included_categories_txt.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    sampled = pd.read_csv(sampled_exemplar_csv)
    sampled = sampled[sampled["trial_type"] == "regular"].copy()
    sampled["category"] = sampled["category"].astype(str).str.strip().str.lower()
    sampled["stem"] = sampled["stem"].astype(str).str.strip().str.lower()
    sampled = sampled[sampled["category"].isin(valid_categories)].copy()
    valid_pairs = set(zip(sampled["category"], sampled["stem"]))
    return valid_categories, valid_pairs


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom <= 0:
        return np.nan
    return float(1.0 - np.dot(u, v) / denom)


def collect_subject_category_centroids(
    embeddings_dir: Path,
    allowed_categories: set[str],
    allowed_pairs: set[tuple[str, str]],
    top_subjects: set[str],
    model_name: str,
) -> pd.DataFrame:
    centroid_rows: list[dict] = []
    category_dirs = [p for p in embeddings_dir.iterdir() if p.is_dir() and p.name in allowed_categories]

    for cat_dir in tqdm(sorted(category_dirs), desc=f"Collecting centroids ({model_name})"):
        category = cat_dir.name
        by_subject: dict[str, list[np.ndarray]] = defaultdict(list)

        for emb_file in cat_dir.glob("*.npy"):
            stem = emb_file.stem.lower()
            if (category, stem) not in allowed_pairs:
                continue
            subject_id = parse_subject_id_from_stem(stem)
            if subject_id is None or subject_id not in top_subjects:
                continue
            try:
                emb = np.load(emb_file).astype(np.float64, copy=False).reshape(-1)
            except Exception:
                continue
            by_subject[subject_id].append(emb)

        for subject_id, emb_list in by_subject.items():
            if len(emb_list) == 0:
                continue
            x = np.stack(emb_list, axis=0)
            centroid_rows.append(
                {
                    "model": model_name,
                    "category": category,
                    "subject_id": subject_id,
                    "n_exemplars": int(x.shape[0]),
                    "centroid": x.mean(axis=0),
                }
            )

    out = pd.DataFrame(centroid_rows)
    if out.empty:
        return out
    return out.sort_values(["category", "subject_id"]).reset_index(drop=True)


def build_pairwise_distance_table(centroids_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if centroids_df.empty:
        return pd.DataFrame()

    for (model, category), g in centroids_df.groupby(["model", "category"]):
        records = g.to_dict("records")
        if len(records) < 2:
            continue
        for a, b in combinations(records, 2):
            ca = a["centroid"]
            cb = b["centroid"]
            rows.append(
                {
                    "model": model,
                    "category": category,
                    "subject_a": a["subject_id"],
                    "subject_b": b["subject_id"],
                    "n_exemplars_a": int(a["n_exemplars"]),
                    "n_exemplars_b": int(b["n_exemplars"]),
                    "euclidean_centroid_distance": float(np.linalg.norm(ca - cb)),
                    "cosine_centroid_distance": cosine_distance(ca, cb),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["model", "category", "subject_a", "subject_b"]).reset_index(drop=True)


def build_category_summary(centroids_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if centroids_df.empty or pairwise_df.empty:
        return pd.DataFrame()
    n_subj = (
        centroids_df.groupby(["model", "category"], as_index=False)["subject_id"]
        .nunique()
        .rename(columns={"subject_id": "n_subjects_with_category"})
    )
    summary = (
        pairwise_df.groupby(["model", "category"], as_index=False)
        .agg(
            n_subject_pairs=("euclidean_centroid_distance", "size"),
            mean_euclidean_centroid_distance=("euclidean_centroid_distance", "mean"),
            median_euclidean_centroid_distance=("euclidean_centroid_distance", "median"),
            std_euclidean_centroid_distance=("euclidean_centroid_distance", "std"),
            mean_cosine_centroid_distance=("cosine_centroid_distance", "mean"),
            median_cosine_centroid_distance=("cosine_centroid_distance", "median"),
            std_cosine_centroid_distance=("cosine_centroid_distance", "std"),
        )
        .merge(n_subj, on=["model", "category"], how="left")
    )
    return summary.sort_values(["model", "mean_euclidean_centroid_distance"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_subject_pair_summary(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()
    summary = (
        pairwise_df.groupby(["model", "subject_a", "subject_b"], as_index=False)
        .agg(
            n_shared_categories=("category", "nunique"),
            mean_euclidean_centroid_distance=("euclidean_centroid_distance", "mean"),
            median_euclidean_centroid_distance=("euclidean_centroid_distance", "median"),
            std_euclidean_centroid_distance=("euclidean_centroid_distance", "std"),
            mean_cosine_centroid_distance=("cosine_centroid_distance", "mean"),
            median_cosine_centroid_distance=("cosine_centroid_distance", "median"),
            std_cosine_centroid_distance=("cosine_centroid_distance", "std"),
        )
        .sort_values(["model", "mean_euclidean_centroid_distance"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def _build_subject_distance_matrix(
    summary_df: pd.DataFrame,
    subjects: list[str],
    value_col: str,
) -> np.ndarray:
    idx = {s: i for i, s in enumerate(subjects)}
    m = np.full((len(subjects), len(subjects)), np.nan, dtype=float)
    np.fill_diagonal(m, 0.0)
    for r in summary_df.itertuples(index=False):
        i = idx.get(str(r.subject_a))
        j = idx.get(str(r.subject_b))
        if i is None or j is None:
            continue
        v = float(getattr(r, value_col))
        m[i, j] = v
        m[j, i] = v
    return m


def _draw_heatmap(ax: plt.Axes, matrix: np.ndarray, labels: list[str], title: str) -> None:
    img = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not np.isfinite(matrix[i, j]):
                continue
            if i == j:
                continue
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


def save_visualizations(
    out_dir: Path,
    top_subjects: list[str],
    pairwise_df: pd.DataFrame,
    category_summary_df: pd.DataFrame,
    subject_pair_summary_df: pd.DataFrame,
    category_set: str,
    per_category_top_n: int,
) -> None:
    fig_dir = out_dir.parent.parent / "figures" / "top8_subject_category_centroid_comparison_valid85_all_regular"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if subject_pair_summary_df.empty:
        print("No subject-pair summary available for visualization.")
        return

    for model_name in ["clip", "dinov3"]:
        sp = subject_pair_summary_df[subject_pair_summary_df["model"] == model_name].copy()
        cs = category_summary_df[category_summary_df["model"] == model_name].copy()
        if sp.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=180, constrained_layout=True)

        eu_mat = _build_subject_distance_matrix(sp, top_subjects, "mean_euclidean_centroid_distance")
        cos_mat = _build_subject_distance_matrix(sp, top_subjects, "mean_cosine_centroid_distance")
        _draw_heatmap(
            axes[0],
            eu_mat,
            top_subjects,
            f"{model_name.upper()} subject-pair mean Euclidean distance\n(across shared categories)",
        )
        _draw_heatmap(
            axes[1],
            cos_mat,
            top_subjects,
            f"{model_name.upper()} subject-pair mean cosine distance\n(across shared categories)",
        )

        cs_top = cs.sort_values("mean_euclidean_centroid_distance", ascending=False).head(15)
        axes[2].barh(
            cs_top["category"],
            cs_top["mean_euclidean_centroid_distance"],
            color="#4C78A8",
            alpha=0.9,
        )
        axes[2].invert_yaxis()
        axes[2].set_title(f"{model_name.upper()} categories with largest cross-subject shift", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Mean pairwise subject centroid distance")
        axes[2].set_ylabel("Category")
        axes[2].grid(axis="x", alpha=0.2)

        fig.suptitle(
            f"Top-8 subject category-centroid comparison ({model_name.upper()}, {category_set})",
            fontsize=13,
            fontweight="bold",
        )
        out_png = fig_dir / f"top8_subject_category_centroid_comparison_{model_name}_{category_set}.png"
        out_pdf = fig_dir / f"top8_subject_category_centroid_comparison_{model_name}_{category_set}.pdf"
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {out_png}")
        print(f"Wrote: {out_pdf}")

        # Per-category subject heatmaps for top shifted categories.
        if per_category_top_n <= 0:
            continue
        top_cats = cs.sort_values("mean_euclidean_centroid_distance", ascending=False)["category"].head(
            per_category_top_n
        )
        if top_cats.empty:
            continue
        n = len(top_cats)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        for metric_col, metric_name, cmap_name, out_stem in [
            ("euclidean_centroid_distance", "Euclidean distance", "magma", "euclidean"),
            ("cosine_centroid_distance", "Cosine distance", "cividis", "cosine"),
        ]:
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), dpi=180, constrained_layout=True
            )
            axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
            for i, cat in enumerate(top_cats.tolist()):
                r, c = divmod(i, ncols)
                ax = axes_arr[r, c]
                cat_df = pairwise_df[(pairwise_df["model"] == model_name) & (pairwise_df["category"] == cat)].copy()
                if cat_df.empty:
                    ax.axis("off")
                    continue
                mat = _build_subject_distance_matrix(cat_df, top_subjects, metric_col)
                img = ax.imshow(mat, cmap=cmap_name, interpolation="nearest")
                ax.set_title(f"{cat}", fontsize=10, fontweight="bold")
                ax.set_xticks(range(len(top_subjects)))
                ax.set_yticks(range(len(top_subjects)))
                ax.set_xticklabels(top_subjects, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(top_subjects, fontsize=7)
                plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

            # Hide unused axes.
            for j in range(n, nrows * ncols):
                r, c = divmod(j, ncols)
                axes_arr[r, c].axis("off")

            fig.suptitle(
                (
                    f"Per-category subject centroid {metric_name.lower()}s "
                    f"({model_name.upper()}, top {n} categories, {category_set})"
                ),
                fontsize=13,
                fontweight="bold",
            )
            cat_out_png = fig_dir / f"top8_per_category_subject_heatmaps_{out_stem}_{model_name}_{category_set}.png"
            cat_out_pdf = fig_dir / f"top8_per_category_subject_heatmaps_{out_stem}_{model_name}_{category_set}.pdf"
            fig.savefig(cat_out_png, dpi=300, bbox_inches="tight")
            fig.savefig(cat_out_pdf, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote: {cat_out_png}")
            print(f"Wrote: {cat_out_pdf}")


def run(category_set: str, n_top: int, out_dir: Path, per_category_top_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    top_subjects = get_top_subjects_from_trajectory(TRAJECTORY_CSV, n_top=n_top)
    top_subject_set = set(top_subjects)
    allowed_categories, allowed_pairs = load_valid85_regular_pairs(INCLUDED_CATEGORIES_TXT, SAMPLED_EXEMPLAR_CSV)

    all_centroids: list[pd.DataFrame] = []
    for model_name, cfg in MODEL_CONFIGS.items():
        centroids = collect_subject_category_centroids(
            embeddings_dir=cfg["embeddings_dir"],
            allowed_categories=allowed_categories,
            allowed_pairs=allowed_pairs,
            top_subjects=top_subject_set,
            model_name=model_name,
        )
        all_centroids.append(centroids)

    centroids_df = pd.concat(all_centroids, ignore_index=True) if all_centroids else pd.DataFrame()
    pairwise_df = build_pairwise_distance_table(centroids_df)
    category_summary_df = build_category_summary(centroids_df, pairwise_df)
    subject_pair_summary_df = build_subject_pair_summary(pairwise_df)

    top_subjects_df = pd.DataFrame({"subject_id": top_subjects})
    top_subjects_path = out_dir / "top8_subjects_from_trajectory.csv"
    centroids_path = out_dir / f"top8_subject_category_centroids_{category_set}.csv"
    pairwise_path = out_dir / f"top8_subject_pairwise_category_centroid_distances_{category_set}.csv"
    category_summary_path = out_dir / f"top8_category_centroid_distance_summary_{category_set}.csv"
    subject_pair_summary_path = out_dir / f"top8_subject_pair_centroid_distance_summary_{category_set}.csv"

    top_subjects_df.to_csv(top_subjects_path, index=False)
    if not centroids_df.empty:
        # Store centroid vectors as compact strings for easier CSV portability.
        export_centroids = centroids_df.copy()
        export_centroids["centroid"] = export_centroids["centroid"].map(
            lambda x: " ".join(f"{v:.8g}" for v in np.asarray(x, dtype=float))
        )
        export_centroids.to_csv(centroids_path, index=False)
    else:
        pd.DataFrame().to_csv(centroids_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    category_summary_df.to_csv(category_summary_path, index=False)
    subject_pair_summary_df.to_csv(subject_pair_summary_path, index=False)

    save_visualizations(
        out_dir=out_dir,
        top_subjects=top_subjects,
        pairwise_df=pairwise_df,
        category_summary_df=category_summary_df,
        subject_pair_summary_df=subject_pair_summary_df,
        category_set=category_set,
        per_category_top_n=per_category_top_n,
    )

    print(f"Top subjects ({len(top_subjects)}): {', '.join(top_subjects)}")
    print(f"Wrote: {top_subjects_path}")
    print(f"Wrote: {centroids_path}")
    print(f"Wrote: {pairwise_path}")
    print(f"Wrote: {category_summary_path}")
    print(f"Wrote: {subject_pair_summary_path}")


if __name__ == "__main__":
    args = parse_args()
    run(
        category_set=args.category_set,
        n_top=args.n_top,
        out_dir=args.out_dir,
        per_category_top_n=args.per_category_top_n,
    )
