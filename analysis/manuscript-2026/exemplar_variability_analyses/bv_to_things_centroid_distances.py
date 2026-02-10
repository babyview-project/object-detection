#!/usr/bin/env python3
"""
BV exemplar distance to THINGS category centroid.

For each category we have:
- THINGS centroid = mean of THINGS exemplar embeddings.
- For each BV exemplar: L2 distance to THINGS centroid (and optionally to BV centroid for comparison).

Outputs:
1. Per-category summary CSV: mean/std of BV-to-THINGS-centroid distance, mean BV-to-BV-centroid,
   correlation between the two distances (spread vs. distance from THINGS), n_bv, n_things.
2. Per-exemplar CSV (optional): category, subject_id, age_mo, dist_to_things_centroid, dist_to_bv_centroid.
3. t-SNE plot for a single category (e.g. giraffe): BV exemplars, THINGS exemplars, THINGS centroid;
   color BV points by distance to THINGS centroid to see if farthest are potential false alarms.
4. Montages of the farthest BV exemplars (by distance to THINGS centroid) per category to visually
   check for false alarms (--montage-farthest).

Usage:
  cd analysis/manuscript-2026/exemplar_variability_analyses
  python bv_to_things_centroid_distances.py --embedding clip
  python bv_to_things_centroid_distances.py --embedding clip --tsne-category giraffe
  python bv_to_things_centroid_distances.py --embedding clip --montage-farthest
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Reuse loaders from this dir
SCRIPT_DIR = Path(__file__).resolve().parent
GROUPED_EMBEDDINGS_BASE = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings")
GROUPED_EMBEDDINGS_DIRS = {
    "clip": GROUPED_EMBEDDINGS_BASE / "clip_embeddings_grouped_by_age-mo_normalized",
    "dinov3": GROUPED_EMBEDDINGS_BASE / "facebook_dinov3-vitb16-pretrain-lvd1689m_grouped_by_age-mo_normalized",
}
THINGS_BASE = Path("/ccn2/dataset/babyview/outputs_20250312")
THINGS_EMBEDDINGS_DIRS = {
    "clip": THINGS_BASE / "things_bv_overlapping_categories_corrected/embeddings/image_embeddings/clip_image_embeddings_npy_by_category",
    "dinov3": THINGS_BASE / "image_embeddings/things_bv_overlapping_categories_corrected/facebook_dinov3-vitb16-pretrain-lvd1689m",
}
CATEGORIES_FILE = SCRIPT_DIR / "../../../data/things_bv_overlap_categories_exclude_zero_precisions.txt"
EXCLUDED_SUBJECT = "00270001"
MIN_EXEMPLARS = 2
DEFAULT_METADATA_CSV = Path("/home/j7yang/babyview-projects/vss2026/object-detection/frame_data/merged_frame_detections_with_metadata.csv")
DEFAULT_CROPPED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class")


def get_category_crop_dir(cropped_dir, cat_name):
    """Return Path to category subfolder (match by name, case-insensitive)."""
    cat_lower = cat_name.strip().lower()
    direct = cropped_dir / cat_name
    if direct.exists() and direct.is_dir():
        return direct
    for p in cropped_dir.iterdir():
        if p.is_dir() and p.name.lower() == cat_lower:
            return p
    return cropped_dir / cat_name


def build_exemplar_to_crop_lookup(metadata_csv, usecols=None, chunksize=500_000):
    """Build (class_name, subject_id, age_mo) -> original_embedding_name (stem for crop). One per group."""
    if usecols is None:
        usecols = ["class_name", "subject_id", "age_mo", "original_embedding_name"]
    lookup = {}
    for chunk in tqdm(
        pd.read_csv(metadata_csv, usecols=usecols, chunksize=chunksize,
                    dtype={"subject_id": str, "class_name": str},
                    na_values=[], keep_default_na=False),
        desc="Metadata chunks",
        unit="chunk",
    ):
        chunk = chunk.dropna(subset=["class_name", "subject_id", "age_mo", "original_embedding_name"])
        chunk["subject_id_norm"] = chunk["subject_id"].str.strip().str.lstrip("S")
        chunk["age_mo_int"] = pd.to_numeric(chunk["age_mo"], errors="coerce").fillna(-1).astype(int)
        chunk = chunk[chunk["age_mo_int"] >= 0]
        for _, row in chunk.iterrows():
            key = (str(row["class_name"]).strip().lower(), row["subject_id_norm"], row["age_mo_int"])
            if key not in lookup:
                stem = str(row["original_embedding_name"]).strip()
                if stem.endswith(".npy"):
                    stem = stem[:-4]
                lookup[key] = stem
    return lookup


def make_montage_with_labels(images, labels, n_cols, cell_size, label_height=24):
    """Tile PIL images into a grid and draw distance label below each cell. Returns PIL Image."""
    if not images:
        return None
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    total_h = n_rows * (cell_size[1] + label_height)
    total_w = n_cols * cell_size[0]
    out = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for idx, img in enumerate(images):
        row, col = idx // n_cols, idx % n_cols
        if img.size != (cell_size[0], cell_size[1]):
            img = img.resize((cell_size[0], cell_size[1]), Image.Resampling.LANCZOS)
        r0 = row * (cell_size[1] + label_height)
        c0 = col * cell_size[0]
        out.paste(img, (c0, r0))
        if idx < len(labels):
            draw.text((c0 + 2, r0 + cell_size[1] + 2), f"d={labels[idx]}", fill=(0, 0, 0), font=font)
    return out


def load_allowed_categories():
    if not CATEGORIES_FILE.exists():
        return None
    with open(CATEGORIES_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def load_bv_embeddings(embedding_type: str, allowed_categories, excluded_subject, min_exemplars=2):
    """category_embeddings[cat] = (n, dim), category_exemplar_ids[cat] = [(subject_id, age_mo), ...]."""
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


def load_things_embeddings(embedding_type: str, allowed_categories, min_exemplars=2):
    """category_embeddings[cat] = (n, dim)."""
    from load_things_embeddings import load_things_dinov3_from_dir

    things_dir = THINGS_EMBEDDINGS_DIRS[embedding_type]
    cat_embs, _ = load_things_dinov3_from_dir(
        things_dir,
        allowed_categories=allowed_categories,
        min_exemplars=min_exemplars,
    )
    return cat_embs


def compute_things_centroids(things_embeddings):
    """Return dict category -> (dim,) centroid."""
    return {cat: X.mean(axis=0) for cat, X in things_embeddings.items()}


def run_distances(
    bv_embeddings,
    bv_ids,
    things_embeddings,
    things_centroids,
    categories_common,
):
    """
    For categories in categories_common, compute per-BV-exemplar distances to THINGS centroid and to BV centroid.
    Returns (summary_rows, exemplar_rows).
    """
    summary_rows = []
    exemplar_rows = []
    for cat in tqdm(categories_common, desc="Categories"):
        bv_X = bv_embeddings[cat]
        bv_id_list = bv_ids[cat]
        th_X = things_embeddings[cat]
        th_centroid = things_centroids[cat]

        bv_centroid = bv_X.mean(axis=0)
        dist_to_things = np.linalg.norm(bv_X - th_centroid, axis=1)
        dist_to_bv = np.linalg.norm(bv_X - bv_centroid, axis=1)

        n_bv = bv_X.shape[0]
        n_things = th_X.shape[0]
        mean_d_th = float(np.mean(dist_to_things))
        std_d_th = float(np.std(dist_to_things))
        mean_d_bv = float(np.mean(dist_to_bv))
        std_d_bv = float(np.std(dist_to_bv))
        if n_bv >= 2 and np.std(dist_to_things) > 0 and np.std(dist_to_bv) > 0:
            r, _ = stats.pearsonr(dist_to_things, dist_to_bv)
        else:
            r = np.nan
        summary_rows.append({
            "category": cat,
            "mean_bv_to_things_centroid": mean_d_th,
            "std_bv_to_things_centroid": std_d_th,
            "mean_bv_to_bv_centroid": mean_d_bv,
            "std_bv_to_bv_centroid": std_d_bv,
            "corr_dist_things_vs_dist_bv": r,
            "n_bv": n_bv,
            "n_things": n_things,
        })
        for i, (sid, age_mo) in enumerate(bv_id_list):
            exemplar_rows.append({
                "category": cat,
                "subject_id": sid,
                "age_mo": age_mo,
                "dist_to_things_centroid": float(dist_to_things[i]),
                "dist_to_bv_centroid": float(dist_to_bv[i]),
            })
    return summary_rows, exemplar_rows


def plot_tsne_one_category(
    category: str,
    bv_X: np.ndarray,
    things_X: np.ndarray,
    things_centroid: np.ndarray,
    dist_bv_to_things: np.ndarray,
    out_path: Path,
    perplexity: float = 30,
    random_state: int = 42,
):
    """
    Fit t-SNE on [BV exemplars, THINGS exemplars, THINGS centroid] and plot.
    BV points colored by distance to THINGS centroid (far = warmer = potential false alarm).
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n_bv = bv_X.shape[0]
    n_things = things_X.shape[0]
    # Stack: BV, THINGS, then centroid as single point
    X_all = np.vstack([bv_X, things_X, things_centroid.reshape(1, -1)])
    n_all = X_all.shape[0]
    # Labels: 0 = BV, 1 = THINGS, 2 = centroid
    source = np.array([0] * n_bv + [1] * n_things + [2])

    # Perplexity must be < n_samples
    perplexity = min(perplexity, max(2, (n_all - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000)
    coords = tsne.fit_transform(X_all)

    bx, by = coords[:n_bv].T
    tx, ty = coords[n_bv : n_bv + n_things].T
    cx, cy = coords[-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    # THINGS exemplars: light gray
    ax.scatter(tx, ty, c="lightgray", s=25, alpha=0.7, label="THINGS exemplars", edgecolors="gray")
    # BV exemplars: colored by distance to THINGS centroid (viridis: low=close, high=far)
    sc = ax.scatter(bx, by, c=dist_bv_to_things, s=40, cmap="viridis", alpha=0.85, edgecolors="black", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="BV exemplar distance to THINGS centroid")
    # THINGS centroid: star
    ax.scatter([cx], [cy], s=400, marker="*", c="red", edgecolors="darkred", linewidths=1.5, label="THINGS centroid", zorder=5)
    ax.set_title(f"t-SNE: {category} — BV (colored by dist to THINGS centroid) vs THINGS")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved t-SNE: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="BV exemplar distance to THINGS category centroid")
    parser.add_argument("--embedding", type=str, default="clip", choices=("clip", "dinov3"))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: script dir)")
    parser.add_argument("--save-exemplar-csv", action="store_true",
                        help="Save per-exemplar distances CSV")
    parser.add_argument("--tsne-category", type=str, default=None,
                        help="Run t-SNE for this single category (e.g. giraffe)")
    parser.add_argument("--tsne-all-categories", action="store_true",
                        help="Run t-SNE for every category (overrides --tsne-category)")
    parser.add_argument("--tsne-perplexity", type=float, default=30)
    parser.add_argument("--montage-farthest", action="store_true",
                        help="Create montages of farthest BV exemplars from THINGS centroid per category (to check false alarms)")
    parser.add_argument("--montage-n", type=int, default=12,
                        help="Number of farthest exemplars per category in montage (default 12)")
    parser.add_argument("--montage-max-categories", type=int, default=0,
                        help="Max categories to make montages for (0 = all, default)")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV,
                        help="Merged frame detections metadata CSV for crop lookup")
    parser.add_argument("--cropped-dir", type=Path, default=DEFAULT_CROPPED_DIR,
                        help="Root dir of cropped images: cropped_dir/class_name/*.jpg")
    parser.add_argument("--montage-cell-size", type=int, nargs=2, default=[128, 128], metavar=("W", "H"))
    parser.add_argument("--montage-cols", type=int, default=4)
    args = parser.parse_args()

    out_dir = args.out_dir or SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"bv_to_things_centroid_{args.embedding}"

    allowed = load_allowed_categories()
    print(f"Using {len(allowed) if allowed else 'all'} categories")

    print("Loading BV embeddings...")
    bv_embeddings, bv_ids = load_bv_embeddings(
        args.embedding,
        allowed_categories=allowed,
        excluded_subject=EXCLUDED_SUBJECT,
        min_exemplars=MIN_EXEMPLARS,
    )
    print("Loading THINGS embeddings...")
    things_embeddings = load_things_embeddings(
        args.embedding,
        allowed_categories=allowed,
        min_exemplars=MIN_EXEMPLARS,
    )
    categories_common = sorted(set(bv_embeddings.keys()) & set(things_embeddings.keys()))
    print(f"Common categories: {len(categories_common)}")

    things_centroids = compute_things_centroids(things_embeddings)

    summary_rows, exemplar_rows = run_distances(
        bv_embeddings, bv_ids, things_embeddings, things_centroids, categories_common
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("mean_bv_to_things_centroid", ascending=False).reset_index(drop=True)
    summary_path = out_dir / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Scatter: mean BV-to-THINGS-centroid vs mean BV-to-BV-centroid (spread)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            summary_df["mean_bv_to_bv_centroid"],
            summary_df["mean_bv_to_things_centroid"],
            alpha=0.7,
            s=25,
        )
        ax.set_xlabel("Mean BV exemplar distance to BV centroid (within-category spread)")
        ax.set_ylabel("Mean BV exemplar distance to THINGS centroid")
        ax.set_title(f"BV spread vs distance to THINGS centroid ({args.embedding})")
        # Diagonal: same distance to both centroids
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
        ax.legend()
        ax.set_aspect("equal")
        plt.tight_layout()
        scatter_path = out_dir / f"{prefix}_spread_vs_things_centroid.png"
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved scatter: {scatter_path}")
    except Exception as e:
        print(f"Scatter plot skip: {e}")

    if args.save_exemplar_csv:
        exemplar_df = pd.DataFrame(exemplar_rows)
        exemplar_path = out_dir / f"{prefix}_per_exemplar.csv"
        exemplar_df.to_csv(exemplar_path, index=False)
        print(f"Saved per-exemplar: {exemplar_path}")

    # t-SNE: one category or all categories
    if args.tsne_all_categories or args.tsne_category:
        tsne_dir = out_dir / f"{prefix}_tsne_by_category"
        tsne_dir.mkdir(exist_ok=True)
        if args.tsne_all_categories:
            categories_to_plot = categories_common
            print(f"Running t-SNE for all {len(categories_to_plot)} categories...")
        else:
            cat = args.tsne_category.strip().lower()
            match = None
            for c in categories_common:
                if c.lower() == cat:
                    match = c
                    break
            if match is None:
                print(f"Category '{args.tsne_category}' not in common list; skipping t-SNE.")
                categories_to_plot = []
            else:
                categories_to_plot = [match]
        for match in tqdm(categories_to_plot, desc="t-SNE"):
            bv_X = bv_embeddings[match]
            th_X = things_embeddings[match]
            th_centroid = things_centroids[match]
            dist_bv = np.linalg.norm(bv_X - th_centroid, axis=1)
            plot_tsne_one_category(
                match,
                bv_X, th_X, th_centroid,
                dist_bv,
                tsne_dir / f"tsne_{match}.png",
                perplexity=args.tsne_perplexity,
            )

    # Montages of farthest BV exemplars from THINGS centroid (to visually check false alarms)
    if args.montage_farthest:
        exemplar_df = pd.DataFrame(exemplar_rows)
        montage_dir = out_dir / f"{prefix}_montages_farthest_from_things_centroid"
        montage_dir.mkdir(exist_ok=True)
        cell_size = tuple(args.montage_cell_size)
        if not args.metadata_csv.exists():
            print(f"Metadata CSV not found: {args.metadata_csv}. Skipping montages.")
        elif not args.cropped_dir.exists():
            print(f"Cropped dir not found: {args.cropped_dir}. Skipping montages.")
        else:
            print("Building exemplar -> crop lookup from metadata...")
            lookup = build_exemplar_to_crop_lookup(args.metadata_csv)
            cats_for_montage = categories_common
            if args.montage_max_categories > 0:
                top_cats = summary_df.head(args.montage_max_categories)["category"].tolist()
                cats_for_montage = [c for c in categories_common if c in top_cats]
            for cat in tqdm(cats_for_montage, desc="Montages (farthest from THINGS centroid)"):
                rows = exemplar_df[exemplar_df["category"] == cat].copy()
                rows = rows.sort_values("dist_to_things_centroid", ascending=False).reset_index(drop=True)
                n_show = min(args.montage_n, len(rows))
                rows = rows.head(n_show)
                cat_key_lower = cat.strip().lower()
                images = []
                labels = []
                for _, row in rows.iterrows():
                    sid, age_mo = row["subject_id"], int(row["age_mo"])
                    key = (cat_key_lower, sid, age_mo)
                    stem = lookup.get(key) or lookup.get((cat, sid, age_mo))
                    if stem is None:
                        continue
                    cat_dir = get_category_crop_dir(args.cropped_dir, cat)
                    crop_path = None
                    for ext in (".jpg", ".jpeg", ".png"):
                        p = cat_dir / f"{stem}{ext}"
                        if p.exists():
                            crop_path = p
                            break
                    if crop_path is None:
                        continue
                    try:
                        img = Image.open(crop_path).convert("RGB")
                        images.append(img)
                        labels.append(f"{row['dist_to_things_centroid']:.1f}")
                    except Exception:
                        continue
                if not images:
                    continue
                n_cols_use = min(args.montage_cols, len(images))
                montage = make_montage_with_labels(images, labels, n_cols_use, cell_size)
                if montage is not None:
                    out_path = montage_dir / f"farthest_from_things_centroid_{cat}.png"
                    montage.save(out_path)
                    with open(montage_dir / f"farthest_from_things_centroid_{cat}_distances.txt", "w") as f:
                        f.write("dist_to_THINGS_centroid (left-to-right, top-to-bottom)\n")
                        f.write("\n".join(labels))
            print(f"Saved farthest-from-THINGS-centroid montages to {montage_dir}")


if __name__ == "__main__":
    main()
