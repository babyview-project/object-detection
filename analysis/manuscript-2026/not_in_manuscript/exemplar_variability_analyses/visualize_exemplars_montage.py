#!/usr/bin/env python3
"""
Visualize exemplars for top/bottom variable categories: montages of 10–12 cropped
exemplars per category, ordered by distance-to-centroid so spread is visible.

Supports four configs: bv_clip, bv_dinov3, things_clip, things_dinov3.
- BV: uses variability CSV + grouped embeddings; picks one crop per (subject_id, age_mo)
  from metadata and cropped images dir.
- THINGS: uses variability CSV + per-category .npy embeddings; images from
  --things-images-dir/{category}/{stem}.jpg (or .png).

Usage:
  cd analysis/manuscript-2026/exemplar_variability_analyses
  # BV CLIP (default)
  python visualize_exemplars_montage.py
  # BV DINOv3
  python visualize_exemplars_montage.py --config bv_dinov3
  # THINGS CLIP / THINGS DINOv3 (requires --things-images-dir for montage images)
  python visualize_exemplars_montage.py --config things_clip --things-images-dir /path/to/things/images
  python visualize_exemplars_montage.py --config things_dinov3 --things-images-dir /path/to/things/images
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Default paths (relative to script or absolute)
SCRIPT_DIR = Path(__file__).resolve().parent
GROUPED_EMBEDDINGS_BASE = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings")
DEFAULT_GROUPED_EMBEDDINGS = {
    "bv_clip": GROUPED_EMBEDDINGS_BASE / "clip_embeddings_grouped_by_age-mo_normalized",
    "bv_dinov3": GROUPED_EMBEDDINGS_BASE / "facebook_dinov3-vitb16-pretrain-lvd1689m_grouped_by_age-mo_normalized",
}
THINGS_BASE = Path("/ccn2/dataset/babyview/outputs_20250312")
THINGS_EMBEDDINGS = {
    "things_clip": THINGS_BASE / "things_bv_overlapping_categories_corrected/embeddings/image_embeddings/clip_image_embeddings_npy_by_category",
    "things_dinov3": THINGS_BASE / "image_embeddings/things_bv_overlapping_categories_corrected/facebook_dinov3-vitb16-pretrain-lvd1689m",
}
DEFAULT_METADATA_CSV = Path("/home/j7yang/babyview-projects/vss2026/object-detection/frame_data/merged_frame_detections_with_metadata.csv")
DEFAULT_CROPPED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class")
EXCLUDED_SUBJECT = "00270001"
CONFIGS = ("bv_clip", "bv_dinov3", "things_clip", "things_dinov3")


def get_category_crop_dir(cropped_dir, cat_name):
    """Return Path to category subfolder (match by name, case-insensitive)."""
    cat_lower = cat_name.strip().lower()
    direct = cropped_dir / cat_name
    if direct.exists() and direct.is_dir():
        return direct
    for p in cropped_dir.iterdir():
        if p.is_dir() and p.name.lower() == cat_lower:
            return p
    return cropped_dir / cat_name  # may not exist


def get_things_image_path(things_images_dir, cat_name, stem):
    """Return Path to THINGS image: things_images_dir/category/stem.jpg or .png."""
    cat_dir = get_category_crop_dir(things_images_dir, cat_name)
    for ext in (".jpg", ".jpeg", ".png"):
        p = cat_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return cat_dir / f"{stem}.jpg"  # may not exist


# Cache: (things_images_dir, cat_name) -> sorted list of image paths (so i-th embedding = i-th image)
_things_image_list_cache = {}


def get_things_image_by_index(things_images_dir, cat_name, index):
    """Return Path to the index-th image in category (sorted by name). THINGS .npy are embedding_000000, ...; images are category_01b.jpg, ... — match by position."""
    key = (Path(things_images_dir), cat_name)
    if key not in _things_image_list_cache:
        cat_dir = get_category_crop_dir(things_images_dir, cat_name)
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(cat_dir.glob(ext))
        paths = sorted(paths, key=lambda p: p.name)
        _things_image_list_cache[key] = paths
    paths = _things_image_list_cache[key]
    if index < 0 or index >= len(paths):
        return None
    return paths[index]


def load_grouped_embeddings_and_ids(grouped_dir, categories_set, excluded_subject=None, min_exemplars=2):
    """Load per-category embeddings and exemplar (subject_id, age_mo) ids. Same logic as notebook."""
    grouped_dir = Path(grouped_dir)
    category_embeddings = {}
    category_exemplar_ids = {}
    for cat_folder in sorted(grouped_dir.iterdir()):
        if not cat_folder.is_dir():
            continue
        cat_name = cat_folder.name
        if categories_set is not None and cat_name not in categories_set:
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


def load_things_embeddings_and_ids(embeddings_dir, categories_set, min_exemplars=2):
    """Load THINGS per-category embeddings from {embeddings_dir}/{category}/*.npy. Returns same format as BV: category_embeddings, category_exemplar_ids (ids are (stem, None))."""
    from load_things_embeddings import load_things_dinov3_from_dir

    return load_things_dinov3_from_dir(
        Path(embeddings_dir),
        allowed_categories=categories_set,
        min_exemplars=min_exemplars,
    )


def build_exemplar_to_crop_lookup(metadata_csv, usecols=None, chunksize=500_000):
    """Build (class_name, subject_id, age_mo) -> original_embedding_name (stem for crop). One per group."""
    if usecols is None:
        usecols = ["class_name", "subject_id", "age_mo", "original_embedding_name"]
    lookup = {}
    for chunk in tqdm(
        pd.read_csv(metadata_csv, usecols=usecols, chunksize=chunksize, dtype={"subject_id": str, "class_name": str},
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


def main():
    parser = argparse.ArgumentParser(description="Montages of exemplar crops for top/bottom variable categories")
    parser.add_argument("--config", type=str, default="bv_clip", choices=CONFIGS,
                        help="Config: bv_clip, bv_dinov3, things_clip, things_dinov3")
    parser.add_argument("--variability-csv", type=Path, default=None,
                        help="Override: variability CSV (default: {config}_within_category_variability.csv)")
    parser.add_argument("--grouped-embeddings-dir", type=Path, default=None,
                        help="Override: BV grouped embeddings dir (used for bv_* configs)")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV,
                        help="Merged frame detections metadata CSV (BV only)")
    parser.add_argument("--cropped-dir", type=Path, default=DEFAULT_CROPPED_DIR,
                        help="Root dir of cropped images: cropped_dir/class_name/*.jpg (BV only)")
    parser.add_argument("--things-images-dir", type=Path, default=None,
                        help="THINGS images root: {dir}/{category}/{stem}.jpg (required for things_* montages)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: exemplar_montages or exemplar_montages_{config})")
    parser.add_argument("--top", type=int, default=20, help="Number of top (highest variability) categories")
    parser.add_argument("--bottom", type=int, default=20, help="Number of bottom (lowest variability) categories")
    parser.add_argument("--n-exemplars", type=int, default=12, help="Exemplars to show per category (10–12)")
    parser.add_argument("--cell-size", type=int, nargs=2, default=[128, 128], metavar=("W", "H"),
                        help="Cell size for each crop in montage")
    parser.add_argument("--n-cols", type=int, default=4, help="Number of columns in montage")
    args = parser.parse_args()

    config = args.config
    is_bv = config.startswith("bv_")
    variability_csv = args.variability_csv or (SCRIPT_DIR / f"{config}_within_category_variability.csv")
    out_dir = args.out_dir or (SCRIPT_DIR / ("exemplar_montages" if config == "bv_clip" else f"exemplar_montages_{config}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_size = tuple(args.cell_size)

    # 1) Top/bottom categories from variability CSV
    var_df = pd.read_csv(variability_csv)
    var_df = var_df.sort_values("mean_dist_to_centroid", ascending=False).reset_index(drop=True)
    top_cats = var_df.head(args.top)["category"].tolist()
    bottom_cats = var_df.tail(args.bottom)["category"].tolist()
    selected_categories = [(c, "top", i + 1) for i, c in enumerate(top_cats)] + [
        (c, "bottom", i + 1) for i, c in enumerate(bottom_cats)
    ]
    categories_set = set(c[0] for c in selected_categories)

    # 2) Load embeddings and exemplar ids
    print(f"Loading embeddings for {config}...")
    if is_bv:
        grouped_dir = args.grouped_embeddings_dir or DEFAULT_GROUPED_EMBEDDINGS[config]
        category_embeddings, category_exemplar_ids = load_grouped_embeddings_and_ids(
            grouped_dir, categories_set, excluded_subject=EXCLUDED_SUBJECT, min_exemplars=2
        )
        print("Building exemplar -> crop lookup from metadata...")
        lookup = build_exemplar_to_crop_lookup(args.metadata_csv)
        things_images_dir = None
    else:
        embeddings_dir = THINGS_EMBEDDINGS[config]
        category_embeddings, category_exemplar_ids = load_things_embeddings_and_ids(
            embeddings_dir, categories_set, min_exemplars=2
        )
        lookup = None
        things_images_dir = args.things_images_dir
        if things_images_dir is None or not Path(things_images_dir).exists():
            print(f"Warning: --things-images-dir not set or missing. THINGS montages need images at {{dir}}/{{category}}/{{stem}}.jpg")
            things_images_dir = None

    # 3) For each selected category: get exemplars sorted by distance, pick n_exemplars, load crops, save montage
    for cat_name, rank_type, rank_idx in tqdm(selected_categories, desc="Montages"):
        if cat_name not in category_embeddings:
            continue
        X = category_embeddings[cat_name]
        ids = category_exemplar_ids[cat_name]
        centroid = X.mean(axis=0)
        dists = np.linalg.norm(X - centroid, axis=1)
        order = np.argsort(dists)
        n_show = min(args.n_exemplars, len(order))
        indices = np.linspace(0, len(order) - 1, n_show, dtype=int) if len(order) > n_show else np.arange(len(order))
        selected_idx = order[indices]

        images = []
        labels = []
        cat_key_lower = cat_name.strip().lower()
        for i in selected_idx:
            exemplar_id, age_mo = ids[i]
            d = float(dists[i])
            if is_bv:
                key = (cat_key_lower, exemplar_id, int(age_mo))
                stem = lookup.get(key) or lookup.get((cat_name, exemplar_id, int(age_mo)))
                if stem is None:
                    continue
                cat_dir = get_category_crop_dir(args.cropped_dir, cat_name)
                crop_path = cat_dir / f"{stem}.jpg"
            else:
                if things_images_dir is None:
                    continue
                # THINGS: .npy are embedding_000000,...; images are category_01b.jpg,... — match by index
                crop_path = get_things_image_by_index(things_images_dir, cat_name, i)
            if crop_path is None or not crop_path.exists():
                continue
            try:
                img = Image.open(crop_path).convert("RGB")
                images.append(img)
                labels.append(f"{d:.1f}")
            except Exception:
                continue

        if not images:
            continue
        n_cols_use = min(args.n_cols, len(images))
        montage = make_montage_with_labels(images, labels, n_cols_use, cell_size)
        if montage is None:
            continue
        rank_label = f"{rank_type}{rank_idx:02d}"
        out_name = f"exemplar_montage_{rank_label}_{cat_name}.png"
        out_path = out_dir / out_name
        montage.save(out_path)
        with open(out_dir / f"exemplar_montage_{rank_label}_{cat_name}_distances.txt", "w") as f:
            f.write("dist_to_centroid (left-to-right, top-to-bottom)\n")
            f.write("\n".join(labels))

    print(f"Saved montages to {out_dir}")


if __name__ == "__main__":
    main()
