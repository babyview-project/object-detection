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
  python bv_to_things_centroid_distances.py --embedding clip --tsne-all-categories --tsne-edge-crops
"""
from pathlib import Path
import argparse
import pickle
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
DEFAULT_THINGS_IMAGES_DIR = THINGS_BASE / "things_bv_overlapping_categories_corrected"
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


# Cache for THINGS image paths by index: (things_images_dir, cat_name) -> sorted list
_things_image_list_cache = {}


def get_things_image_by_index(things_images_dir, cat_name, index):
    """Return Path to the index-th image in category (sorted by name), or None if missing."""
    if things_images_dir is None:
        return None
    key = (Path(things_images_dir), cat_name)
    if key not in _things_image_list_cache:
        cat_dir = get_category_crop_dir(things_images_dir, cat_name)
        if not cat_dir.exists():
            _things_image_list_cache[key] = []
        else:
            paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                paths.extend(cat_dir.glob(ext))
            _things_image_list_cache[key] = sorted(paths, key=lambda p: p.name)
    paths = _things_image_list_cache[key]
    if index < 0 or index >= len(paths):
        return None
    return paths[index]


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


def get_exemplar_to_crop_lookup_cached(metadata_csv: Path, cache_dir: Path):
    """
    Load exemplar->crop lookup from disk cache if metadata CSV unchanged; otherwise build and cache.
    Cache key uses CSV path, mtime, and size so we only rebuild when the file changes.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = Path(metadata_csv)
    try:
        stat = metadata_csv.stat()
        key = (str(metadata_csv.resolve()), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None
    cache_file = cache_dir / "exemplar_to_crop_lookup.pkl"
    meta_file = cache_dir / "exemplar_to_crop_lookup_meta.pkl"
    if key is not None and cache_file.exists() and meta_file.exists():
        try:
            with open(meta_file, "rb") as f:
                cached_key = pickle.load(f)
            if cached_key == key:
                with open(cache_file, "rb") as f:
                    lookup = pickle.load(f)
                print("Loaded exemplar -> crop lookup from cache.")
                return lookup
        except Exception:
            pass
    lookup = build_exemplar_to_crop_lookup(metadata_csv)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(lookup, f)
        with open(meta_file, "wb") as f:
            pickle.dump(key, f)
        print("Cached exemplar -> crop lookup.")
    except Exception as e:
        print(f"Could not write lookup cache: {e}")
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


def _angles_from_centroid(bx, by):
    """Return angles (radians) of each point from BV centroid, for ordering around perimeter."""
    cx, cy = np.mean(bx), np.mean(by)
    return np.arctan2(by - cy, bx - cx)


def _select_edge_indices(n_bv, n_show, bx, by):
    """
    Select up to n_show BV indices spread broadly around the t-SNE space (by angle from BV centroid)
    to showcase diversity. Uses evenly spaced angles so exemplars cover the full 360°.
    Returns list of (index, angle) sorted by angle so crops can be placed in order around the edge.
    """
    angles = _angles_from_centroid(bx, by)
    if n_bv <= n_show:
        idx_angle = list(zip(range(n_bv), angles))
    else:
        # Evenly spaced by angle for broad diversity (one per angular sector)
        order = np.argsort(angles)
        # Partition the circle into n_show equal angle bins and take one index per bin
        positions = np.linspace(0, n_bv, n_show, endpoint=False)
        indices = [order[min(int(p), n_bv - 1)] for p in positions]
        # Deduplicate while preserving angular order (if n_bv < n_show we get duplicates)
        seen = set()
        unique_indices = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                unique_indices.append(i)
        idx_angle = [(i, angles[i]) for i in unique_indices]
        idx_angle.sort(key=lambda x: x[1])
    return idx_angle


def _assign_crops_to_perimeter_sides(idx_angle_list, bx, by, sides_filter=None):
    """
    Assign each (idx, angle) to one of four sides: 'left', 'right', 'top', 'bottom'.
    Returns dict side -> list of (idx, angle) sorted for that side (e.g. left = top-to-bottom by by).
    If sides_filter is set (e.g. ["left", "bottom"]), only assign to those sides; others go to nearest allowed.
    """
    pi, pi4 = np.pi, np.pi / 4
    sides = {"left": [], "right": [], "top": [], "bottom": []}
    for idx, angle in idx_angle_list:
        if -pi4 <= angle < pi4:
            side = "right"
        elif pi4 <= angle < 3 * pi4:
            side = "top"
        elif -3 * pi4 <= angle < -pi4:
            side = "bottom"
        else:
            side = "left"
        if sides_filter is not None and side not in sides_filter:
            # Map to nearest allowed side by angle
            allowed = list(sides_filter)
            if not allowed:
                continue
            # pick closest allowed side
            side_angles = {"right": 0, "top": pi / 2, "left": np.pi, "bottom": -np.pi / 2}
            a = angle
            best = min(allowed, key=lambda s: abs(np.arctan2(np.sin(a - side_angles.get(s, 0)), np.cos(a - side_angles.get(s, 0)))))
            side = best
        sides[side].append((idx, angle))
    # Sort: left/right by y descending (top to bottom); top/bottom by x ascending (left to right)
    for side in ("left", "right"):
        sides[side].sort(key=lambda ia: -by[ia[0]])
    for side in ("top", "bottom"):
        sides[side].sort(key=lambda ia: bx[ia[0]])
    return sides


def _ray_bbox_intersection(px, py, tx, ty, xmin, xmax, ymin, ymax):
    """
    Return the point where the segment from (px, py) toward (tx, ty) first hits
    the rectangle [xmin, xmax] x [ymin, ymax]. If (px, py) is inside, return
    the boundary exit point. Uses the smallest t in (0, 1] so the line clearly
    runs from the dot to the axes edge.
    """
    dx, dy = tx - px, ty - py
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return px, py
    t = 2.0
    if abs(dx) >= 1e-12:
        for xb in (xmin, xmax):
            t_cand = (xb - px) / dx
            if 0 < t_cand <= 1:
                yy = py + t_cand * dy
                if ymin <= yy <= ymax and t_cand < t:
                    t = t_cand
    if abs(dy) >= 1e-12:
        for yb in (ymin, ymax):
            t_cand = (yb - py) / dy
            if 0 < t_cand <= 1:
                xx = px + t_cand * dx
                if xmin <= xx <= xmax and t_cand < t:
                    t = t_cand
    if t > 1:
        return tx, ty
    return px + t * dx, py + t * dy


def plot_tsne_one_category(
    category: str,
    bv_X: np.ndarray,
    things_X: np.ndarray,
    things_centroid: np.ndarray,
    dist_bv_to_things: np.ndarray,
    out_path: Path,
    perplexity: float = 30,
    random_state: int = 42,
    bv_id_list: list | None = None,
    exemplar_to_crop_lookup: dict | None = None,
    cropped_dir: Path | None = None,
    n_edge_crops: int = 12,
    crop_cell_size: int = 72,
    things_images_dir: Path | None = None,
    n_things_edge_crops: int = 6,
):
    """
    Fit t-SNE on [BV exemplars, THINGS exemplars, THINGS centroid] and plot.
    BV points colored by distance to THINGS centroid (far = warmer = potential false alarm).

    If bv_id_list, exemplar_to_crop_lookup, and cropped_dir are provided, up to n_edge_crops
    BV exemplar crops are placed on the left/bottom with connector lines from the corresponding
    t-SNE points. If things_images_dir is provided, up to n_things_edge_crops THINGS exemplar
    crops are placed on the right/top. BV crop frames use a blue edge; THINGS use orange.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from sklearn.manifold import TSNE

    n_bv = bv_X.shape[0]
    n_things = things_X.shape[0]
    # Stack: BV, THINGS, then centroid as single point
    X_all = np.vstack([bv_X, things_X, things_centroid.reshape(1, -1)])
    n_all = X_all.shape[0]

    # Perplexity must be < n_samples
    perplexity = min(perplexity, max(2, (n_all - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000)
    coords = tsne.fit_transform(X_all)

    bx, by = coords[:n_bv].T
    tx, ty = coords[n_bv : n_bv + n_things].T
    cx, cy = coords[-1]

    use_bv_crops = (
        bv_id_list is not None
        and exemplar_to_crop_lookup is not None
        and cropped_dir is not None
        and n_edge_crops > 0
    )
    use_things_crops = things_images_dir is not None and n_things_edge_crops > 0 and n_things > 0
    use_edge_crops = use_bv_crops or use_things_crops

    # Layout: main scatter in center; four strip axes for BV (4 per edge); four corner axes for THINGS
    strip_axes = {}
    corner_axes = {}
    # Aesthetic: soft background so main plot reads clearly
    strip_facecolor = "#f8f9fa"
    if use_edge_crops:
        fig = plt.figure(figsize=(14, 12), facecolor="#fafafa")
        main_rect = [0.16, 0.16, 0.68, 0.68]
        ax = fig.add_axes(main_rect)
        ax.set_facecolor("white")
        strip_w = 0.11  # wider strips so BV crops don't overlap (4 per edge)
        strip_rects = {
            "left": [0.02, 0.16, strip_w, 0.68],
            "right": [0.87, 0.16, strip_w, 0.68],
            "top": [0.16, 0.87, 0.68, strip_w],
            "bottom": [0.16, 0.02, 0.68, strip_w],
        }
        strip_axes = {s: fig.add_axes(r) for s, r in strip_rects.items()}
        for sa in strip_axes.values():
            sa.set_axis_off()
            sa.set_xlim(0, 1)
            sa.set_ylim(0, 1)
            sa.set_facecolor(strip_facecolor)
        # Four corner axes for THINGS (one exemplar per corner)
        if use_things_crops:
            corner_sz = 0.12
            corner_rects = {
                "top_left": [0.02, 0.88, corner_sz, corner_sz],
                "top_right": [0.86, 0.88, corner_sz, corner_sz],
                "bottom_left": [0.02, 0.02, corner_sz, corner_sz],
                "bottom_right": [0.86, 0.02, corner_sz, corner_sz],
            }
            corner_axes = {c: fig.add_axes(r) for c, r in corner_rects.items()}
            for ca in corner_axes.values():
                ca.set_axis_off()
                ca.set_xlim(0, 1)
                ca.set_ylim(0, 1)
                ca.set_facecolor(strip_facecolor)
    else:
        fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
        ax.set_facecolor("white")

    # THINGS exemplars: soft gray
    ax.scatter(tx, ty, c="#e9ecef", s=28, alpha=0.8, label="THINGS exemplars", edgecolors="#adb5bd", linewidths=0.4)
    # BV exemplars: colored by distance to THINGS centroid (viridis)
    sc = ax.scatter(bx, by, c=dist_bv_to_things, s=45, cmap="viridis", alpha=0.9, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="BV → THINGS centroid distance", shrink=0.7, pad=0.02)
    # THINGS centroid: star
    ax.scatter([cx], [cy], s=380, marker="*", c="#c92a2a", edgecolors="#a61e1e", linewidths=1.2, label="THINGS centroid", zorder=5)
    ax.set_title(f"t-SNE: {category}", fontsize=13, fontweight="600")
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=9)
    xmin, xmax = float(bx.min() - 1), float(bx.max() + 1)
    ymin, ymax = float(by.min() - 1), float(by.max() + 1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Slot layout: more margin so crops have breathing room
    slot_margin = 0.10
    slot_span = 1.0 - 2 * slot_margin

    # Force layout so transforms are correct for figure-coord connector lines
    if use_edge_crops:
        fig.canvas.draw()

    def data_to_figure(x_data, y_data, axes):
        """Transform (x, y) in axes coordinates to figure coordinates."""
        disp = axes.transData.transform((x_data, y_data))
        return fig.transFigure.inverted().transform(disp)

    # Aesthetic: subtle connectors and tidy crop frames (rounded, soft colors)
    connector_color = "#94a3b8"
    connector_alpha = 0.35
    connector_lw = 0.75
    crop_zoom = 0.92  # slightly smaller so scatter stays the focus
    bv_border = "#2563eb"   # clear blue
    things_border = "#ea580c"  # clear orange

    def place_crops_and_connectors(
        sides_dict,
        get_image_at_index,
        is_bv=True,
    ):
        """Place crops in strip_ax and draw connector from scatter point to crop (full line in figure coords)."""
        edge_color = bv_border if is_bv else things_border
        for side, items in sides_dict.items():
            if not items:
                continue
            strip_ax = strip_axes[side]
            n_slot = len(items)
            for slot_i, (idx, _angle) in enumerate(items):
                crop_path = get_image_at_index(idx, slot_i, n_slot)
                if crop_path is None or not Path(crop_path).exists():
                    continue
                try:
                    img = Image.open(crop_path).convert("RGB")
                except Exception:
                    continue
                img = img.resize((crop_cell_size, crop_cell_size), Image.Resampling.LANCZOS)
                img_arr = np.asarray(img)

                # Slot position with margin so crops don't overlap
                t = slot_margin + (slot_i + 0.5) / n_slot * slot_span
                if side in ("left", "right"):
                    slot_x, slot_y = 0.5, t
                else:
                    slot_x, slot_y = t, 0.5

                # Crop center in figure coords (for connector endpoint)
                crop_display = strip_ax.transAxes.transform((slot_x, slot_y))
                crop_fig = fig.transFigure.inverted().transform(crop_display)

                if is_bv:
                    px, py = float(bx[idx]), float(by[idx])
                else:
                    px, py = float(tx[idx]), float(ty[idx])

                # Scatter point in figure coords (connector start)
                point_fig = data_to_figure(px, py, ax)

                # Subtle connector line (light gray, thin)
                line = Line2D(
                    [point_fig[0], crop_fig[0]],
                    [point_fig[1], crop_fig[1]],
                    transform=fig.transFigure,
                    color=connector_color,
                    alpha=connector_alpha,
                    linewidth=connector_lw,
                    zorder=1,
                )
                fig.add_artist(line)

                imbox = OffsetImage(img_arr, zoom=crop_zoom)
                imbox.image.axes = strip_ax
                ab = AnnotationBbox(
                    imbox,
                    (slot_x, slot_y),
                    xycoords=strip_ax.transAxes,
                    frameon=True,
                    pad=0.08,
                    bboxprops=dict(
                        edgecolor=edge_color,
                        linewidth=2.0,
                        facecolor="white",
                        boxstyle="round,pad=0.02",
                    ),
                )
                strip_ax.add_artist(ab)

    if use_edge_crops:
        cat_key_lower = category.strip().lower()
        cat_crop_dir = get_category_crop_dir(cropped_dir, category) if cropped_dir else None

        if use_bv_crops:
            idx_angle_list = _select_edge_indices(n_bv, n_edge_crops, bx, by)
            # Always distribute BV equally across all four edges (4 per side for default 16)
            sides_bv = _assign_crops_to_perimeter_sides(idx_angle_list, bx, by, sides_filter=None)

            def get_bv_image(idx, _slot_i, _n_slot):
                sid, age_mo = bv_id_list[idx]
                sid_norm = str(sid).strip().lstrip("S")
                age_mo_int = int(age_mo)
                key = (cat_key_lower, sid_norm, age_mo_int)
                stem = exemplar_to_crop_lookup.get(key) or exemplar_to_crop_lookup.get((category, sid_norm, age_mo_int))
                if stem is None:
                    return None
                for ext in (".jpg", ".jpeg", ".png"):
                    p = cat_crop_dir / f"{stem}{ext}"
                    if p.exists():
                        return p
                return None

            place_crops_and_connectors(sides_bv, get_bv_image, is_bv=True)

        if use_things_crops:
            # Select 4 THINGS indices scattered by angle (one per quadrant) for the four corners
            angles_t = np.arctan2(ty - cy, tx - cx)
            n_show = min(n_things_edge_crops, n_things, 4)
            if n_things <= n_show:
                th_indices = list(range(n_things))
            else:
                # One per quadrant: take at 0°, 90°, 180°, -90° (or evenly spaced angles)
                order = np.argsort(angles_t)
                positions = np.linspace(0, n_things, n_show, endpoint=False)
                th_indices = [order[min(int(p), n_things - 1)] for p in positions]
            # Map angle order (right=0, top=π/2, left=π, bottom=-π/2) to corners
            corner_order = ["top_right", "top_left", "bottom_left", "bottom_right"]
            for corner_i, th_idx in enumerate(th_indices[:4]):
                corner_key = corner_order[corner_i]
                crop_path = get_things_image_by_index(things_images_dir, category, th_idx)
                if crop_path is None or not Path(crop_path).exists():
                    continue
                try:
                    img = Image.open(crop_path).convert("RGB")
                except Exception:
                    continue
                img = img.resize((crop_cell_size, crop_cell_size), Image.Resampling.LANCZOS)
                img_arr = np.asarray(img)
                strip_ax = corner_axes[corner_key]
                slot_x, slot_y = 0.5, 0.5  # center of corner axis
                crop_display = strip_ax.transAxes.transform((slot_x, slot_y))
                crop_fig = fig.transFigure.inverted().transform(crop_display)
                px, py = float(tx[th_idx]), float(ty[th_idx])
                point_fig = data_to_figure(px, py, ax)
                line = Line2D(
                    [point_fig[0], crop_fig[0]],
                    [point_fig[1], crop_fig[1]],
                    transform=fig.transFigure,
                    color=connector_color,
                    alpha=connector_alpha,
                    linewidth=connector_lw,
                    zorder=1,
                )
                fig.add_artist(line)
                imbox = OffsetImage(img_arr, zoom=crop_zoom)
                imbox.image.axes = strip_ax
                ab = AnnotationBbox(
                    imbox,
                    (slot_x, slot_y),
                    xycoords=strip_ax.transAxes,
                    frameon=True,
                    pad=0.08,
                    bboxprops=dict(
                        edgecolor=things_border,
                        linewidth=2.0,
                        facecolor="white",
                        boxstyle="round,pad=0.02",
                    ),
                )
                strip_ax.add_artist(ab)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.95, edgecolor="#dee2e6")
    if not use_edge_crops:
        plt.tight_layout()
    # When edge crops are used, connector lines are in figure coords; avoid bbox_inches='tight'
    # so the layout does not change on save and lines stay aligned with dots and crops.
    if use_edge_crops:
        plt.savefig(out_path, dpi=150, bbox_inches=None, pad_inches=0.1)
    else:
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
    parser.add_argument("--tsne-edge-crops", action="store_true",
                        help="Place BV exemplar crop images around the t-SNE plot edge with connector lines")
    parser.add_argument("--tsne-n-edge-crops", type=int, default=16,
                        help="Number of BV exemplar crops to show (4 per edge, default 16)")
    parser.add_argument("--tsne-crop-cell-size", type=int, default=72,
                        help="Size of each edge crop in pixels (default 72)")
    parser.add_argument("--things-images-dir", type=Path, default=DEFAULT_THINGS_IMAGES_DIR,
                        help="THINGS images root: {dir}/{category}/*.jpg (default: %(default)s). Set to empty to disable THINGS crops.")
    parser.add_argument("--tsne-n-things-edge-crops", type=int, default=4,
                        help="Number of THINGS exemplar crops to show at corners when --things-images-dir is set (default 4)")
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

    # Optional: load exemplar->crop lookup once (cached) for t-SNE edge crops and/or montages
    exemplar_to_crop_lookup = None
    if (args.tsne_edge_crops or args.montage_farthest) and args.metadata_csv.exists() and args.cropped_dir.exists():
        exemplar_to_crop_lookup = get_exemplar_to_crop_lookup_cached(args.metadata_csv, out_dir)

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
            bv_id_list = bv_ids[match]
            th_X = things_embeddings[match]
            th_centroid = things_centroids[match]
            dist_bv = np.linalg.norm(bv_X - th_centroid, axis=1)
            plot_tsne_one_category(
                match,
                bv_X, th_X, th_centroid,
                dist_bv,
                tsne_dir / f"tsne_{match}.png",
                perplexity=args.tsne_perplexity,
                bv_id_list=bv_id_list if args.tsne_edge_crops else None,
                exemplar_to_crop_lookup=exemplar_to_crop_lookup,
                cropped_dir=args.cropped_dir if args.tsne_edge_crops else None,
                n_edge_crops=args.tsne_n_edge_crops if args.tsne_edge_crops else 0,
                crop_cell_size=args.tsne_crop_cell_size,
                things_images_dir=args.things_images_dir if (args.things_images_dir and args.things_images_dir.exists()) else None,
                n_things_edge_crops=args.tsne_n_things_edge_crops if (args.things_images_dir and args.things_images_dir.exists()) else 0,
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
        elif exemplar_to_crop_lookup is None:
            print("Exemplar -> crop lookup not available. Skipping montages.")
        else:
            lookup = exemplar_to_crop_lookup
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
                    sid_norm = str(sid).strip().lstrip("S")
                    age_mo_int = int(age_mo)
                    key = (cat_key_lower, sid_norm, age_mo_int)
                    stem = lookup.get(key) or lookup.get((cat, sid_norm, age_mo_int))
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
