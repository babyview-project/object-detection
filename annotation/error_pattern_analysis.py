#!/usr/bin/env python3
"""
Error pattern analysis: annotations vs YOLO detections.

- Loads human annotations and YOLO (initial) detections at bbox level.
- Matches YOLO boxes to human boxes by IoU; when labels differ, records error (true_label -> pred_label).
- Plots error pattern graph (true X -> wrong Y) color-coded by CDI semantic category.
- Creates error montages grouped by (true category) and optionally by (true, pred) pair.

Usage:
  cd annotation && python error_pattern_analysis.py [--yolo-dir DIR] [--annotation-dirs DIR ...] [--frames-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATION_BASE = SCRIPT_DIR
N_OBJECT_CLASSES = 295

DEFAULT_YOLO_INITIAL_DIR = ANNOTATION_BASE / "3000_inverse_proportional_sampled_frames_split" / "Dora_yoloe_annotations"
# Frames as frame_key.jpg; used for cropping montages (crop = human annotation bbox)
DEFAULT_FRAMES_DIR = ANNOTATION_BASE / "3000_inverse_proportional_sampled_frames"
DEFAULT_ANNOTATION_DATASET_DIRS = [
    ANNOTATION_BASE / "annotation_data" / "Mira_20251001_completed_edited_VEDI",
    ANNOTATION_BASE / "annotation_data" / "Dora_20251001_completed_VEDI",
]
CATEGORY_WHITELIST_PATH = PROJECT_ROOT / "data" / "things_bv_overlap_categories_exclude_zero_precisions.txt"
CDI_WORDS_CSV = PROJECT_ROOT / "data" / "cdi_words.csv"

# CDI semantic colors (subset used in long_tailed_distribution_163cats.py)
CDI_SEMANTIC_COLORS = {
    "animals": "#00B894",
    "body_parts": "#E74C3C",
    "clothing": "#6C5CE7",
    "food_drink": "#E67E22",
    "furniture_rooms": "#27AE60",
    "household": "#E056BD",
    "outside": "#00CEC9",
    "people": "#F1C40F",
    "toys": "#9B59B6",
    "vehicles": "#3498DB",
    "other": "#95A5A6",
}
CDI_SEMANTIC_ORDER = [
    "animals", "body_parts", "clothing", "food_drink", "furniture_rooms",
    "household", "outside", "people", "toys", "vehicles", "other",
]


def _project_root():
    if (SCRIPT_DIR / "annotation_data").exists():
        return SCRIPT_DIR.parent
    if (SCRIPT_DIR.parent / "annotation" / "annotation_data").exists():
        return SCRIPT_DIR.parent
    return SCRIPT_DIR.parent.parent


def load_category_whitelist(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    out = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        w = line.split(":", 1)[1].strip().lower() if ":" in line else line.lower()
        if w:
            out.add(w)
    return out


def load_obj_names(obj_names_path: Path) -> dict[int, str]:
    with open(obj_names_path) as f:
        names = [line.strip() for line in f if line.strip()]
    return {i: name for i, name in enumerate(names)}


def parse_yolo_line(line: str, class_dict: dict[int, str]) -> tuple[int, str, tuple[float, float, float, float]] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cid = int(parts[0])
    except (ValueError, TypeError):
        return None
    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = xc + w / 2, yc + h / 2
    name = class_dict.get(cid, "unknown")
    return (cid, name, (x1, y1, x2, y2))


def load_cdi_semantic_mapping(cdi_path: Path) -> dict[str, str]:
    df = pd.read_csv(cdi_path, usecols=["uni_lemma", "category"])
    df["uni_lemma"] = df["uni_lemma"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    return df.drop_duplicates(subset=["uni_lemma"], keep="first").set_index("uni_lemma")["category"].to_dict()


def iou_box(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    """IoU of two boxes (x1, y1, x2, y2) in normalized coords."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def normalized_to_pixel(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(x1 * width)),
        max(0, int(y1 * height)),
        min(width, int(x2 * width)),
        min(height, int(y2 * height)),
    )


# ---------------------------------------------------------------------------
# Load YOLO and human annotations (bbox-level)
# ---------------------------------------------------------------------------


def load_yolo_boxes_by_frame(
    yolo_dir: Path,
    class_dict: dict[int, str],
    category_whitelist: set[str] | None,
) -> tuple[set[str], dict[str, list[tuple[str, tuple[float, float, float, float]]]]]:
    """Returns (annotation_frames, frame_key -> [(class_name, bbox), ...])."""
    txts = [p for p in yolo_dir.glob("*.txt") if p.name != "train.txt"]
    annotation_frames = {p.stem for p in txts}
    frame_to_boxes: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for txt_path in txts:
        frame_key = txt_path.stem
        with open(txt_path) as f:
            for line in f:
                parsed = parse_yolo_line(line, class_dict)
                if parsed is None:
                    continue
                cid, name, bbox = parsed
                if cid >= N_OBJECT_CLASSES:
                    continue
                name_norm = name.strip().lower()
                if category_whitelist is not None and name_norm not in category_whitelist:
                    continue
                frame_to_boxes[frame_key].append((name_norm, bbox))
    return annotation_frames, dict(frame_to_boxes)


def load_human_boxes_by_frame(
    annotation_datasets: list[tuple[str, Path, Path, list[Path]]],
    category_whitelist: set[str] | None,
    annotation_frames: set[str],
) -> dict[str, list[tuple[str, tuple[float, float, float, float]]]]:
    """Returns frame_key -> [(class_name, bbox), ...] for frames in annotation_frames."""
    out: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for _ds_name, obj_names_path, _ann_dir, txt_files in annotation_datasets:
        class_dict = load_obj_names(obj_names_path)
        for ann_file in txt_files:
            frame_key = ann_file.stem
            if frame_key not in annotation_frames:
                continue
            with open(ann_file) as f:
                for line in f:
                    parsed = parse_yolo_line(line, class_dict)
                    if parsed is None:
                        continue
                    cid, name, bbox = parsed
                    if cid >= N_OBJECT_CLASSES:
                        continue
                    name_norm = name.strip().lower()
                    if category_whitelist is not None and name_norm not in category_whitelist:
                        continue
                    out[frame_key].append((name_norm, bbox))
    return {k: v for k, v in out.items() if v}


def collect_annotation_datasets(annotation_base: Path, dataset_dirs: list[Path] | None) -> list[tuple[str, Path, Path, list[Path]]]:
    if dataset_dirs is not None:
        dirs_to_scan = list(dataset_dirs)
    else:
        dirs_to_scan = list((annotation_base / "annotation_data").iterdir())
    result = []
    for sub in dirs_to_scan:
        if not sub.exists() or not sub.is_dir():
            continue
        obj_names = sub / "obj.names"
        obj_train = sub / "obj_train_data"
        if not obj_names.exists() or not obj_train.exists():
            continue
        txt_files = [p for p in obj_train.rglob("*.txt") if p.name != "obj.names"]
        if not txt_files:
            continue
        ann_dir = txt_files[0].parent
        if (ann_dir / "obj.names").exists():
            obj_names = ann_dir / "obj.names"
        result.append((sub.name, obj_names, ann_dir, txt_files))
    return result


def compute_errors_corrects_misses(
    yolo_frame_to_boxes: dict[str, list[tuple[str, tuple[float, float, float, float]]]],
    human_frame_to_boxes: dict[str, list[tuple[str, tuple[float, float, float, float]]]],
    iou_threshold: float = 0.3,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    For each YOLO box, find best-overlapping human box.
    - If IoU >= threshold and labels differ -> error (true_label, pred_label, frame_key, bbox=human).
    - If IoU >= threshold and labels match -> correct (frame_key, bbox=human, label).
    Then: for each human box, if no YOLO box matched it (same label, IoU >= threshold) -> miss (frame_key, bbox, label).

    Returns (errors, corrects, misses). All bboxes are human (annotation) bbox for cropping.
    """
    errors = []
    corrects = []
    # Track which human boxes were matched by some YOLO (same label)
    matched_human: set[tuple[str, int]] = set()  # (frame_key, human_box_index)

    for frame_key, yolo_boxes in yolo_frame_to_boxes.items():
        human_boxes = human_frame_to_boxes.get(frame_key, [])
        if not human_boxes:
            continue
        for pred_label, yolo_bbox in yolo_boxes:
            best_iou = 0.0
            best_true_label = None
            best_bbox = None
            best_hi = None
            for hi, (true_label, human_bbox) in enumerate(human_boxes):
                iou = iou_box(yolo_bbox, human_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_true_label = true_label
                    best_bbox = human_bbox
                    best_hi = hi
            if best_iou < iou_threshold or best_true_label is None:
                continue
            if best_true_label != pred_label:
                errors.append({
                    "true_label": best_true_label,
                    "pred_label": pred_label,
                    "frame_key": frame_key,
                    "bbox": best_bbox,
                })
            else:
                corrects.append({"frame_key": frame_key, "bbox": best_bbox, "label": best_true_label})
                if best_hi is not None:
                    matched_human.add((frame_key, best_hi))

    # Misses: human boxes that were never matched by a same-label YOLO box
    misses = []
    for frame_key, human_boxes in human_frame_to_boxes.items():
        yolo_boxes = yolo_frame_to_boxes.get(frame_key, [])
        for hi, (true_label, human_bbox) in enumerate(human_boxes):
            if (frame_key, hi) in matched_human:
                continue
            found = False
            for pred_label, yolo_bbox in yolo_boxes:
                if pred_label != true_label:
                    continue
                if iou_box(yolo_bbox, human_bbox) >= iou_threshold:
                    found = True
                    break
            if not found:
                misses.append({"frame_key": frame_key, "bbox": human_bbox, "label": true_label})

    # Dedupe corrects (same human box can be best match for multiple YOLO boxes)
    seen = set()
    corrects_uniq = []
    for c in corrects:
        key = (c["frame_key"], c["label"], c["bbox"])
        if key not in seen:
            seen.add(key)
            corrects_uniq.append(c)

    return errors, corrects_uniq, misses


# ---------------------------------------------------------------------------
# Resolve frame key to image path
# ---------------------------------------------------------------------------


def resolve_frame_image(frame_key: str, frames_dirs: list[Path]) -> Path | None:
    for d in frames_dirs:
        p = d / f"{frame_key}.jpg"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Plotting: error pattern graph (true -> pred) color by CDI semantic
# ---------------------------------------------------------------------------


def plot_error_pattern(
    errors_df: pd.DataFrame,
    cdi_semantic_map: dict[str, str],
    out_base: Path,
    top_n_true: int = 25,
    top_n_pred: int = 25,
    min_count: int = 2,
) -> None:
    """
    Plot heatmap: rows = true label (X), cols = pred label (Y). Count = number of errors (X -> Y).
    Color sidebar by CDI semantic of true label. Saves both .png and .pdf when out_base has no extension.
    """
    if errors_df.empty:
        print("No errors to plot.")
        return

    # Aggregate counts (true_label -> pred_label)
    count_df = errors_df.groupby(["true_label", "pred_label"]).size().reset_index(name="count")
    count_df = count_df[count_df["count"] >= min_count]

    if count_df.empty:
        print("No error pairs with count >= min_count.")
        return

    # Top true labels by total errors
    true_totals = count_df.groupby("true_label")["count"].sum().sort_values(ascending=False)
    top_true = true_totals.head(top_n_true).index.tolist()
    # Top pred labels by total errors
    pred_totals = count_df.groupby("pred_label")["count"].sum().sort_values(ascending=False)
    top_pred = pred_totals.head(top_n_pred).index.tolist()

    sub = count_df[
        count_df["true_label"].isin(top_true) & count_df["pred_label"].isin(top_pred)
    ]
    pivot = sub.pivot_table(index="true_label", columns="pred_label", values="count", fill_value=0)

    # Align order: true by total (desc), pred by total (desc)
    true_order = [t for t in top_true if t in pivot.index]
    pred_order = [p for p in top_pred if p in pivot.columns]
    pivot = pivot.reindex(index=true_order, columns=pred_order, fill_value=0)

    # CDI semantic for true labels (row colors)
    row_cdi = pivot.index.map(lambda c: cdi_semantic_map.get(c, "other"))
    row_colors = [CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]) for s in row_cdi]

    fig, ax = plt.subplots(figsize=(max(10, len(pred_order) * 0.45), max(8, len(true_order) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=pivot.values.max() or 1)
    ax.set_xticks(np.arange(len(pred_order)))
    ax.set_xticklabels(pred_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(true_order)))
    ax.set_yticklabels(true_order)
    ax.set_xlabel("YOLO prediction (wrong label Y)")
    ax.set_ylabel("Human label (true label X)")
    ax.set_title("Error pattern: true label X → wrong prediction Y (color = CDI semantic of X)")

    # Row color bar (CDI semantic of true label)
    for i, color in enumerate(row_colors):
        ax.axhspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.35, zorder=0)
    ax.set_ylim(len(true_order) - 0.5, -0.5)

    used_semantics = sorted(set(row_cdi), key=lambda s: CDI_SEMANTIC_ORDER.index(s) if s in CDI_SEMANTIC_ORDER else 99)
    legend_handles = [
        Patch(facecolor=CDI_SEMANTIC_COLORS.get(s, CDI_SEMANTIC_COLORS["other"]), label=s.replace("_", " "))
        for s in CDI_SEMANTIC_ORDER if s in used_semantics
    ]
    ax.legend(handles=legend_handles, title="CDI semantic (true)", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.colorbar(im, ax=ax, label="Number of errors")
    fig.tight_layout()
    base = out_base.with_suffix("") if out_base.suffix else out_base
    for ext in (".png", ".pdf"):
        path = base.with_suffix(ext)
        fig.savefig(path, dpi=150 if ext == ".png" else None, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Montages
# ---------------------------------------------------------------------------


def make_montage_grid(images: list[Image.Image], n_cols: int, cell_size: tuple[int, int]) -> np.ndarray | None:
    if not images:
        return None
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    out = np.zeros((n_rows * cell_size[1], n_cols * cell_size[0], 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        row, col = idx // n_cols, idx % n_cols
        if img.size != cell_size:
            img = img.resize(cell_size, Image.Resampling.LANCZOS)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[:, :, :3]
        h, w = arr.shape[:2]
        out[row * cell_size[1] : row * cell_size[1] + h, col * cell_size[0] : col * cell_size[0] + w, :] = arr
    return out


def build_error_montages(
    errors: list[dict],
    frames_dirs: list[Path],
    out_dir: Path,
    max_crops_per_group: int = 24,
    crop_size: tuple[int, int] = (128, 128),
    n_cols: int = 6,
) -> None:
    """
    Create montages by cropping each full frame using the human annotation bbox (the region
    the annotator drew around the object). So each crop shows "what was actually there" (true
    label X) that YOLO misclassified as Y.

    1) By true category: one montage per true_label.
    2) By (true_label, pred_label): one montage per pair with at least 2 errors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by true_label
    by_true: dict[str, list[dict]] = defaultdict(list)
    for e in errors:
        by_true[e["true_label"]].append(e)

    for true_label, group in sorted(by_true.items()):
        crops = []
        for e in group[:max_crops_per_group]:
            frame_key = e["frame_key"]
            bbox = e["bbox"]
            path = resolve_frame_image(frame_key, frames_dirs)
            if path is None:
                continue
            try:
                im = Image.open(path).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            x1, y1, x2, y2 = normalized_to_pixel(bbox, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = im.crop((x1, y1, x2, y2))
            crops.append(crop)
        if not crops:
            continue
        montage = make_montage_grid(crops, n_cols, crop_size)
        if montage is not None:
            safe_name = true_label.replace("/", "_").replace(" ", "_")
            out_path = out_dir / f"error_montage_true_{safe_name}.png"
            Image.fromarray(montage).save(out_path)
            print(f"Saved {out_path} ({len(crops)} crops, true={true_label})")

    # By (true_label, pred_label) for pairs with multiple errors
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in errors:
        by_pair[(e["true_label"], e["pred_label"])].append(e)

    pair_dir = out_dir / "by_pair"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for (true_label, pred_label), group in sorted(by_pair.items(), key=lambda x: -len(x[1])):
        if len(group) < 2:
            continue
        crops = []
        for e in group[:max_crops_per_group]:
            path = resolve_frame_image(e["frame_key"], frames_dirs)
            if path is None:
                continue
            try:
                im = Image.open(path).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            x1, y1, x2, y2 = normalized_to_pixel(e["bbox"], w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(im.crop((x1, y1, x2, y2)))
        if not crops:
            continue
        montage = make_montage_grid(crops, n_cols, crop_size)
        if montage is not None:
            safe_true = true_label.replace("/", "_").replace(" ", "_")
            safe_pred = pred_label.replace("/", "_").replace(" ", "_")
            out_path = pair_dir / f"error_{safe_true}_to_{safe_pred}.png"
            Image.fromarray(montage).save(out_path)
            print(f"Saved {out_path} ({len(crops)} crops, {true_label} -> {pred_label})")


def _crop_list_to_montages(
    items: list[dict],
    frames_dirs: list[Path],
    out_dir: Path,
    prefix: str,
    label_key: str,
    max_crops_per_group: int,
    crop_size: tuple[int, int],
    n_cols: int,
) -> None:
    """Build montages by label: prefix_X.png = crops for label X. Each item has frame_key, bbox, and label_key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_label[it[label_key]].append(it)

    for label, group in sorted(by_label.items()):
        crops = []
        for e in group[:max_crops_per_group]:
            path = resolve_frame_image(e["frame_key"], frames_dirs)
            if path is None:
                continue
            try:
                im = Image.open(path).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            x1, y1, x2, y2 = normalized_to_pixel(e["bbox"], w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(im.crop((x1, y1, x2, y2)))
        if not crops:
            continue
        montage = make_montage_grid(crops, n_cols, crop_size)
        if montage is not None:
            safe_name = label.replace("/", "_").replace(" ", "_")
            out_path = out_dir / f"{prefix}_{safe_name}.png"
            Image.fromarray(montage).save(out_path)
            print(f"Saved {out_path} ({len(crops)} crops, {label})")


def build_correct_and_miss_montages(
    corrects: list[dict],
    misses: list[dict],
    frames_dirs: list[Path],
    out_dir: Path,
    max_crops_per_group: int = 24,
    crop_size: tuple[int, int] = (128, 128),
    n_cols: int = 6,
) -> None:
    """
    Create montages for correct (true positive) and missed (false negative) detections.
    - correct_X: human said X, YOLO also said X (crop = human bbox).
    - missed_X: human said X, YOLO did not detect X on that frame (crop = human bbox).
    """
    _crop_list_to_montages(
        corrects, frames_dirs, out_dir,
        prefix="correct", label_key="label",
        max_crops_per_group=max_crops_per_group, crop_size=crop_size, n_cols=n_cols,
    )
    _crop_list_to_montages(
        misses, frames_dirs, out_dir,
        prefix="missed", label_key="label",
        max_crops_per_group=max_crops_per_group, crop_size=crop_size, n_cols=n_cols,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Error pattern analysis: annotations vs YOLO, CDI-colored plots and montages")
    ap.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO_INITIAL_DIR, help="YOLO initial .txt dir (3000-frame set)")
    ap.add_argument("--annotation-dirs", type=Path, nargs="*", default=None, help="Annotation dataset dirs (default: Mira + Dora VEDI)")
    ap.add_argument("--frames-dir", type=Path, action="append", default=None, help="Dir(s) with frame .jpg (frame_key.jpg). Default: annotation/3000_inverse_proportional_sampled_frames. Can repeat.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: annotation/error_pattern_outputs)")
    ap.add_argument("--iou-threshold", type=float, default=0.3, help="Min IoU to match YOLO box to human box")
    ap.add_argument("--top-n", type=int, default=25, help="Top N true/pred labels in heatmap")
    ap.add_argument("--min-count", type=int, default=2, help="Min error count to show in heatmap")
    ap.add_argument("--no-montages", action="store_true", help="Skip montage generation (e.g. if no frame images)")
    ap.add_argument("--max-crops", type=int, default=24, help="Max crops per montage")
    args = ap.parse_args()

    project_root = _project_root()
    if args.out_dir is None:
        args.out_dir = SCRIPT_DIR / "error_pattern_outputs"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    category_whitelist = load_category_whitelist(CATEGORY_WHITELIST_PATH)
    cdi_semantic_map = load_cdi_semantic_mapping(CDI_WORDS_CSV)

    # Annotation datasets
    if args.annotation_dirs is None:
        annotation_dirs = DEFAULT_ANNOTATION_DATASET_DIRS
    else:
        annotation_dirs = list(args.annotation_dirs)
    annotation_datasets = collect_annotation_datasets(ANNOTATION_BASE, annotation_dirs)
    if not annotation_datasets:
        raise SystemExit("No annotation datasets found.")
    print(f"Annotation datasets: {[d[0] for d in annotation_datasets]}")

    # YOLO boxes
    yolo_dir = args.yolo_dir
    if not yolo_dir.exists():
        raise SystemExit(f"YOLO dir not found: {yolo_dir}")
    obj_names = yolo_dir / "obj.names"
    class_dict = load_obj_names(obj_names)
    annotation_frames, yolo_frame_to_boxes = load_yolo_boxes_by_frame(yolo_dir, class_dict, category_whitelist)
    print(f"YOLO: {len(annotation_frames)} frames, {sum(len(b) for b in yolo_frame_to_boxes.values())} boxes")

    # Human boxes (only on frames we have)
    human_frame_to_boxes = load_human_boxes_by_frame(annotation_datasets, category_whitelist, annotation_frames)
    n_human = sum(len(b) for b in human_frame_to_boxes.values())
    print(f"Human: {len(human_frame_to_boxes)} frames, {n_human} boxes")

    # Errors, corrects (true positive), misses (false negative)
    errors, corrects, misses = compute_errors_corrects_misses(
        yolo_frame_to_boxes, human_frame_to_boxes, iou_threshold=args.iou_threshold
    )
    print(f"Errors (true X -> wrong Y): {len(errors)}")
    print(f"Correct (true positive): {len(corrects)}")
    print(f"Missed (false negative): {len(misses)}")

    if not errors:
        print("No errors found. Exiting.")
        return

    errors_df = pd.DataFrame(errors)
    errors_df["cdi_semantic"] = errors_df["true_label"].map(lambda c: cdi_semantic_map.get(c, "other"))

    # Save error table
    errors_df.to_csv(args.out_dir / "error_pairs.csv", index=False)
    count_df = errors_df.groupby(["true_label", "pred_label", "cdi_semantic"]).size().reset_index(name="count")
    count_df.to_csv(args.out_dir / "error_pattern_counts.csv", index=False)
    pd.DataFrame(corrects).to_csv(args.out_dir / "correct_pairs.csv", index=False)
    pd.DataFrame(misses).to_csv(args.out_dir / "missed_pairs.csv", index=False)
    print(f"Saved error_pairs.csv, error_pattern_counts.csv, correct_pairs.csv, missed_pairs.csv")

    # Plot: error pattern heatmap (true -> pred) color by CDI semantic (saves .png and .pdf)
    plot_error_pattern(
        errors_df,
        cdi_semantic_map,
        args.out_dir / "error_pattern_heatmap",
        top_n_true=args.top_n,
        top_n_pred=args.top_n,
        min_count=args.min_count,
    )

    # Montages: crop full frames using human annotation bbox; need frame images
    frames_dirs = []
    if args.frames_dir:
        frames_dirs.extend(args.frames_dir)
    else:
        # Default: 3000 sampled frames (frame_key.jpg)
        if DEFAULT_FRAMES_DIR.exists():
            frames_dirs.append(DEFAULT_FRAMES_DIR)
    for _name, _on, _ad, txt_files in annotation_datasets:
        parent = txt_files[0].parent if txt_files else None
        if parent and parent not in frames_dirs:
            frames_dirs.append(parent)

    if not args.no_montages and frames_dirs:
        build_error_montages(
            errors,
            frames_dirs,
            args.out_dir,
            max_crops_per_group=args.max_crops,
            crop_size=(128, 128),
            n_cols=6,
        )
        build_correct_and_miss_montages(
            corrects,
            misses,
            frames_dirs,
            args.out_dir,
            max_crops_per_group=args.max_crops,
            crop_size=(128, 128),
            n_cols=6,
        )
    elif args.no_montages:
        print("Skipping montages (--no-montages).")
    else:
        print("No frame image dirs found; skipping montages. Use --frames-dir to point to frame .jpg dir(s).")


if __name__ == "__main__":
    main()
