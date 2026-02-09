#!/usr/bin/env python3
"""
Generate montages of exemplar object crops for each annotation format
(toy, drawing, photo, real, video). For each format, collects category bounding-box
crops from images that contain that format and tiles them into a grid.

Usage:
  cd annotation && python make_format_montages.py [--dataset NAME] [--frames DIR] [--out DIR]
"""
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
from PIL import Image


FORMAT_TYPES = ["toy_format", "drawing_format", "photo_format", "real_format", "video_format"]


def load_class_names(obj_names_file):
    with open(obj_names_file) as f:
        class_names = [line.strip() for line in f.readlines()]
    class_dict = {i: name for i, name in enumerate(class_names)}
    format_classes = [c for c in class_names if c.endswith("_format")]
    return class_dict, format_classes


def parse_yolo_line(line, class_dict):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cid = int(parts[0])
    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = xc + w / 2, yc + h / 2
    name = class_dict.get(cid, "unknown")
    return (cid, name, (x1, y1, x2, y2))


def load_annotations(ann_dir, class_dict, format_classes):
    ann_files = list(ann_dir.glob("*.txt"))
    all_boxes_by_image = []
    for ann_file in ann_files:
        boxes = []
        with open(ann_file) as f:
            for line in f:
                parsed = parse_yolo_line(line, class_dict)
                if parsed is None:
                    continue
                cid, cname, bbox = parsed
                boxes.append({"class_id": cid, "class_name": cname, "bbox": bbox})
        all_boxes_by_image.append((ann_file, boxes))
    return all_boxes_by_image


def normalized_to_pixel(bbox_norm, width, height):
    x1, y1, x2, y2 = bbox_norm
    return (
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height),
    )


def make_montage(images, n_cols, cell_size):
    """Tile PIL images into a grid; each image resized to cell_size."""
    if not images:
        return None
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    out = np.zeros((n_rows * cell_size[1], n_cols * cell_size[0], 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        row, col = idx // n_cols, idx % n_cols
        if img.size != (cell_size[0], cell_size[1]):
            img = img.resize((cell_size[0], cell_size[1]), Image.Resampling.LANCZOS)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = arr[:, :, :3]
        h, w = arr.shape[:2]
        out[row * cell_size[1] : row * cell_size[1] + h, col * cell_size[0] : col * cell_size[0] + w, :] = arr
    return out


def generate_montages(
    all_boxes_by_image,
    format_classes,
    frames_base: Path,
    ann_dir: Path,
    out_dir: Path,
    max_crops_per_format=30,
    crop_size=(128, 128),
    n_cols=6,
):
    """Generate one montage per format; save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in FORMAT_TYPES:
        crops = []
        for ann_file, boxes in all_boxes_by_image:
            format_boxes = [b for b in boxes if b["class_name"] == fmt]
            category_boxes = [b for b in boxes if b["class_name"] not in format_classes]
            if not format_boxes or not category_boxes:
                continue
            if len(crops) >= max_crops_per_format:
                break
            frame_path = frames_base / f"{ann_file.stem}.jpg"
            if not frame_path.exists():
                frame_path = ann_dir / f"{ann_file.stem}.jpg"
            if not frame_path.exists():
                continue
            try:
                im = Image.open(frame_path).convert("RGB")
            except Exception:
                continue
            w, h = im.size
            for b in category_boxes:
                if len(crops) >= max_crops_per_format:
                    break
                x1, y1, x2, y2 = normalized_to_pixel(b["bbox"], w, h)
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = im.crop((x1, y1, x2, y2))
                crops.append(crop)
        if not crops:
            print(f"{fmt}: no crops (missing frames or no exemplars)")
            continue
        montage = make_montage(crops, n_cols, crop_size)
        if montage is not None:
            name = fmt.replace("_format", "")
            out_path = out_dir / f"{name}_montage.jpg"
            Image.fromarray(montage).save(out_path)
            print(f"Saved {out_path} ({len(crops)} crops)")
    print("Done. Montages in", out_dir.resolve())


def main():
    ap = argparse.ArgumentParser(description="Generate format exemplar montages from annotation data")
    ap.add_argument("--dataset", default="Mira_20251112_completed_edited_VEDI", help="Subfolder name under annotation_data/")
    ap.add_argument("--frames", type=Path, default=None, help="Directory containing source .jpg frames (default: same as annotation .txt)")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: figures/format_montages)")
    ap.add_argument("--max-crops", type=int, default=30, help="Max crops per format montage")
    ap.add_argument("--crop-size", type=int, nargs=2, default=[128, 128], metavar=("W", "H"))
    ap.add_argument("--cols", type=int, default=6, help="Montage grid columns")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    annotation_base = script_dir
    if not (annotation_base / "annotation_data").exists():
        annotation_base = script_dir.parent / "annotation"
    dataset_dir = annotation_base / "annotation_data" / args.dataset
    obj_names_file = dataset_dir / "obj.names"
    obj_train_data_dir = dataset_dir / "obj_train_data"

    if not obj_names_file.exists():
        raise FileNotFoundError(f"Not found: {obj_names_file}")

    txt_files = list(obj_train_data_dir.rglob("*.txt"))
    ann_dir = txt_files[0].parent if txt_files else obj_train_data_dir
    frames_base = args.frames if args.frames is not None else ann_dir
    out_dir = args.out if args.out is not None else (script_dir / "figures" / "format_montages")

    class_dict, format_classes = load_class_names(obj_names_file)
    all_boxes_by_image = load_annotations(ann_dir, class_dict, format_classes)
    print(f"Loaded {len(all_boxes_by_image)} annotation files")

    generate_montages(
        all_boxes_by_image,
        format_classes,
        frames_base,
        ann_dir,
        out_dir,
        max_crops_per_format=args.max_crops,
        crop_size=tuple(args.crop_size),
        n_cols=args.cols,
    )


if __name__ == "__main__":
    main()
