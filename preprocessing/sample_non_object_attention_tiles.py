#!/usr/bin/env python3
"""
Sample random non-object image tiles from BabyView frames.

Primary use case:
- Create an alternative attention-check image pool by cropping random regions
  from frames while avoiding known object bounding boxes (when YOLO labels exist).

Example:
  python preprocessing/sample_non_object_attention_tiles.py \
      --frames-dir /data2/dataset/babyview/868_hours/frames \
      --labels-dir /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_labels \
      --n-tiles 163
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "annotation" / "attention_check_non_object_tiles_163"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "annotation" / "attention_check_non_object_tiles_163.csv"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample random non-object tiles from BabyView frames."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Root directory containing source frame images (searched recursively).",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing YOLO txt label files with matching stems. "
            "If provided, sampled crops are constrained to have low overlap with these boxes."
        ),
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=163,
        help="Number of non-object tiles to create (default 163).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("W", "H"),
        help="Tile width/height in pixels (default 224 224).",
    )
    parser.add_argument(
        "--max-overlap-frac",
        type=float,
        default=0.01,
        help=(
            "Maximum allowed intersection fraction between sampled tile area and "
            "any label box area, measured against tile area (default 0.01)."
        ),
    )
    parser.add_argument(
        "--max-tries-per-frame",
        type=int,
        default=30,
        help="Max random crop attempts per chosen frame (default 30).",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=8.0,
        help="Reject very flat/blank crops with pixel std below this threshold (default 8.0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20000,
        help="Cap on number of candidate frames to scan (default 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where sampled tile images are written.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV manifest path for sampled tiles.",
    )
    return parser.parse_args()


def list_images(root: Path, max_frames: int) -> list[Path]:
    out: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        out.extend(root.rglob(f"*{ext}"))
    out = sorted(out)
    if max_frames > 0:
        out = out[:max_frames]
    return out


def load_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc = float(parts[1])
                yc = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue
            x1 = int((xc - bw / 2.0) * width)
            y1 = int((yc - bh / 2.0) * height)
            x2 = int((xc + bw / 2.0) * width)
            y2 = int((yc + bh / 2.0) * height)
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def is_crop_non_object_enough(
    crop_box: tuple[int, int, int, int],
    object_boxes: Iterable[tuple[int, int, int, int]],
    max_overlap_frac: float,
) -> bool:
    crop_area = max(1, (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]))
    max_frac = 0.0
    for obj_box in object_boxes:
        frac = intersection_area(crop_box, obj_box) / crop_area
        if frac > max_frac:
            max_frac = frac
        if max_frac > max_overlap_frac:
            return False
    return True


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    frames_dir = args.frames_dir
    labels_dir = args.labels_dir
    tile_w, tile_h = int(args.tile_size[0]), int(args.tile_size[1])

    if not frames_dir.exists():
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")
    if labels_dir is not None and not labels_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")
    if args.n_tiles <= 0:
        raise ValueError("--n-tiles must be > 0")
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("--tile-size must be positive")

    frame_paths = list_images(frames_dir, max_frames=args.max_frames)
    if not frame_paths:
        raise RuntimeError(f"No images found under: {frames_dir}")
    rng.shuffle(frame_paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    n_saved = 0

    # Keep cycling through shuffled frames until we have exactly n tiles or hit a hard limit.
    max_global_tries = max(1000, args.n_tiles * 500)
    global_tries = 0
    frame_idx = 0

    while n_saved < args.n_tiles and global_tries < max_global_tries:
        frame_path = frame_paths[frame_idx % len(frame_paths)]
        frame_idx += 1
        global_tries += 1

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception:
            continue

        width, height = image.size
        if width < tile_w or height < tile_h:
            continue

        object_boxes: list[tuple[int, int, int, int]] = []
        label_path = None
        if labels_dir is not None:
            label_path = labels_dir / f"{frame_path.stem}.txt"
            object_boxes = load_yolo_boxes(label_path, width, height)

        accepted = False
        for _ in range(args.max_tries_per_frame):
            x1 = rng.randint(0, width - tile_w)
            y1 = rng.randint(0, height - tile_h)
            crop_box = (x1, y1, x1 + tile_w, y1 + tile_h)
            if object_boxes and not is_crop_non_object_enough(
                crop_box=crop_box,
                object_boxes=object_boxes,
                max_overlap_frac=args.max_overlap_frac,
            ):
                continue

            crop = image.crop(crop_box)
            if crop.getextrema() is None:
                continue
            stat = crop.getbbox()
            if stat is None:
                continue
            # Convert to grayscale and use pixel std as a simple quality filter.
            gray = crop.convert("L")
            std = float(ImageStat.Stat(gray).stddev[0])
            if std < args.min_std:
                continue

            n_saved += 1
            out_name = f"nonobject_tile_{n_saved:04d}.jpg"
            out_path = args.output_dir / out_name
            crop.save(out_path, quality=95)
            rows.append(
                {
                    "tile_index": n_saved,
                    "tile_path": str(out_path.resolve()),
                    "source_frame_path": str(frame_path.resolve()),
                    "source_frame_stem": frame_path.stem,
                    "crop_x1": x1,
                    "crop_y1": y1,
                    "crop_x2": x1 + tile_w,
                    "crop_y2": y1 + tile_h,
                    "tile_width": tile_w,
                    "tile_height": tile_h,
                    "frame_width": width,
                    "frame_height": height,
                    "label_path": str(label_path.resolve()) if label_path and label_path.exists() else "",
                    "n_object_boxes_in_frame": len(object_boxes),
                    "seed": args.seed,
                }
            )
            accepted = True
            break

        if not accepted:
            continue

    if n_saved != args.n_tiles:
        raise RuntimeError(
            f"Could only create {n_saved}/{args.n_tiles} tiles. "
            "Try increasing --max-frames, lowering --min-std, increasing "
            "--max-tries-per-frame, or relaxing --max-overlap-frac."
        )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_index",
                "tile_path",
                "source_frame_path",
                "source_frame_stem",
                "crop_x1",
                "crop_y1",
                "crop_x2",
                "crop_y2",
                "tile_width",
                "tile_height",
                "frame_width",
                "frame_height",
                "label_path",
                "n_object_boxes_in_frame",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {n_saved} non-object tiles to {args.output_dir}")
    print(f"Wrote metadata CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
