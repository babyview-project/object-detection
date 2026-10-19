#!/usr/bin/env python3
"""Render the two images annotators see for each sampled item.

  out/private/images/<item_id>_crop.jpg   the exact box CLIP scored (no padding)
  out/private/images/<item_id>_frame.jpg  the full frame with the box outlined

The crop is saved at native resolution; the annotation tool scales it for display.
Resumable: existing images are skipped.

Usage
  nice -n 19 python render_items.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
IMG = OUT / "images"
BOX_COLOR = (255, 40, 200)


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(OUT / "items.csv")
    n_new = 0
    for i, r in enumerate(s.itertuples(index=False), 1):
        crop_p, frame_p = IMG / f"{r.item_id}_crop.jpg", IMG / f"{r.item_id}_frame.jpg"
        if crop_p.exists() and frame_p.exists():
            continue
        im = Image.open(r.frame_path).convert("RGB")
        box = (r.xmin, r.ymin, r.xmax, r.ymax)
        im.crop(box).save(crop_p, quality=95)
        fr = im.copy()
        w = max(3, round(min(fr.size) / 120))
        ImageDraw.Draw(fr).rectangle(box, outline=BOX_COLOR, width=w)
        fr.save(frame_p, quality=92)
        n_new += 1
        if i % 500 == 0:
            print(f"{i}/{len(s)}", flush=True)
    print(f"done: {n_new} rendered, {len(s)} items in {IMG}")


if __name__ == "__main__":
    main()
