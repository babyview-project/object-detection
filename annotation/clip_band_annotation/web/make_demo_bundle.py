#!/usr/bin/env python3
"""Write web/demo_bundle/: a tiny synthetic bundle (drawn shapes, no real data) for trying
the task and for tests. Same layout as make_inhouse_bundles.py output."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "demo_bundle"
BOX = (255, 40, 200)
PAGES = {"ball": ["circle"] * 5 + ["square"] * 2, "cup": ["cup"] * 4 + ["circle", "triangle"]}
COLORS = [(214, 88, 60), (60, 120, 200), (70, 160, 90), (230, 170, 40), (150, 90, 190), (90, 90, 90), (40, 170, 170)]


def shape(d, kind, box, color):
    x0, y0, x1, y1 = box
    if kind == "circle":
        d.ellipse(box, fill=color)
    elif kind == "square":
        d.rectangle(box, fill=color)
    elif kind == "triangle":
        d.polygon([(x0, y1), ((x0 + x1) / 2, y0), (x1, y1)], fill=color)
    else:  # cup: body + handle
        w = x1 - x0
        d.rectangle((x0, y0 + 10, x1 - w * 0.3, y1), fill=color)
        d.ellipse((x1 - w * 0.42, y0 + (y1 - y0) * 0.3, x1, y0 + (y1 - y0) * 0.7), outline=color, width=10)


def main():
    (OUT / "crop").mkdir(parents=True, exist_ok=True)
    (OUT / "frame").mkdir(exist_ok=True)
    pages, k = [], 0
    for cat, kinds in PAGES.items():
        items = []
        for kind in kinds:
            iid = hashlib.sha256(f"demo:{k}".encode()).hexdigest()[:12]
            fr = Image.new("RGB", (512, 910), (236, 230, 218))
            d = ImageDraw.Draw(fr)
            d.rectangle((0, 600, 512, 910), fill=(190, 160, 120))  # "table"
            box = (120 + (k * 37) % 180, 420 + (k * 53) % 200, 0, 0)
            box = (box[0], box[1], box[0] + 140, box[1] + 140)
            shape(d, kind, box, COLORS[k % len(COLORS)])
            fr.crop(box).save(OUT / "crop" / f"{iid}.jpg", quality=90)
            d.rectangle(box, outline=BOX, width=4)
            fr.save(OUT / "frame" / f"{iid}.jpg", quality=88)
            items.append(iid)
            k += 1
        pages.append({"category": cat, "items": items})
    m = {"study": "clip-band-context", "version": 1, "rater": "DEMO",
         "bundle_id": hashlib.sha256(json.dumps(pages).encode()).hexdigest()[:10],
         "n_items": k, "pages": pages}
    (OUT / "manifest.json").write_text(json.dumps(m, indent=1))
    print(f"wrote {OUT}: {k} items, {len(pages)} pages")


if __name__ == "__main__":
    main()
