#!/usr/bin/env python3
"""Frame-by-frame comparison on the depiction frames (build_depiction_frames.py).

Per frame, five panels:
  1. Human (Mira / Dora CVAT): each object box with "label · format"
  2. Gemini open vocabulary, neutral prompt (detect_open_neutral)
  3. Gemini open vocabulary, prompt that asks to name what toys / pictures / screens show
  4. Gemini combined prompt (detect_combo): "label (CDI word) · format", box colour = format
  5. YOLOE + CLIP >= threshold (solid kept, dashed dropped), as in compare_boxes_montage.py
Frames are grouped by stratum (toy, photo, drawing, video, real_only).

Output (private): out/private/depiction_compare/<model>/page_XX.jpg

  python depiction_montage.py [--model gemini-2.5-flash] [--threshold 0.27]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

import compare_boxes_montage as cb

VIOLET = (98, 80, 214)
ORDER = ["toy", "photo", "drawing", "video", "real_only"]


def load(path):
    recs = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            recs[r["unit_id"]] = r
    return recs


def gem_panel(frame, rec, title):
    boxes = [(l.strip().lower(), b) for l, b in cb.gemini_boxes(rec["raw"])] if rec and rec["error"] is None else []
    return cb.draw_panel(frame, [(l, b, cb.GREEN, False, 3) for l, b in boxes], title,
                         f"{len(boxes)} objects" if rec else "no reply")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-2.5-flash", choices=["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    ap.add_argument("--threshold", type=float, default=0.27)
    ap.add_argument("--per-page", type=int, default=4)
    args = ap.parse_args()
    P = cb.PRIV
    out = P / "depiction_compare" / args.model
    out.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(P / "depiction_frames.csv", dtype={"frame_id": str})
    frames["o"] = frames.stratum.map(ORDER.index)
    frames = frames.sort_values(["o", "unit_id"])
    human = pd.read_csv(P / "depiction_boxes.csv")
    neutral = load(P / f"{args.model}__detect_open_neutral__p1_depiction_frames.jsonl")
    asked = load(P / f"{args.model}__detect_open__p1_depiction_frames.jsonl")
    combo = load(P / f"{args.model}__detect_combo__p1_depiction_frames.jsonl")
    yi, clip = cb.yoloe_index(), cb.Clip()
    short = args.model.replace("gemini-", "")

    rows = []
    for r in frames.itertuples():
        frame = Image.open(r.frame_path).convert("RGB")
        rgb = np.asarray(frame)
        h = human[human.unit_id == r.unit_id]
        hp = cb.draw_panel(frame, [(f"{x.label} · {str(x.format).removesuffix('_format') if isinstance(x.format, str) else '?'}",
                                    (x.xmin, x.ymin, x.xmax, x.ymax), VIOLET, False, 3) for x in h.itertuples()],
                           f"Human ({r.stratum.replace('_', ' ')} frame)", f"{len(h)} labelled objects · label · format")
        p2 = gem_panel(frame, neutral.get(r.unit_id), f"Gemini {short} · open, neutral prompt")
        p3 = gem_panel(frame, asked.get(r.unit_id), f"Gemini {short} · open, asks for formats")
        b = pd.read_csv(yi[r.video] / "bounding_box_predictions.csv", dtype={"frame_id": str})
        b = b[b.xmin.notna()].copy()
        b["frame_id"] = b.frame_id.str.zfill(5)
        b["class_name"] = b.class_name.astype(str).str.strip().str.lower()
        b = b[(b.frame_id == r.frame_id) & ~b.class_name.isin(cb.EXCLUDE)]
        crops, items = [], []
        for y in b.itertuples():
            x0, x1 = sorted((max(0, min(cb.FW, y.xmin)), max(0, min(cb.FW, y.xmax))))
            y0, y1 = sorted((max(0, min(cb.FH, y.ymin)), max(0, min(cb.FH, y.ymax))))
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                crops.append(rgb[y0:y1, x0:x1]); items.append((y.class_name, (x0, y0, x1, y1)))
        sims = clip.score(crops, [c for c, _ in items]) if crops else []
        yb = [(f"{c} {s:.2f}", bb, cb.ORANGE if s >= args.threshold else cb.GREY, s < args.threshold,
               3 if s >= args.threshold else 2) for (c, bb), s in zip(items, sims)]
        yb.sort(key=lambda t: t[2] == cb.ORANGE)
        kept = sum(s >= args.threshold for s in sims)
        p4 = cb.draw_panel(frame, yb, f"YOLOE + CLIP ≥ {args.threshold}",
                           f"{len(yb)} YOLOE boxes · {kept} kept (solid) · {len(yb) - kept} dropped (dashed)")
        p5 = cb.combo_panel(frame, combo.get(r.unit_id), f"Gemini {short} · combined (label, CDI, format)")
        panels = [hp, p2, p3, p5, p4]
        row = Image.new("RGB", (len(panels) * (cb.FW + 12) - 12, cb.FH + 48), cb.WHITE)
        for k, im in enumerate(panels):
            row.paste(im, (k * (cb.FW + 12), 0))
        rows.append(row)

    g = 16
    for p in range(0, len(rows), args.per_page):
        chunk = rows[p:p + args.per_page]
        w, hgt = chunk[0].size
        page = Image.new("RGB", (w + 2 * g, len(chunk) * (hgt + g) + g), (236, 235, 230))
        for i, im in enumerate(chunk):
            page.paste(im, (g, g + i * (hgt + g)))
        page.save(out / f"page_{p // args.per_page + 1:02d}.jpg", quality=86)
    print(f"{len(rows)} frames -> {out} ({-(-len(rows) // args.per_page)} pages)")


if __name__ == "__main__":
    main()
