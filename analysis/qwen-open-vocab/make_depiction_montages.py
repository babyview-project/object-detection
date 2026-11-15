#!/usr/bin/env python3
"""Montages of one child's objects by visual format (Qwen setup B), one page per CDI category.

Rows = Qwen's format label (real, toy, drawing, photo, screen); each row shows up to --per-row crops
of boxes whose open label maps to the category (qwen_data `cdi`), sampled at random with at most one
crop per video where possible (so one long video does not fill a row). Each crop keeps some context
(the box is outlined) and is captioned with Qwen's exact label and the child's age in months.
Rows with no boxes say so, with the count.

Output (private: frames of the child): out/private/depiction_montages/<subject>/<category>.jpg

  python make_depiction_montages.py                                  # default categories
  python make_depiction_montages.py --categories dog cat car --per-row 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import sys
HERE = Path(__file__).resolve().parent
QP = HERE.parents[1] / "annotation" / "qwen_pilot"        # pipeline: loaders, embeddings, manifest
sys.path.insert(0, str(QP))
import qwen_data as qd  # noqa: E402

HERE = Path(__file__).resolve().parent
FORMATS = ["real", "toy", "drawing", "photo", "screen"]
FMT_COLOR = {"real": (82, 81, 78), "toy": (237, 161, 0), "drawing": (235, 104, 52),
             "photo": (27, 175, 122), "screen": (42, 120, 214)}
DEFAULT = ["dog", "cat", "bear", "elephant", "horse", "bird", "fish", "car", "truck", "airplane", "ball", "tree"]
CELL, PAD, CAP, LEFT = 190, 0.15, 34, 130
FONT = font_manager.findfont("DejaVu Sans")
FONT_B = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def sample(d: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Up to n boxes, one per video first, larger boxes preferred within a video."""
    d = d.sample(frac=1, random_state=rng).sort_values("area", ascending=False)
    first = d.drop_duplicates("video_id")
    pick = first.sample(min(n, len(first)), random_state=rng)
    if len(pick) < n:
        pick = pd.concat([pick, d.drop(pick.index).head(n - len(pick))])
    return pick.sort_values("age_months")


def crop_cell(r, font) -> Image.Image:
    im = Image.open(r.frame_path).convert("RGB")
    w, h = im.size
    x0, x1, y0, y1 = r.xmin / 1000 * w, r.xmax / 1000 * w, r.ymin / 1000 * h, r.ymax / 1000 * h
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L, T = cx - side / 2, cy - side / 2
    c = Image.new("RGB", (round(side), round(side)), (245, 244, 240))
    c.paste(im.crop((round(L), round(T), round(L + side), round(T + side))), (0, 0))
    s = CELL / side
    c = c.resize((CELL, CELL), Image.LANCZOS)
    dr = ImageDraw.Draw(c)
    dr.rectangle([(x0 - L) * s, (y0 - T) * s, (x1 - L) * s, (y1 - T) * s], outline=FMT_COLOR[r.format], width=2)
    cell = Image.new("RGB", (CELL, CELL + CAP), "white")
    cell.paste(c, (0, 0))
    dr = ImageDraw.Draw(cell)
    lab = r.label if len(r.label) <= 26 else r.label[:25] + "…"
    dr.text((3, CELL + 2), lab, fill=(11, 11, 11), font=font)
    dr.text((3, CELL + 17), f"{r.age_months:.1f} mo", fill=(110, 109, 105), font=font)
    return cell


def page(cat: str, o: pd.DataFrame, per_row: int, rng, out: Path):
    f_lab, f_title = ImageFont.truetype(FONT, 12), ImageFont.truetype(FONT_B, 20)
    f_row, f_n = ImageFont.truetype(FONT_B, 15), ImageFont.truetype(FONT, 12)
    rows = FORMATS
    W = LEFT + per_row * (CELL + 6) + 10
    H = 50 + len(rows) * (CELL + CAP + 14)
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    d = o[o.cdi == cat]
    dr.text((10, 12), f"{cat}  ·  {len(d):,} boxes in {d.unit_id.nunique():,} frames ({SUBJECT})",
            fill=(11, 11, 11), font=f_title)
    for i, fmt in enumerate(rows):
        y = 50 + i * (CELL + CAP + 14)
        df = d[d.format == fmt]
        dr.rectangle([10, y, 16, y + CELL], fill=FMT_COLOR[fmt])
        dr.text((24, y + 4), fmt, fill=(11, 11, 11), font=f_row)
        dr.text((24, y + 26), f"{len(df):,} boxes", fill=(82, 81, 78), font=f_n)
        dr.text((24, y + 42), f"{df.video_id.nunique():,} videos", fill=(82, 81, 78), font=f_n)
        if df.empty:
            dr.text((LEFT, y + CELL / 2 - 8), "none", fill=(138, 137, 132), font=f_n)
            continue
        for j, r in enumerate(sample(df, per_row, rng).itertuples()):
            img.paste(crop_cell(r, f_lab), (LEFT + j * (CELL + 6), y))
    img.save(out / f"{cat}.jpg", quality=88)


def main():
    global SUBJECT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=DEFAULT, help="CDI words (valid129 / CDI list)")
    ap.add_argument("--subject", default="S00320001")
    ap.add_argument("--tag", default="subj_S00320001_t1_pp0")
    ap.add_argument("--per-row", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    SUBJECT = args.subject
    rec = qd.load_records("detect_open_format", args.tag)
    fr = qd.frames_table(rec)
    o = qd.dedup(qd.objects_table(rec)).merge(fr[["unit_id", "age_months"]], on="unit_id")
    m = pd.read_csv(QP.parent / "gemini_full_run" / "out" / "private" / "manifest.csv", usecols=["unit_id", "frame_path"])
    o = o.merge(m, on="unit_id")
    o = o[o.format.isin(FORMATS)]
    out = HERE / "out" / "private" / "depiction_montages" / args.subject
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    for cat in args.categories:
        if not (o.cdi == cat).any():
            print(f"{cat}: no boxes"); continue
        page(cat, o, args.per_row, rng, out)
        print(f"{cat}: " + ", ".join(f"{f} {int((o[o.cdi == cat].format == f).sum())}" for f in FORMATS))
    print("wrote", out)


if __name__ == "__main__":
    main()
