#!/usr/bin/env python3
"""Montages of the generic-label second pass (qwen_generic_crops.py): what Qwen renamed "toy" / "animal" / ... to.

One page per (run, original generic category): one row per new name (the most frequent --rows names, then "unclear" and
"none"), 8 random crops per row (at most one per video), each the context crop Qwen saw with the box outlined in red.
Row titles: name, boxes, share of the category. Reads the merged output or, while the pass runs, its shard files.

Output (private: frames): out/private/montages_generic_rename/<tag>_<category>.jpg
  python make_generic_rename_montages.py [--tags fullrun_bv fullrun_vedb] [--rows 16]      # either env
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"
OUT = PRIV / "montages_generic_rename"
TILE, PER_ROW, LEFT, PAD = 150, 8, 230, 0.25


def font(bold=False, size=13):
    from matplotlib import font_manager
    p = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(p, size)


def results(tag: str) -> pd.DataFrame:
    merged = PRIV / f"qwen_generic_crops_{tag}.jsonl"
    files = [merged] if merged.exists() else sorted(PRIV.glob(f"qwen_generic_crops_{tag}.jsonl.shard*"))
    rows = [json.loads(l) for f in files if not f.name.endswith(".merged") for l in open(f)]
    r = pd.DataFrame(rows).drop_duplicates(["unit_id", "obj"])
    b = pd.read_csv(PRIV / f"qwen_generic_boxes_{tag}.csv.gz")
    return r.merge(b[["unit_id", "obj", "ymin", "xmin", "ymax", "xmax", "frame_path"]], on=["unit_id", "obj"])


def tile(r) -> Image.Image:
    im = Image.open(r.frame_path).convert("RGB")
    w, h = im.size
    bx0, bx1, by0, by1 = r.xmin / 1000 * w, r.xmax / 1000 * w, r.ymin / 1000 * h, r.ymax / 1000 * h
    side = max(bx1 - bx0, by1 - by0) * (1 + 2 * PAD)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    L, T = max(0, cx - side / 2), max(0, cy - side / 2)
    c = im.crop((round(L), round(T), round(min(w, cx + side / 2)), round(min(h, cy + side / 2))))
    ImageDraw.Draw(c).rectangle([bx0 - L, by0 - T, bx1 - L, by1 - T], outline=(230, 30, 30), width=max(2, round(c.width / 100)))
    s = max(c.size)
    sq = Image.new("RGB", (s, s), (245, 244, 240)); sq.paste(c, ((s - c.width) // 2, (s - c.height) // 2))
    return sq.resize((TILE, TILE), Image.LANCZOS)


def page(tag, cat, g, n_rows, rng):
    names = g.name.fillna("(error)").value_counts()
    top = [n for n in names.index if n not in ("unclear", "none")][:n_rows] + [n for n in ("unclear", "none") if n in names.index]
    f_t, f_r, f_s = font(True, 18), font(True, 13), font(False, 11)
    W, H = LEFT + PER_ROW * (TILE + 4) + 10, 50 + len(top) * (TILE + 10)
    img = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(img)
    dr.text((10, 12), f'{tag}: Qwen\'s "{cat}" boxes renamed ({len(g):,} boxes; {len(names)} names)', fill=(11, 11, 11), font=f_t)
    g = g.assign(video=g.unit_id.str.rsplit("/", n=1).str[0])
    for k, n in enumerate(top):
        y = 50 + k * (TILE + 10)
        dr.text((10, y + 6), n[:28], fill=(11, 11, 11), font=f_r)
        dr.text((10, y + 26), f"{names[n]:,} boxes, {names[n] / len(g):.1%}", fill=(82, 81, 78), font=f_s)
        s = g[g.name.fillna("(error)") == n].sample(frac=1, random_state=int(rng.integers(1 << 30))).drop_duplicates("video").head(PER_ROW)
        for j, r in enumerate(s.itertuples()):
            try:
                img.paste(tile(r), (LEFT + j * (TILE + 4), y))
            except Exception:
                pass
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{tag}_{cat}.jpg"
    img.save(p, quality=88)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=["fullrun_bv", "fullrun_vedb"])
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--min-boxes", type=int, default=200, help="skip original categories with fewer boxes")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    for tag in args.tags:
        try:
            r = results(tag)
        except (FileNotFoundError, ValueError, KeyError):
            print(f"{tag}: no results yet"); continue
        summ = []
        for cat, g in r.groupby("cat"):
            nm = g.name.fillna("(error)")
            summ.append(dict(category=cat, boxes=len(g), named=(~nm.isin(["unclear", "none", "(error)"])).mean(),
                             unclear=(nm == "unclear").mean(), none=(nm == "none").mean(), distinct_names=nm.nunique(),
                             top_names=", ".join(f"{a} ({b})" for a, b in nm[~nm.isin(['unclear', 'none'])].value_counts().head(10).items())))
            if len(g) >= args.min_boxes:
                print("wrote", page(tag, cat, g, args.rows, rng), flush=True)
        pd.DataFrame(summ).sort_values("boxes", ascending=False).to_csv(OUT / f"{tag}_summary.csv", index=False)
        print(pd.DataFrame(summ).sort_values("boxes", ascending=False).drop(columns="top_names").round(3).to_string())


if __name__ == "__main__":
    main()
