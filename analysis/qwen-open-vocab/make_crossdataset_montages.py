#!/usr/bin/env python3
"""Montages for categories rich in depictions in BabyView but not in the other datasets.

One page per category: BabyView rows by Qwen's format (toy, picture, screen, real), then VEDB (when its store exists),
THINGS images, LVIS box crops and ImageNet images of the mapped items; 8 random exemplars per row, row titles with the
dataset's format shares (format_by_category.csv). Categories: the top --n of results/crossdataset/depiction_gap_categories.csv
(BabyView vs the mean of the other datasets; crossdataset_analysis.ipynb), of depiction_gap_bv_vs_<dataset>.csv with
--vs, or --categories. Pages are numbered by gap rank.

Output (private: BabyView / VEDB frames): out/private/montages_crossdataset/[bv_vs_<dataset>/]<rank>_<category>.jpg
  python make_crossdataset_montages.py [--n 24] [--vs things|lvis|imagenet] [--categories dog cat]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "annotation" / "qwen_pilot"))
from matplotlib import font_manager  # noqa: E402

BV_STORE = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1")
BV_FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")
THINGS_IMG = Path("/data2/dataset/babyview/related_data/things/object_images")
IMAGENET = Path("/data2/j7yang/imagenet-1k/extracted/train")
CD_MAP = HERE.parents[1] / "analysis" / "comparison-datasets" / "outputs" / "mapping"
XD = HERE / "out" / "crossdataset"
OUT = HERE / "out" / "private" / "montages_crossdataset"
TILE, PER_ROW, LEFT = 150, 8, 210
FONT = font_manager.findfont("DejaVu Sans")
FONT_B = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def square(im: Image.Image) -> Image.Image:
    w, h = im.size; s = max(w, h)
    c = Image.new("RGB", (s, s), (245, 244, 240)); c.paste(im, ((s - w) // 2, (s - h) // 2))
    return c.resize((TILE, TILE), Image.LANCZOS)


def bv_rows(cat: str, ix: pd.DataFrame, rng) -> list[tuple[str, list[Image.Image]]]:
    g = ix[ix.category == cat]
    rows = []
    for f in ("toy", "picture", "screen", "real"):
        s = g[g.fmt == f]
        if s.empty:
            continue
        s = s.sample(frac=1, random_state=int(rng.integers(1 << 30))).drop_duplicates("video_id").head(PER_ROW)   # one per video
        ims = []
        for r in s.itertuples():
            fr = BV_FRAMES / r.video_id / f"{int(r.frame_number):05d}.jpg"
            try:
                im = Image.open(fr).convert("RGB")
                pad = .1 * max(r.xmax - r.xmin, r.ymax - r.ymin)
                ims.append(square(im.crop((max(0, r.xmin - pad), max(0, r.ymin - pad), min(im.width, r.xmax + pad), min(im.height, r.ymax + pad)))))
            except Exception:
                pass
        rows.append((f"BabyView: {f}", ims))
    return rows


def things_row(cat, mt, rng):
    ids = mt[mt.category == cat].id.tolist()
    files = [p for i in ids for p in sorted((THINGS_IMG / i).glob("*.jpg"))]
    pick = [files[k] for k in rng.choice(len(files), min(PER_ROW, len(files)), replace=False)] if files else []
    return [square(Image.open(p).convert("RGB")) for p in pick]


_IMNET = {}
def imagenet_files():
    if not _IMNET:                                   # train/ is flat: <wnid>_<n>_<wnid>.JPEG
        import os
        for f in os.listdir(IMAGENET):
            _IMNET.setdefault(f.split("_", 1)[0], []).append(IMAGENET / f)
    return _IMNET


def imagenet_row(cat, mi, rng):
    files = [p for w in mi[mi.category == cat].id for p in imagenet_files().get(w, [])[:300]]
    pick = [files[k] for k in rng.choice(len(files), min(PER_ROW, len(files)), replace=False)] if files else []
    return [square(Image.open(p).convert("RGB")) for p in pick]


def lvis_row(cat, ml, crops, rng):
    c = crops[crops.category_id.isin(set(ml[ml.category == cat].id))]
    if c.empty:
        return []
    c = c.sample(min(PER_ROW, len(c)), random_state=int(rng.integers(1 << 30)))
    out = []
    for r in c.itertuples():
        try:
            out.append(square(Image.open(r.path).convert("RGB").crop((r.xmin, r.ymin, r.xmax, r.ymax))))
        except Exception:
            pass
    return out


def page(cat, rows, shares, out, name, subtitle=""):
    f_t, f_r, f_s = ImageFont.truetype(FONT_B, 18), ImageFont.truetype(FONT_B, 13), ImageFont.truetype(FONT, 11)
    W = LEFT + PER_ROW * (TILE + 4) + 10
    H = 44 + len(rows) * (TILE + 10)
    img = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(img)
    dr.text((10, 10), cat, fill=(11, 11, 11), font=f_t)
    if subtitle:
        dr.text((LEFT, 15), subtitle, fill=(82, 81, 78), font=f_r)
    for k, (title, ims) in enumerate(rows):
        y = 44 + k * (TILE + 10)
        dr.text((10, y + 6), title, fill=(11, 11, 11), font=f_r)
        if title in shares:
            dr.text((10, y + 26), shares[title], fill=(82, 81, 78), font=f_s)
        if not ims:
            dr.text((LEFT, y + TILE // 2 - 6), "none", fill=(138, 137, 132), font=f_s)
        for j, im in enumerate(ims[:PER_ROW]):
            img.paste(im, (LEFT + j * (TILE + 4), y))
    img.save(out / f"{name.replace('/', '_')}.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--vs", choices=["things", "lvis", "imagenet", "vedb"])
    ap.add_argument("--categories", nargs="*")
    args = ap.parse_args()
    gapf = HERE / "results" / "crossdataset" / (f"depiction_gap_bv_vs_{args.vs}.csv" if args.vs else "depiction_gap_categories.csv")
    gap = pd.read_csv(gapf, index_col=0)
    cats = args.categories or list(gap.index[:args.n])
    from build_open_means import basic_of
    ix = pd.read_csv(BV_STORE / "boxes_index.csv.gz", usecols=["video_id", "frame_number", "label", "format", "xmin", "ymin", "xmax", "ymax"])
    ix["label"] = ix.label.astype(str)
    b = basic_of(ix.label.unique()); ix["category"] = ix.label.map(b)
    ix = ix[ix.category.isin(set(cats))]
    ix["fmt"] = ix.format.replace({"photo": "picture", "drawing": "picture"})
    FMT = pd.read_csv(XD / "format_by_category.csv", index_col=[0, 1])
    mt, ml, mi = (pd.read_csv(HERE / "mapping" / f"map_{d}.csv") for d in ("things", "lvis", "imagenet"))
    old = pd.read_csv(CD_MAP / "lvis_crop_manifest.csv").merge(pd.read_csv(CD_MAP / "lvis_crop_metadata.csv.gz",
                                                               usecols=["crop_id", "category_id"]), left_on="image_id", right_on="crop_id")
    ex = np.load(XD / "lvis_extra_crops_sample.npz", allow_pickle=True) if (XD / "lvis_extra_crops_sample.npz").exists() else None
    crops = old[["category_id", "path", "xmin", "ymin", "xmax", "ymax"]]
    if ex is not None:
        b_ = ex["bbox"]
        crops = pd.concat([crops, pd.DataFrame({"category_id": ex["category_id"], "path": ex["path"], "xmin": b_[:, 0],
                                                "ymin": b_[:, 1], "xmax": b_[:, 2], "ymax": b_[:, 3]})])
    out = OUT / f"bv_vs_{args.vs}" if args.vs else OUT
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for k, c in enumerate(cats, 1):
        def sh(d):
            if (d, c) not in FMT.index:
                return ""
            r = FMT.loc[(d, c)]
            return f"toy {r.toy:.0%}  pic {r.picture:.0%}  scr {r.screen:.0%}  (n {int(r.n):,})"
        rows = bv_rows(c, ix, rng)
        rows += [("THINGS", things_row(c, mt, rng)), ("LVIS", lvis_row(c, ml, crops, rng)), ("ImageNet", imagenet_row(c, mi, rng))]
        shares = {"THINGS": sh("things"), "LVIS": sh("lvis"), "ImageNet": sh("imagenet")}
        if rows:
            shares[rows[0][0]] = sh("bv")
        sub = ""
        if c in gap.index:
            sub = f"gap rank {list(gap.index).index(c) + 1}: BabyView depicted {gap.loc[c, 'bv_depicted']:.0%} vs " + (
                f"{args.vs} {gap.loc[c, args.vs + '_depicted']:.0%}" if args.vs else f"others {gap.loc[c, 'others_depicted']:.0%}")
        page(c, rows, shares, out, f"{k:02d}_{c}" if not args.categories else c, sub)
        print("wrote", c, flush=True)


if __name__ == "__main__":
    main()
