#!/usr/bin/env python3
"""Category audit, step 1: sample Qwen boxes per category and dataset and render the crops RAs judge.

Why: the 1,000-frame gold set covers BabyView only, and most heavily depicted categories have < 5 boxes in it. This audit
checks, per category and dataset, (1) whether Qwen's name is right and (2) whether the object is real or depicted.

Categories: every core category (data/included_categories_qwen_core.txt) that is >= 30% depicted in BabyView or VEDB
(analysis/qwen-open-vocab/results/crossdataset_renamed/depiction_gap_bv_vs_vedb.csv), plus the core categories the gold
set flagged as weakest (cat, teddy bear, coat, fish, pig, car). Boxes: N per category and dataset, drawn at random from
all of Qwen's boxes for the category (renamed labels; VEDB adults), spread over children / sessions (round-robin over
units, one box per video before any video repeats). No size or person filter, so the sample matches what the analyses use.

Outputs (private: children's homes; never commit, share bundles only via IRB-approved storage):
  out/private/items.csv          item_id (opaque), dataset, unit, video_id, frame_number, box, category, qwen_format
  out/private/img/<id>_crop.jpg  the box with 15% context on each side, box outlined, long side <= 360 px
  out/private/img/<id>_frame.jpg the whole frame with the box outlined in magenta, long side <= 720 px

  /data/j7yang/anaconda3/bin/python sample_audit.py [--n 30] [--seed 0]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
Q = REPO / "analysis" / "qwen-open-vocab"
sys.path.insert(0, str(Q)); sys.path.insert(0, str(REPO / "annotation" / "qwen_pilot"))
os.environ.setdefault("QOV_LABELS", "renamed")
import variant as V  # noqa: E402

OUT = HERE / "out" / "private"
STORE = {"bv": Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1"),
         "vedb": Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv_adults")}
FRAMES = {"bv": lambda v, f: Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps") / v / f"{int(f):05d}.jpg",
          "vedb": lambda v, f: Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps_adults") / v / f"{int(f):08d}.jpg"}
WEAK = ["cat", "teddy bear", "coat", "fish", "pig", "car"]          # weakest core categories in the gold set (2026-11-15)
MIN_DEPICTED = 0.30
MAGENTA = (236, 0, 140)


def categories() -> list[str]:
    core = [l.strip() for l in open(REPO / "data" / "included_categories_qwen_core.txt") if l.strip()]
    g = pd.read_csv(Q / "results" / "crossdataset_renamed" / "depiction_gap_bv_vs_vedb.csv").set_index("category").reindex(core)
    dep = g[(g.bv_depicted >= MIN_DEPICTED) | (g.vedb_depicted >= MIN_DEPICTED)].index
    return sorted(set(dep) | (set(WEAK) & set(core)))


def load(ds: str, cats: set[str]) -> pd.DataFrame:
    from build_open_means import basic_of
    ix = V.read_index(STORE[ds], usecols=["video_id", "frame_number", "label", "format", "xmin", "ymin", "xmax", "ymax"])
    ix["label"] = ix.label.astype(str); ix["video_id"] = ix.video_id.astype(str)
    ix["category"] = ix.label.map(basic_of(ix.label.unique()))
    ix = ix[ix.category.isin(cats)].copy()
    ix["unit"] = np.where(ix.video_id.str.startswith("session_"), ix.video_id.str.split("_").str[1], ix.video_id.str[:9])
    return ix


def spread(d: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """n boxes, round-robin over units (children / sessions), one box per video before any video repeats."""
    d = d.sample(frac=1, random_state=int(rng.integers(1 << 30))).copy()
    d["v_rank"] = d.groupby("video_id").cumcount()
    d = d.sort_values("v_rank", kind="stable")
    d["u_rank"] = d.groupby("unit").cumcount()
    return d.sort_values(["u_rank", "v_rank"], kind="stable").head(n).drop(columns=["v_rank", "u_rank"])


def render(ds: str, r, iid: str) -> bool:
    try:
        im = Image.open(FRAMES[ds](r.video_id, r.frame_number)).convert("RGB")
    except Exception:
        return False
    x0, y0, x1, y1 = r.xmin, r.ymin, r.xmax, r.ymax
    w, h = x1 - x0, y1 - y0
    L, T = max(0, int(x0 - .15 * w)), max(0, int(y0 - .15 * h))
    R, B = min(im.width, int(x1 + .15 * w) + 1), min(im.height, int(y1 + .15 * h) + 1)
    c = im.crop((L, T, R, B)); d = ImageDraw.Draw(c)
    lw = max(1, round(max(c.size) / 150))
    d.rectangle((x0 - L, y0 - T, x1 - L, y1 - T), outline=MAGENTA, width=lw)
    s = 360 / max(c.size)
    if s < 1:
        c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)
    elif s > 1:                                                        # small boxes: enlarge up to 360 px, at most 4x
        s = min(s, 4); c = c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS)
    c.save(OUT / "img" / f"{iid}_crop.jpg", quality=90)
    f = im.copy(); d = ImageDraw.Draw(f)
    d.rectangle((x0, y0, x1, y1), outline=MAGENTA, width=max(2, round(max(f.size) / 200)))
    s = 720 / max(f.size)
    if s < 1:
        f = f.resize((round(f.width * s), round(f.height * s)), Image.LANCZOS)
    f.save(OUT / "img" / f"{iid}_frame.jpg", quality=88)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30, help="boxes per category and dataset")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    cats = categories(); print(len(cats), "categories:", ", ".join(cats), flush=True)
    rng = np.random.default_rng(a.seed); rows = []
    for ds in ("bv", "vedb"):
        ix = load(ds, set(cats)); print(ds, f"{len(ix):,} boxes in the audit categories", flush=True)
        for cat in cats:
            d = ix[ix.category == cat]
            got = 0
            for r in spread(d, a.n * 2, rng).itertuples():                # spares for unreadable frames
                if got == a.n:
                    break
                iid = hashlib.sha1(f"{ds}|{r.video_id}|{r.frame_number}|{r.xmin}|{r.ymin}|{cat}".encode()).hexdigest()[:12]
                if render(ds, r, iid):
                    rows.append(dict(item_id=iid, dataset=ds, unit=r.unit, video_id=r.video_id, frame_number=int(r.frame_number),
                                     xmin=r.xmin, ymin=r.ymin, xmax=r.xmax, ymax=r.ymax, category=cat, qwen_label=r.label,
                                     qwen_format=str(r.format), pool=len(d)))
                    got += 1
            if got < a.n:
                print(f"  {ds} {cat}: only {got} of {a.n} ({len(d)} boxes)", flush=True)
    it = pd.DataFrame(rows); it.to_csv(OUT / "items.csv", index=False)
    print(f"wrote {len(it):,} items:", it.groupby("dataset").size().to_dict())


if __name__ == "__main__":
    main()
