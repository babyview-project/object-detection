#!/usr/bin/env python3
"""Map the public gold crops (valid5921_public) back to their source frames and YOLOE boxes.

Public crop ids are {category}_{sha256(category:stem)[:16]}; the original stems
are in the non-public valid7018 embeddings manifest. A stem
{category}_{conf}_{subject}_{gcp_name}[_processed]_{frame} gives the 868-hour
frame, and the YOLOE row with the same video, frame, class and confidence
(3 decimals) gives the box. Ties are broken by the box size closest to the crop.

Output (git-ignored, contains subject ids and dates): out/private/gold_frames.csv
"""
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from run_gold_pilot import GOLD_ZIP, OUT_DIR

VAR = Path("/home/j7yang/babyview-projects/ccn-2026-object-variability/data/shared_data_ccn_2026")
EMB_ZIP = VAR / "embeddings" / "valid7018_bv_embeddings.zip"
FRAMES = Path("/ccn2a/dataset/babyview/868_hours/sampled_frames")
YOLOE = Path("/ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_allframes_1fps")
STEM = re.compile(r"^(?P<cat>.+?)_(?P<conf>[\d.]+)_(?P<subj>\d+)_(?P<gcp>.+?)(?:_processed)?_(?P<frame>\d+)$")


def dir_index(root):
    """lowercase video name (without _processed) -> directory."""
    return {p.name.lower().removesuffix("_processed"): p for p in root.iterdir()}


def main():
    pub = zipfile.ZipFile(GOLD_ZIP)
    crops = {n.split("/")[-1][:-4]: n for n in pub.namelist() if n.endswith(".jpg")}
    rows = list(csv.DictReader(io.TextIOWrapper(zipfile.ZipFile(EMB_ZIP).open("manifest.csv"))))
    frames_dirs, yoloe_dirs = dir_index(FRAMES), dir_index(YOLOE)

    recs = []
    for r in rows:
        cat, stem = r["category"].strip().lower(), r["stem"].strip()
        anon = f"{cat}_{hashlib.sha256(f'{cat}:{stem.lower()}'.encode()).hexdigest()[:16]}"
        if anon not in crops:
            continue
        m = STEM.match(stem)
        video = f"{m['subj']}_{m['gcp']}".lower()
        w, h = Image.open(io.BytesIO(pub.read(crops[anon]))).size
        recs.append({"crop_id": anon, "category": cat, "stem": stem, "subject": m["subj"],
                     "video": video, "frame_id": m["frame"], "conf_stem": float(m["conf"]),
                     "crop_w": w, "crop_h": h})
    g = pd.DataFrame(recs)
    g["frame_path"] = [str(frames_dirs[v] / f"{f}.jpg") if v in frames_dirs else None
                       for v, f in zip(g.video, g.frame_id)]

    boxes = []
    for video, d in g.groupby("video"):
        if video not in yoloe_dirs:
            continue
        b = pd.read_csv(yoloe_dirs[video] / "bounding_box_predictions.csv", dtype={"frame_id": str})
        b["frame_id"] = b.frame_id.str.zfill(5)
        b = b[b.frame_id.isin(set(d.frame_id))]
        for r in d.itertuples():
            c = b[(b.frame_id == r.frame_id) & (b.class_name == r.category)
                  & (np.abs(b.confidence - r.conf_stem) < 0.0006)]
            if len(c) == 0:
                continue
            err = np.abs((c.xmax - c.xmin) - r.crop_w) + np.abs((c.ymax - c.ymin) - r.crop_h)
            best = c.loc[err.idxmin()]
            boxes.append({"crop_id": r.crop_id, "xmin": best.xmin, "ymin": best.ymin, "xmax": best.xmax,
                          "ymax": best.ymax, "confidence": best.confidence, "n_candidates": len(c),
                          "size_err_px": float(err.min()), "n_yoloe_boxes_in_frame": int((b.frame_id == r.frame_id).sum())})
    g = g.merge(pd.DataFrame(boxes), on="crop_id", how="left")

    out = OUT_DIR / "private"
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "gold_frames.csv", index=False)
    print(f"crops {len(g)} | frame found {g.frame_path.notna().sum()} | box matched {g.xmin.notna().sum()} "
          f"| ties {(g.n_candidates > 1).sum()} | size err >2px {(g.size_err_px > 2).sum()}")
    print(f"unique frames {g.frame_path.nunique()} | videos {g.video.nunique()} | subjects {g.subject.nunique()}")
    print(g.size_err_px.describe().round(2).to_dict())
    with Image.open(g.frame_path.dropna().iloc[0]) as im:
        print("frame size", im.size)


if __name__ == "__main__":
    main()
