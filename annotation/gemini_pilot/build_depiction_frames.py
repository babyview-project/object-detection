#!/usr/bin/env python3
"""Depiction-heavy frames with human format labels, for open-vocabulary Gemini runs.

Source: Mira / Dora CVAT exports in annotation/cvat_3000_frames/annotation_data/. Besides the object
box, raters tagged its visual format with a box on the same object (96% at IoU >= 0.9):
real_format, toy_format, photo_format, drawing_format, video_format.

Sample (seeded): up to --per-format frames containing each non-real format (a frame is
used once), plus the same number of frames whose tags are all real_format (control:
how often Gemini uses depiction words for real objects).

Outputs (out/private/, git-ignored)
  depiction_frames.csv   unit_id, video, frame_id, frame_path, stratum, formats, objects
  depiction_boxes.csv    unit_id, label, format, xmin, ymin, xmax, ymax (pixels, 512x910)
  depiction_units.txt    unit ids for run_gold_pilot.py --frames / --units

  python build_depiction_frames.py [--per-format 16]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANN = HERE.parent / "cvat_3000_frames" / "annotation_data"
PRIV = HERE / "out" / "private"
FRAMES = Path("/ccn2/dataset/babyview/outputs_20250312/sampled_frames")
W, H = 512, 910
FORMATS = ["toy_format", "photo_format", "drawing_format", "video_format"]
SEED = 20261019


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-format", type=int, default=16)
    args = ap.parse_args()
    boxes = []
    for d in sorted(p for p in ANN.iterdir() if p.is_dir()):
        names = open(next(d.rglob("obj.names"))).read().split("\n")
        for f in d.rglob("*.txt"):
            if f.name in ("obj.names", "train.txt"):
                continue
            video, fid = f.stem.rsplit("_", 1)
            unit = f"{video.removesuffix('_processed').lower()}/{fid.zfill(5)}"
            objs, tags = [], []
            for l in open(f):
                p = l.split()
                if not p:
                    continue
                cx, cy, w, h = map(float, p[1:5])
                b = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
                (objs if int(p[0]) < 295 else tags).append((names[int(p[0])], b))
            for lab, b in objs:  # format = the tag on the same box
                m = max(tags, key=lambda t: iou(t[1], b), default=None)
                fmt = m[0] if m and iou(m[1], b) >= 0.5 else ""
                boxes.append({"unit_id": unit, "rater": d.name.split("_")[0], "label": lab, "format": fmt,
                              "xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]})
    bx = pd.DataFrame(boxes).drop_duplicates(["unit_id", "label", "format"])
    fr = bx.groupby("unit_id").agg(formats=("format", lambda x: ";".join(sorted(set(x) - {""}))),
                                   objects=("label", lambda x: ";".join(x)))
    rng = np.random.default_rng(SEED)
    used, pick = set(), []
    for fm in FORMATS:  # rarest first would also work; video has only ~21 frames
        cand = [u for u in fr.index[fr.formats.str.contains(fm)] if u not in used]
        take = list(rng.permutation(cand)[:args.per_format])
        used.update(take)
        pick += [(u, fm.removesuffix("_format")) for u in take]
    real = [u for u in fr.index[fr.formats == "real_format"] if u not in used]
    pick += [(u, "real_only") for u in rng.permutation(real)[:args.per_format]]

    s = pd.DataFrame(pick, columns=["unit_id", "stratum"]).join(fr, on="unit_id")
    s["video"], s["frame_id"] = s.unit_id.str.split("/").str[0], s.unit_id.str.split("/").str[1]
    fdirs = {p.name.lower().removesuffix("_processed"): p for p in FRAMES.iterdir()}
    s["frame_path"] = [str(fdirs[v] / f"{f}.jpg") for v, f in zip(s.video, s.frame_id)]
    assert s.frame_path.map(lambda p: Path(p).exists()).all()
    s["category"] = s.stratum  # run_gold_pilot.subsample() groups by category
    s.to_csv(PRIV / "depiction_frames.csv", index=False)
    bx[bx.unit_id.isin(set(s.unit_id))].to_csv(PRIV / "depiction_boxes.csv", index=False)
    (PRIV / "depiction_units.txt").write_text("\n".join(s.unit_id) + "\n")
    b = bx[bx.unit_id.isin(set(s.unit_id))]
    print(f"{len(s)} frames: {s.stratum.value_counts().to_dict()}; boxes {len(b)} by format "
          f"{b.format.replace('', 'untagged').value_counts().to_dict()}")


if __name__ == "__main__":
    main()
