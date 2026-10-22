#!/usr/bin/env python3
"""Rater-deleted YOLOE detections (Mira / Dora CVAT exports) as a second pilot set.

The CVAT tasks in annotation/cvat_3000_frames/annotation_data/ cover the 3000 inverse-proportionally
sampled 868 h frames; each frame was sampled for ONE YOLOE detection
(/ccn2/dataset/babyview/outputs_20250312/3000_inverse_proportional_sampled_frames.csv),
pre-filled, and the rater edited only that box (checked 2026-10-19: in 99% of labelled
frames the rater box matches the sampled box at IoU >= 0.9): kept the label (44%),
relabelled it to the right CDI word (56%), or deleted it. A label file with no line
at all is a deletion: the rater judged that YOLOE box not to be any recognisable CDI
object. The frames themselves are NOT empty (other objects are usually visible), so
these are negatives for box-level checks (CLIP filter, Gemini verify), not for
whole-frame detection.

Outputs (out/private/, git-ignored: video names, subject ids, frames)
  empty_frames.csv        unit_id (video/frame_id), video, frame_id, subject, rater, export,
                          frame_path, n_yoloe (boxes, person/picture excluded), yoloe_labels
  empty_frames_units.txt  unit ids, for run_gold_pilot.py --units
  empty_frames_sheet_XX.jpg  contact sheets with YOLOE's boxes drawn (visual check)

  python build_empty_frames.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ANN = HERE.parent / "cvat_3000_frames" / "annotation_data"
PRIV = HERE / "out" / "private"
FRAMES = Path("/ccn2/dataset/babyview/outputs_20250312/sampled_frames")
YOLOE = Path("/ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_allframes_1fps")
EXCLUDE = {"person", "picture"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main():
    rows, seen = [], {}
    for d in sorted(p for p in ANN.iterdir() if p.is_dir()):
        for f in d.rglob("*.txt"):
            if f.name in ("obj.names", "train.txt"):
                continue
            seen.setdefault(f.stem, []).append(d.name)
            if any(l.strip() for l in open(f)):
                continue
            stem = f.stem  # <video>_processed_<frame> or <video>_<frame>
            video, frame_id = stem.rsplit("_", 1)
            video = video.removesuffix("_processed")
            rows.append({"stem": stem, "video": video.lower(), "frame_id": frame_id.zfill(5),
                         "subject": video.split("_")[0], "rater": d.name.split("_")[0], "export": d.name})
    e = pd.DataFrame(rows)
    e["n_label_files"] = e.stem.map(lambda s: len(seen[s]))
    e["unit_id"] = e.video + "/" + e.frame_id
    fdirs = {p.name.lower().removesuffix("_processed"): p for p in FRAMES.iterdir()}
    ydirs = {p.name.lower().removesuffix("_processed"): p for p in YOLOE.iterdir()}
    e["frame_path"] = [str(fdirs[v] / f"{f}.jpg") if v in fdirs else None for v, f in zip(e.video, e.frame_id)]
    assert e.frame_path.map(lambda p: p and Path(p).exists()).all(), "missing frames"

    boxes = {}
    for v, g in e.groupby("video"):
        b = pd.read_csv(ydirs[v] / "bounding_box_predictions.csv", dtype={"frame_id": str})
        b = b[b.xmin.notna()].copy()
        b["frame_id"] = b.frame_id.str.zfill(5)
        b["class_name"] = b.class_name.astype(str).str.strip().str.lower()
        b = b[b.frame_id.isin(set(g.frame_id)) & ~b.class_name.isin(EXCLUDE)]
        for fid, bb in b.groupby("frame_id"):
            boxes[f"{v}/{fid}"] = bb
    e["n_yoloe"] = e.unit_id.map(lambda u: len(boxes.get(u, [])))
    e["yoloe_labels"] = e.unit_id.map(lambda u: ";".join(sorted(set(boxes[u].class_name))) if u in boxes else "")
    e = e.sort_values(["rater", "unit_id"]).reset_index(drop=True)
    e.to_csv(PRIV / "empty_frames.csv", index=False)
    (PRIV / "empty_frames_units.txt").write_text("\n".join(e.unit_id) + "\n")

    f = ImageFont.truetype(FONT, 11)
    tw, th, cols, per = 171, 304, 8, 32
    for p in range(0, len(e), per):
        chunk = e.iloc[p:p + per]
        sheet = Image.new("RGB", (cols * (tw + 6) + 6, -(-len(chunk) // cols) * (th + 22) + 6), (236, 235, 230))
        for i, r in enumerate(chunk.itertuples()):
            im = Image.open(r.frame_path).convert("RGB")
            d = ImageDraw.Draw(im)
            for b in boxes.get(r.unit_id, pd.DataFrame()).itertuples():
                d.rectangle((b.xmin, b.ymin, b.xmax, b.ymax), outline=(235, 104, 52), width=4)
            im = im.resize((tw, th))
            x, y = 6 + (i % cols) * (tw + 6), 6 + (i // cols) * (th + 22)
            sheet.paste(im, (x, y))
            ImageDraw.Draw(sheet).text((x, y + th + 3), f"{r.rater} · YOLOE {r.n_yoloe}", font=f, fill=(40, 40, 40))
        sheet.save(PRIV / f"empty_frames_sheet_{p // per + 1:02d}.jpg", quality=85)
    print(f"{len(e)} empty frames ({e.rater.value_counts().to_dict()}), {e.subject.nunique()} children, "
          f"{e.video.nunique()} videos; YOLOE boxes per frame: mean {e.n_yoloe.mean():.1f}, "
          f"none on {(e.n_yoloe == 0).sum()}; also in the gold pilot: "
          f"{e.unit_id.isin(set(pd.read_csv(PRIV / 'gold_frames.csv', dtype=str).pipe(lambda g: g.video + '/' + g.frame_id))).sum()}")


if __name__ == "__main__":
    main()
