#!/usr/bin/env python3
"""Shareable export of the Qwen3.8-27B open-vocabulary annotations on BabyView 2025.2 (BV-main, every 5th 1 fps frame).

Writes to OUT (default /data2/j7yang/qwen_2025.2_release):
  objects.csv.gz   one row per box: video_id, frame_number, obj_idx, label (Qwen's name), label_renamed (generic names such
                   as "toy" renamed from a crop), format (real / toy / drawing / photo / screen), xmin, ymin, xmax, ymax
                   (pixels in the 512 x 910 extracted frame), box_norm (0-1000 [ymin, xmin, ymax, xmax], recomputed from pixels),
                   clip_sim (CLIP crop-label similarity). Repeat boxes (same label, IoU > 0.7 in one frame) are left out,
                   as in the analyses.
  frames.csv.gz    every frame Qwen was run on: video_id, frame_number, parsed (reply was valid JSON), n_objects (incl.
                   repeats), at_cap (30-object cap reached), marching (>= 8 near-identical boxes in a row: a loop)
  README.txt       columns, model settings, counts
  python export_release.py [--out DIR]          # analysis env (vislearnlabpy), nice -n 19
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STORE = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1")
RAW = HERE.parent / "qwen_pilot" / "out" / "runs"
TAGS = ["fullrun2025.2_t1_pp0", "subj_S00320001_t1_pp0"]


def frames():
    rows = {}
    for t in TAGS:
        for line in open(RAW / f"qwen3.8-27b-fp8__detect_open_format__p1_{t}.jsonl"):
            r = json.loads(line)
            ok = r.get("error") is None
            if not ok and r["unit_id"] in rows:
                continue
            objs = []
            if ok:
                try:
                    objs = [o for o in json.loads(r["raw"]).get("objects", []) if len(o.get("box_2d", [])) == 4]
                except Exception:
                    ok = False
            best = run = 1
            for a, b in zip(objs, objs[1:]):
                A, B = a["box_2d"], b["box_2d"]
                hA, wA, hB, wB = A[2] - A[0], A[3] - A[1], B[2] - B[0], B[3] - B[1]
                same = str(a.get("label", "")).strip().lower() == str(b.get("label", "")).strip().lower()
                run = run + 1 if same and abs(hA - hB) <= max(8, .1 * hA) and abs(wA - wB) <= max(8, .1 * wA) else 1
                best = max(best, run)
            v, f = r["unit_id"].rsplit("/", 1)
            rows[r["unit_id"]] = (v, int(f), ok, len(objs), len(objs) >= 30, bool(objs) and best >= 8)
    fr = pd.DataFrame(rows.values(), columns=["video_id", "frame_number", "parsed", "n_objects", "at_cap", "marching"])
    return fr.sort_values(["video_id", "frame_number"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/data2/j7yang/qwen_2025.2_release")
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)

    fr = frames()
    fr.to_csv(out / "frames.csv.gz", index=False)
    print(f"frames: {len(fr):,}", flush=True)

    ob = pd.read_csv(STORE / "boxes_index_renamed.csv.gz", usecols=["video_id", "frame_number", "obj_idx", "label", "format",
                     "xmin", "ymin", "xmax", "ymax", "clip_sim", "label_renamed"])
    W, H = 512, 910
    ob["box_norm"] = [f"[{round(y0 / H * 1000)}, {round(x0 / W * 1000)}, {round(y1 / H * 1000)}, {round(x1 / W * 1000)}]"
                      for x0, y0, x1, y1 in zip(ob.xmin, ob.ymin, ob.xmax, ob.ymax)]
    ob = ob[["video_id", "frame_number", "obj_idx", "label", "label_renamed", "format", "xmin", "ymin", "xmax", "ymax",
             "box_norm", "clip_sim"]]
    ob["clip_sim"] = ob.clip_sim.round(4)
    ob.to_csv(out / "objects.csv.gz", index=False)
    print(f"objects: {len(ob):,}", flush=True)

    fmt = ob.format.value_counts()
    (out / "README.txt").write_text(f"""Qwen3.8-27B open-vocabulary object annotations, BabyView release 2025.2 (BV-main)
=================================================================================

Frames: every 5th frame of the 1 fps extracted frames (one frame per 5 s) of the BV-main videos in 2025.2, excluding
subject S00270001. Frames are 512 x 910 (portrait). Video ids name the subject (S0XXXXXX), date and recording session.

Model: Qwen/Qwen3.8-27B-FP8, run locally with vLLM (no frames left our servers). Prompt: every clearly visible discrete
object, up to 30 per frame, each with an everyday name, a visual format and a box. Temperature 1.0, top_p 0.8, top_k 20,
thinking off. Generic names ("toy", "animal", ...) were re-named by a second Qwen pass on a crop.

objects.csv.gz  {len(ob):,} boxes in {ob[['video_id', 'frame_number']].drop_duplicates().shape[0]:,} frames
  video_id, frame_number     frame = extracted_frames_1fps/<video_id>/<frame_number:05d>.jpg
  obj_idx                    position in Qwen's reply for that frame
  label                      Qwen's open-vocabulary name (lowercase)
  label_renamed              label after a second pass on generic names ("toy", "animal", ...): the specific thing
                             named from a crop; equal to label otherwise
  format                     real / toy / drawing / photo / screen  ({', '.join(f'{k} {v / len(ob):.1%}' for k, v in fmt.items())})
  xmin, ymin, xmax, ymax     box in pixels of the 512 x 910 frame
  box_norm                   the box as [ymin, xmin, ymax, xmax] on Qwen's 0-1000 grid (recomputed from pixels;
                             may differ from Qwen's exact reply by ~1 unit)
  clip_sim                   CLIP ViT-B/32 cosine between the crop and "a photo of a <label>" (diagnostic only)
  Repeat boxes (same label, IoU > 0.7 within a frame) are left out, as in the analyses.

frames.csv.gz   {len(fr):,} frames Qwen was run on (including frames with no objects)
  parsed      the reply was valid JSON ({fr.parsed.mean():.2%})
  n_objects   boxes in the reply, repeats included
  at_cap      the 30-object cap was reached ({fr.at_cap.mean():.2%})
  marching    >= 8 near-identical boxes in a row (a generation loop) ({fr.marching.mean():.2%})

Validation (gold set v2, two RAs correcting Qwen's boxes on all 1,000 frames): precision 0.89, recall 0.74
(0.81 without body parts), name kept 0.86, format kept 0.99. Qwen vs people box F1 0.81; inter-rater 0.80.

Data use: derived from BabyView video; share only under the BabyView data-use agreement. Labels can include text read
from the frame. Code: object-detection repo, annotation/qwen_pilot/ (run_qwen_pilot.py, export_release.py).
""")
    print("README written")


if __name__ == "__main__":
    main()
