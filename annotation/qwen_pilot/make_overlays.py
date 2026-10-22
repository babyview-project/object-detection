#!/usr/bin/env python3
"""Box overlays for the Qwen pilot, and Qwen | Gemini side by side, on one frame set.

Per frame (out/private/overlays_<set>/, git-ignored: frames show children's homes):
  qwen/<n>_<unit>.jpg          Qwen3.8-27B boxes on the frame
  side_by_side/<n>_<unit>.jpg  Qwen (left) | Gemini 2.5 Flash (right), same frame
  pages/page_XXX.jpg           6 side-by-side pairs per page
  index.csv                    n, unit_id, objects per model, looping flag, labels
Box colour = format (green real, amber toy, red photo, pink drawing, blue screen);
text = "label · format". Frames are numbered in the order of the frame list.

  python make_overlays.py [--set gold_v2] [--per-page 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "gemini_pilot"
sys.path.insert(0, str(PILOT))
import compare_boxes_montage as cb  # noqa: E402  (draw_panel, colours)
from compare_qwen_gemini import FULLRUN, load, looping, objects  # noqa: E402

PRIV = PILOT / "out" / "private"
SETS = {  # frame list, Qwen file, Gemini file
    "gold_v2": (HERE.parent / "gold_set" / "out" / "private" / "gold_v2_frames.csv",
                PRIV / "qwen3.8-27b-fp8__detect_open_format__p1_gold_v2_frames_cap30.jsonl",
                PRIV / "gemini-2.5-flash__detect_open_format__p1_gold_v2_frames_cap30.jsonl"),
    "depiction": (PRIV / "depiction_frames.csv",
                  PRIV / "qwen3.8-27b-fp8__detect_open_format__p1_depiction_frames.jsonl",
                  PRIV / "gemini-2.5-flash__detect_combo__p1_depiction_frames.jsonl"),
    "fullrun500": (None, PRIV / "qwen3.8-27b-fp8__detect_open_format__p1_fullrun500.jsonl", FULLRUN),
}


def panel(frame, rec, title):
    o = objects(rec)
    if o is None:
        return cb.draw_panel(frame, [], title, "no parsed reply" if rec else "no reply")
    boxes = []
    for x in o:
        y0, x0, y1, x1 = (min(1000, max(0, v)) for v in x["box_2d"])
        if x1 <= x0 or y1 <= y0:
            continue
        fm = str(x.get("format", "")).lower()
        boxes.append((f"{x['label'].strip().lower()} · {fm}" if fm else x["label"].strip().lower(),
                      (x0 * cb.FW / 1000, y0 * cb.FH / 1000, x1 * cb.FW / 1000, y1 * cb.FH / 1000),
                      cb.FORMAT_COLORS.get(fm, cb.GREY), False, 3))
    n = {k: sum(b[2] == c for b in boxes) for k, c in cb.FORMAT_COLORS.items()}
    sub = f"{len(boxes)} objects · " + ", ".join(f"{v} {k}" for k, v in n.items() if v) + \
          ("  · LOOPING" if looping(o) else "")
    return cb.draw_panel(frame, boxes, title, sub)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=list(SETS), default="gold_v2")
    ap.add_argument("--per-page", type=int, default=6)
    args = ap.parse_args()
    frames_csv, qpath, gpath = SETS[args.set]
    q, g = load(qpath), load(gpath)
    if frames_csv is not None:
        frames = pd.read_csv(frames_csv, dtype={"frame_id": str})[["unit_id", "frame_path"]]
    else:  # fullrun500: frames from the full run's manifest
        m = pd.read_csv(HERE.parent / "gemini_full_run" / "out" / "private" / "manifest.csv")
        frames = m[m.unit_id.isin(set(q))][["unit_id", "frame_path"]]
    out = HERE / "out" / "private" / f"overlays_{args.set}"
    for d in ("qwen", "side_by_side", "pages"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows, pairs = [], []
    for n, r in enumerate(frames.itertuples(), 1):
        frame = Image.open(r.frame_path).convert("RGB")
        if frame.size != (cb.FW, cb.FH):
            frame = frame.resize((cb.FW, cb.FH))
        qp = panel(frame, q.get(r.unit_id), f"#{n} Qwen3.8-27B (local, FP8)")
        gp = panel(frame, g.get(r.unit_id), f"#{n} Gemini 2.5 Flash (Vertex)")
        safe = r.unit_id.replace("/", "_")
        qp.save(out / "qwen" / f"{n:04d}_{safe}.jpg", quality=88)
        pair = Image.new("RGB", (2 * cb.FW + 12, cb.FH + 48), cb.WHITE)
        pair.paste(qp, (0, 0)); pair.paste(gp, (cb.FW + 12, 0))
        pair.save(out / "side_by_side" / f"{n:04d}_{safe}.jpg", quality=88)
        pairs.append(pair)
        qo, go = objects(q.get(r.unit_id)), objects(g.get(r.unit_id))
        rows.append({"n": n, "unit_id": r.unit_id, "qwen_objects": len(qo) if qo is not None else None,
                     "gemini_objects": len(go) if go is not None else None,
                     "qwen_looping": bool(qo and looping(qo)), "gemini_looping": bool(go and looping(go)),
                     "qwen_labels": ";".join(sorted({x["label"].strip().lower() for x in qo or []})),
                     "gemini_labels": ";".join(sorted({x["label"].strip().lower() for x in go or []}))})
        if len(pairs) == args.per_page or n == len(frames):
            cols, gap = 3, 16
            w, h = pairs[0].size
            rws = -(-len(pairs) // cols)
            page = Image.new("RGB", (cols * (w + gap) + gap, rws * (h + gap) + gap), (236, 235, 230))
            for i, im in enumerate(pairs):
                page.paste(im, (gap + (i % cols) * (w + gap), gap + (i // cols) * (h + gap)))
            page.save(out / "pages" / f"page_{(n - 1) // args.per_page + 1:03d}.jpg", quality=82)
            pairs = []
        if n % 100 == 0:
            print(f"  {n}/{len(frames)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    print(f"{len(rows)} frames -> {out} (qwen/, side_by_side/, pages/, index.csv)")


if __name__ == "__main__":
    main()
