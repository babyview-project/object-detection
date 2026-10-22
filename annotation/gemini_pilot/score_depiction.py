#!/usr/bin/env python3
"""Do Gemini's open-vocabulary labels carry visual-format information for free?

For each human-tagged object box in out/private/depiction_boxes.csv (format: real / toy /
photo / drawing / video), take Gemini's best-overlapping box (IoU >= MIN_IOU) from a
detect_open[_neutral] run on out/private/depiction_frames.csv and check its label for
format words (CUES below; a hand-made list, edit freely):
  any_cue     the label has a word for any non-real format
  right_cue   the label has a word for this box's format
For real-format boxes, any_cue is a false format word.

Prompts: detect_open_neutral says nothing about formats (any cue is unprompted);
detect_open asks to name what toys / pictures / screens show (prompted).

Output: results/depiction_cues.csv (rates; no ids); per-box table with labels in
out/private/depiction_scored.csv.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"
MIN_IOU = 0.3
W, H = 512, 910
CUES = {
    "toy": ["toy", "stuffed", "plush", "figurine", "figure", "teddy", "puppet", "plastic", "rubber", "model",
            "squishy", "squeaky", "play", "doll", "lego", "wooden"],
    "photo": ["photo", "photograph", "picture", "poster", "print", "framed", "frame", "calendar", "magazine",
              "card", "image", "postcard", "portrait"],
    "drawing": ["drawing", "cartoon", "illustration", "painting", "sticker", "decal", "picture", "book",
                "sign", "logo", "graphic", "artwork", "art", "poster", "print", "pattern", "label", "character",
                "puzzle", "wall art", "page"],
    "video": ["screen", "tv", "television", "video", "monitor", "phone", "tablet", "ipad", "laptop",
              "cartoon", "show", "game", "display", "computer"],
}
ALL = sorted({w for v in CUES.values() for w in v})


def has(label, words):
    return any(re.search(rf"\b{re.escape(w)}s?\b", label) for w in words)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def gem_boxes(raw):
    out = []
    for o in json.loads(raw).get("objects", []):
        b = o.get("box_2d", [])
        if len(b) == 4:
            y0, x0, y1, x1 = (min(1000, max(0, v)) for v in b)
            if x1 > x0 and y1 > y0:
                out.append((str(o.get("label", "")).strip().lower(),
                            (x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000)))
    return out


def main():
    bx = pd.read_csv(PRIV / "depiction_boxes.csv")
    bx = bx[bx.format.notna() & (bx.format != "")]
    bx["fmt"] = bx.format.str.removesuffix("_format")
    rows, rates = [], []
    for f in sorted(PRIV.glob("*__detect_open*__p1_depiction_frames.jsonl")):
        model, task = f.stem.split("__")[:2]
        recs = {}
        for line in open(f):
            r = json.loads(line)
            recs[r["unit_id"]] = r
        all_labels = []
        for r in bx.itertuples():
            rec = recs.get(r.unit_id)
            gb = gem_boxes(rec["raw"]) if rec and rec["error"] is None else []
            hb = (r.xmin, r.ymin, r.xmax, r.ymax)
            best = max(((iou(b, hb), l) for l, b in gb), default=(0.0, ""))
            lab = best[1] if best[0] >= MIN_IOU else ""
            rows.append({"model": model, "prompt": task.replace("detect_open_neutral", "neutral").replace("detect_open", "asks for formats"),
                         "unit_id": r.unit_id, "human_label": r.label, "format": r.fmt, "gemini_label": lab,
                         "iou": round(best[0], 2), "matched": bool(lab),
                         "any_cue": has(lab, ALL) if lab else False,
                         "right_cue": has(lab, CUES.get(r.fmt, [])) if lab and r.fmt != "real" else False})
    d = pd.DataFrame(rows)
    d.to_csv(PRIV / "depiction_scored.csv", index=False)
    g = d.groupby(["model", "prompt", "format"])
    t = pd.DataFrame({"boxes": g.size(), "matched": g.matched.mean(),
                      "any_cue_of_matched": g.apply(lambda x: x[x.matched].any_cue.mean(), include_groups=False),
                      "right_cue_of_matched": g.apply(lambda x: x[x.matched].right_cue.mean(), include_groups=False)})
    t.loc[t.index.get_level_values("format") == "real", "right_cue_of_matched"] = float("nan")
    (HERE / "results").mkdir(exist_ok=True)
    t.round(3).to_csv(HERE / "results" / "depiction_cues.csv")
    pd.set_option("display.width", 200)
    print(t.round(2).to_string())
    for (m, p), x in d[d.matched & (d.format != "real")].groupby(["model", "prompt"]):
        print(f"\n{m} · {p}: human label (format) -> Gemini label")
        for fmt, y in x.groupby("format"):
            print(f"  {fmt:8s}", "; ".join(f"{a} -> {b}" for a, b in zip(y.human_label, y.gemini_label)))


if __name__ == "__main__":
    main()
