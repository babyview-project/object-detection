#!/usr/bin/env python3
"""Qwen3.8-27B vs Gemini 2.5 Flash on the same frames (open label + format, setup B).

Sets (whichever have Qwen output):
  fullrun500  Gemini = the full run's 500-frame batch pilot (temperature 0)
  gold_v2     Gemini = gemini-2.5-flash detect_open_format cap30 on the 1,000 gold v2 frames
  depiction   Gemini = gemini-2.5-flash detect_combo (has a format field) on the 80 depiction
              frames; also format accuracy against Mira / Dora's format tags
Per model and set: parse failures, replies cut at the token cap, objects per frame, frames at
the cap, looping frames (one label >= 10 times with >= 3 near-duplicate boxes), distinct
labels per frame, format mix, output tokens; and for Qwen how many of Gemini's boxes it also
finds (IoU >= 0.5) and with the same label. CDI-list detection on the 5,919 gold frames is
scored by ../gemini_pilot/score_gold_pilot.py with the Gemini runs.

Output: results/qwen_vs_gemini.csv, results/qwen_format_vs_human.csv (rates only).

  python compare_qwen_gemini.py [--qwen qwen3.8-27b-fp8]
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PRIV = HERE.parent / "gemini_pilot" / "out" / "private"
FULLRUN = HERE.parent / "gemini_full_run" / "out" / "private" / "pilot_results" / "results_pilot.jsonl"
RES = HERE / "results"
W, H = 512, 910


def load(path):
    recs = {}
    if Path(path).exists():
        for line in open(path):
            r = json.loads(line)
            recs[r["unit_id"]] = r
    return recs


def objects(rec):
    if not rec or rec.get("error") is not None:
        return None
    try:
        return [o for o in json.loads(rec["raw"]).get("objects", []) if len(o.get("box_2d", [])) == 4]
    except (ValueError, TypeError):
        return None


def iou(a, b):  # box_2d = [ymin, xmin, ymax, xmax]
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def looping(objs):
    by = collections.defaultdict(list)
    for o in objs:
        by[o["label"].strip().lower()].append(o["box_2d"])
    return any(len(b) >= 10 and sum(iou(x, y) > 0.7 for x, y in itertools.combinations(b, 2)) >= 3
               for b in by.values())


def describe(recs, units, cap):
    rs = [recs.get(u) for u in units]
    parsed = [objects(r) for r in rs]
    ok = [o for o in parsed if o is not None]
    fm = collections.Counter(x.get("format", "") for o in ok for x in o)
    tot = sum(fm.values()) or 1
    return {"frames": len(units), "answered": sum(r is not None for r in rs),
            "parse_failures": sum(r is not None and p is None for r, p in zip(rs, parsed)),
            "cut_at_token_cap": sum(1 for r in rs if r and str(r.get("finish", "")).upper() in ("LENGTH", "MAX_TOKENS")),
            "objects_per_frame": sum(map(len, ok)) / max(1, len(ok)),
            "at_object_cap": sum(len(o) >= cap for o in ok) / max(1, len(ok)),
            "looping_frames": sum(looping(o) for o in ok) / max(1, len(ok)),
            "distinct_labels_per_frame": sum(len({x["label"].strip().lower() for x in o}) for o in ok) / max(1, len(ok)),
            "fmt_real": fm["real"] / tot, "fmt_toy": fm["toy"] / tot, "fmt_drawing": fm["drawing"] / tot,
            "fmt_photo": fm["photo"] / tot, "fmt_screen": fm["screen"] / tot,
            "out_tokens_mean": sum((r or {}).get("out_tok", 0) or 0 for r in rs) / max(1, len(rs)),
            "sec_per_frame": sum((r or {}).get("latency", 0) or 0 for r in rs) / max(1, len(rs))}


def agreement(q, g, units):
    found = same = total = 0
    for u in units:
        qo, go = objects(q.get(u)), objects(g.get(u))
        if qo is None or go is None:
            continue
        for b in go:
            total += 1
            best = max(((iou(b["box_2d"], x["box_2d"]), x) for x in qo), key=lambda t: t[0], default=(0, None))
            if best[0] >= 0.5:
                found += 1
                same += best[1]["label"].strip().lower() == b["label"].strip().lower()
    return {"gemini_boxes_found_by_qwen": found / max(1, total), "found_with_same_label": same / max(1, found)}


def format_vs_human(q_recs, name):
    """Format on the best-overlapping box (IoU >= 0.3) for each human-tagged box."""
    bx = pd.read_csv(PRIV / "depiction_boxes.csv")
    bx = bx[bx.format.notna() & (bx.format != "")]
    rows = []
    for r in bx.itertuples():
        o = objects(q_recs.get(r.unit_id))
        if o is None:
            continue
        hb = [r.ymin / H * 1000, r.xmin / W * 1000, r.ymax / H * 1000, r.xmax / W * 1000]
        best = max(((iou(hb, x["box_2d"]), x) for x in o), key=lambda t: t[0], default=(0, None))
        hf = r.format.removesuffix("_format").replace("video", "screen")
        gf = best[1].get("format", "") if best[0] >= 0.3 else None
        rows.append({"model": name, "human_format": hf, "matched": gf is not None, "correct": gf == hf,
                     "real_vs_depiction_ok": gf is not None and ((gf == "real") == (hf == "real"))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwen", default="qwen3.8-27b-fp8")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    sets = {
        "fullrun500": (load(FULLRUN), "_fullrun500", 30),
        "gold_v2": (load(PRIV / "gemini-2.5-flash__detect_open_format__p1_gold_v2_frames_cap30.jsonl"), "_gold_v2_frames_cap30", 30),
        "depiction": (load(PRIV / "gemini-2.5-flash__detect_combo__p1_depiction_frames.jsonl"), "_depiction_frames", 30),
    }
    rows, fmt = [], []
    for s, (gem, tag, cap) in sets.items():
        for sampling in ("", "_qwen"):
            q = load(PRIV / f"{args.qwen}__detect_open_format__p1{tag}{sampling}.jsonl")
            if not q:
                continue
            units = sorted(set(q) & set(gem)) if gem else sorted(q)
            label = f"{args.qwen}{sampling or ' (temp 0)'}"
            rows.append({"set": s, "model": label, **describe(q, units, cap), **(agreement(q, gem, units) if gem else {})})
            if s == "depiction":
                fmt += format_vs_human(q, label)
            if sampling == "" and gem:
                rows.append({"set": s, "model": "gemini-2.5-flash (temp 0)", **describe(gem, units, cap)})
                if s == "depiction":
                    fmt += format_vs_human(gem, "gemini-2.5-flash (combo)")
    if not rows:
        raise SystemExit("no Qwen output yet")
    t = pd.DataFrame(rows)
    t.to_csv(RES / "qwen_vs_gemini.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(t.round(3).to_string(index=False))
    if fmt:
        f = pd.DataFrame(fmt)
        a = f[f.matched].groupby(["model", "human_format"]).agg(boxes=("correct", "size"), correct=("correct", "mean"),
                                                                real_vs_depiction=("real_vs_depiction_ok", "mean"))
        a.to_csv(RES / "qwen_format_vs_human.csv")
        print("\n== format vs human tags (depiction frames)\n" + a.round(2).to_string())


if __name__ == "__main__":
    main()
