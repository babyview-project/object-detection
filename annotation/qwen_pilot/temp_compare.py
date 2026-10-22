#!/usr/bin/env python3
"""Qwen sampling settings on the 500 full-run pilot frames: looping vs objects lost.

Settings (detect_open_format, setup B): temperature 0 (pilot), Qwen's recommended
(temperature 0.7 + presence penalty 1.5), and the four isolating runs: t0.5, t1, t0 + penalty,
t0.5 + penalty. Per setting: marching loops (>= 8 consecutive same-label, same-size boxes),
near-duplicate loops, frames at the 30-box cap, objects and distinct labels per frame
(all frames, and frames that did NOT loop at temperature 0, so lost loop boxes do not count
as lost objects), format mix, output tokens, and how many of temperature 0's boxes
(IoU >= 0.5) come back, and with the same label.

Output: results/temperature_compare.csv
"""
from __future__ import annotations

import collections
from pathlib import Path

import pandas as pd

from compare_qwen_gemini import iou, load, looping, objects

HERE = Path(__file__).resolve().parent
P = HERE.parent / "gemini_pilot" / "out" / "private"
F = "qwen3.8-27b-fp8__detect_open_format__p1_fullrun500{}.jsonl"
SETTINGS = {"t0 (pilot)": "", "t0.5": "_t0.5_pp0", "t1.0": "_t1_pp0", "t0 + penalty 1.5": "_t0_pp1.5",
            "t0.5 + penalty 1.5": "_t0.5_pp1.5", "t0.7 + penalty 1.5 (Qwen rec.)": "_qwen"}


def marching(o):
    best = run = 1
    for a, b in zip(o, o[1:]):
        A, B = a["box_2d"], b["box_2d"]
        hA, wA, hB, wB = A[2] - A[0], A[3] - A[1], B[2] - B[0], B[3] - B[1]
        same = a["label"].strip().lower() == b["label"].strip().lower()
        size = abs(hA - hB) <= max(8, .1 * hA) and abs(wA - wB) <= max(8, .1 * wA)
        run = run + 1 if same and size else 1
        best = max(best, run)
    return best if o else 0


def main():
    base = load(P / F.format(""))
    base_obj = {u: objects(r) or [] for u, r in base.items()}
    clean = {u for u, o in base_obj.items() if marching(o) < 8 and not looping(o) and len(o) < 30}
    rows = []
    for name, suf in SETTINGS.items():
        recs = load(P / F.format(suf))
        if not recs:
            rows.append({"setting": name, "frames": 0}); continue
        obj = {u: objects(r) for u, r in recs.items()}
        ok = {u: o for u, o in obj.items() if o is not None}
        fm = collections.Counter(x.get("format") for o in ok.values() for x in o)
        tot = sum(fm.values()) or 1
        found = same = total = 0
        for u, o in ok.items():
            for b in base_obj.get(u, []):
                if u not in clean:
                    continue
                total += 1
                best = max(((iou(b["box_2d"], x["box_2d"]), x) for x in o), key=lambda t: t[0], default=(0, None))
                if best[0] >= 0.5:
                    found += 1
                    same += best[1]["label"].strip().lower() == b["label"].strip().lower()
        cl = [o for u, o in ok.items() if u in clean]
        rows.append({
            "setting": name, "frames": len(recs), "parse_failures": len(recs) - len(ok),
            "marching_loops": sum(marching(o) >= 8 for o in ok.values()) / len(ok),
            "near_dup_loops": sum(looping(o) for o in ok.values()) / len(ok),
            "at_30_cap": sum(len(o) >= 30 for o in ok.values()) / len(ok),
            "objects_per_frame": sum(map(len, ok.values())) / len(ok),
            "objects_per_frame_clean": sum(map(len, cl)) / max(1, len(cl)),
            "distinct_labels_clean": sum(len({x["label"].strip().lower() for x in o}) for o in cl) / max(1, len(cl)),
            "t0_boxes_found_clean": found / max(1, total), "same_label": same / max(1, found),
            "fmt_toy": fm["toy"] / tot, "fmt_picture": (fm["photo"] + fm["drawing"]) / tot, "fmt_screen": fm["screen"] / tot,
            "out_tokens": sum(r.get("out_tok", 0) for r in recs.values()) / len(recs)})
    t = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    t.to_csv(HERE / "results" / "temperature_compare.csv", index=False)
    pd.set_option("display.width", 260)
    print(f"'clean' = {len(clean)} of {len(base_obj)} frames with no loop and under the cap at temperature 0")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
