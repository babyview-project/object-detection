#!/usr/bin/env python3
"""Score the combined prompt (detect_combo: open label + CDI word + format) against the
single-task runs on the same frames.

1. CDI (72 gold frames, montage72): gold object found (cdi_word same / near-synonym and
   IoU >= 0.5 with the gold box) and word anywhere in the frame, vs the CDI-list detect run.
2. Anchoring (80 depiction frames): share of open labels that are exactly a vocabulary
   word, combo vs the neutral open run; distinct labels; share of labels == cdi_word.
3. Format (depiction frames): Gemini format vs the human tag on the best-overlapping box
   (IoU >= 0.3; human video_format -> screen); accuracy per human format, confusion,
   and how often toy boxes are labelled just "toy" (identity lost).

Output: results/combo_*.csv (rates only).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_gold_pilot as g
from score_depiction import gem_boxes, iou
from vocab_groups import related

HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"
RES = HERE / "results"
MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
W, H = 512, 910


def load(path):
    recs = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            recs[r["unit_id"]] = r
    return recs


def objects(rec):
    """combo objects with pixel boxes; [] if the call failed."""
    if not rec or rec["error"] is not None:
        return None
    out = []
    for o in json.loads(rec["raw"]).get("objects", []):
        b = o.get("box_2d", [])
        if len(b) != 4:
            continue
        y0, x0, y1, x1 = (min(1000, max(0, v)) for v in b)
        if x1 > x0 and y1 > y0:
            out.append({"label": str(o.get("label", "")).strip().lower(),
                        "cdi": str(o.get("cdi_word", "")).strip().lower(), "format": o.get("format", ""),
                        "box": (x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000)})
    return out


def main():
    RES.mkdir(exist_ok=True)
    vocab = set(g.load_vocab()[0])
    gold = pd.read_csv(PRIV / "gold_frames.csv", dtype={"frame_id": str})
    gold["unit_id"] = gold.video + "/" + gold.frame_id
    units = [l.strip() for l in open(PRIV / "montage72.txt") if l.strip()]
    gold = gold[gold.unit_id.isin(units)]
    dep = pd.read_csv(PRIV / "depiction_boxes.csv")
    dep = dep[dep.format.notna() & (dep.format != "")].copy()
    dep["fmt"] = dep.format.str.removesuffix("_format").replace({"video": "screen"})
    pd.set_option("display.width", 200)

    cdi_rows, anchor_rows, fmt_rows = [], [], []
    for m in MODELS:
        combo_g = load(PRIV / f"{m}__detect_combo__p1_montage72.jsonl")
        cdi_only = load(PRIV / f"{m}__detect__p1.jsonl")
        for name, recs, get in (("combo", combo_g, lambda o: [(x["cdi"], x["box"]) for x in o]),
                                ("CDI-list detect", cdi_only, None)):
            found = word = n = ok = none = 0
            for r in gold.itertuples():
                rec = recs.get(r.unit_id)
                if name == "combo":
                    o = objects(rec)
                    if o is None:
                        continue
                    pairs = get(o)
                    none += sum(c == "none" for c, _ in pairs)
                else:
                    if not rec or rec["error"] is not None:
                        continue
                    pairs = [(l.strip().lower(), b) for l, b in gem_boxes(rec["raw"])]
                gb = (r.xmin, r.ymin, r.xmax, r.ymax)
                hits = [iou(b, gb) for c, b in pairs if related(c, r.category)]
                found += any(v >= 0.5 for v in hits); word += bool(hits); n += len(pairs); ok += 1
            cdi_rows.append({"model": m, "run": name, "frames": ok, "gold_found_iou50": found / ok,
                             "gold_word_anywhere": word / ok, "objects_per_frame": n / ok,
                             "cdi_none_share": none / n if name == "combo" else float("nan")})

        combo_d = load(PRIV / f"{m}__detect_combo__p1_depiction_frames.jsonl")
        neutral = load(PRIV / f"{m}__detect_open_neutral__p1_depiction_frames.jsonl")
        cl = [x for rec in combo_d.values() for x in (objects(rec) or [])]
        nl = [l.strip().lower() for rec in neutral.values() if rec["error"] is None for l, _ in gem_boxes(rec["raw"])]
        anchor_rows += [
            {"model": m, "run": "combo", "labels": len(cl), "distinct": len({x["label"] for x in cl}),
             "label_is_vocab_word": sum(x["label"] in vocab for x in cl) / len(cl),
             "label_equals_cdi_word": sum(x["label"] == x["cdi"] for x in cl) / len(cl),
             "cdi_none": sum(x["cdi"] == "none" for x in cl) / len(cl)},
            {"model": m, "run": "neutral open", "labels": len(nl), "distinct": len(set(nl)),
             "label_is_vocab_word": sum(l in vocab for l in nl) / len(nl)}]

        for r in dep.itertuples():
            o = objects(combo_d.get(r.unit_id))
            if o is None:
                continue
            hb = (r.xmin, r.ymin, r.xmax, r.ymax)
            best = max(((iou(x["box"], hb), x) for x in o), key=lambda t: t[0], default=(0, None))
            x = best[1] if best[0] >= 0.3 else None
            fmt_rows.append({"model": m, "human_format": r.fmt, "human_label": r.label,
                             "matched": x is not None, "gemini_format": x["format"] if x else "",
                             "gemini_label": x["label"] if x else "", "gemini_cdi": x["cdi"] if x else ""})

    c = pd.DataFrame(cdi_rows); a = pd.DataFrame(anchor_rows); f = pd.DataFrame(fmt_rows)
    f.to_csv(PRIV / "combo_format_scored.csv", index=False)
    mf = f[f.matched]
    acc = mf.assign(correct=mf.human_format == mf.gemini_format,
                    nonreal_ok=(mf.human_format == "real") == (mf.gemini_format == "real"),
                    just_toy=mf.gemini_label.isin(["toy", "toys"])) \
            .groupby(["model", "human_format"]).agg(boxes=("correct", "size"), format_correct=("correct", "mean"),
                                                   real_vs_depiction_correct=("nonreal_ok", "mean"),
                                                   label_just_toy=("just_toy", "mean"))
    conf = pd.crosstab([mf.model, mf.human_format], mf.gemini_format)
    for k, t in (("combo_cdi", c), ("combo_anchoring", a), ("combo_format", acc), ("combo_format_confusion", conf)):
        t.round(3).to_csv(RES / f"{k}.csv", index=k in ("combo_format", "combo_format_confusion"))
        print(f"\n== {k}\n{t.round(2).to_string()}")
    print("\nmatched depicted boxes (human label/format -> Gemini label/cdi/format), gemini-2.5-flash:")
    for fmt, y in mf[(mf.model == "gemini-2.5-flash") & (mf.human_format != "real")].groupby("human_format"):
        print(f"  {fmt:8s}", "; ".join(f"{a} -> {b}/{c}/{d}" for a, b, c, d in
                                        zip(y.human_label, y.gemini_label, y.gemini_cdi, y.gemini_format)))


if __name__ == "__main__":
    main()
