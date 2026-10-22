#!/usr/bin/env python3
"""Score the Gemini gold-crop pilot (outputs of run_gold_pilot.py).

Writes results/{summary,per_category,by_size,model_differences,retest}.csv and
prints the decision table. Detection (whole frame) counts a gold crop as found
when Gemini returns a box with the same or a near-synonym label at IoU >= 0.5. Paired model differences use a category bootstrap
(categories resampled, crops within them kept), 2,000 draws.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from run_gold_pilot import GOLD_ZIP, OUT_DIR, load_vocab
from vocab_groups import related

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
N_BOOT = 2000
# On-demand $ per 1M tokens (input, output incl. thinking); batch is half.
PRICES = {"gemini-2.5-flash-lite": (0.10, 0.40), "gemini-3.1-flash-lite": (0.25, 1.50),
          "gemini-2.5-flash": (0.30, 2.50)}


TASK_COLS = {"name": ["strict", "lenient", "domain_ok"],
             "verify": ["hit", "fa_same", "fa_other", "all_correct"],
             "detect": ["found", "label_found", "box_found", "best_iou"]}


def read_runs():
    runs = {}
    for f in sorted(OUT_DIR.glob("*__*__p*.jsonl")) + sorted((OUT_DIR / "private").glob("*__*__p*.jsonl")):
        model, task, p = f.stem.split("__")
        if "_" in p or task not in TASK_COLS:  # sanity / subset runs and detect_open are not scored here
            continue
        rows = [json.loads(l) for l in open(f)]
        df = pd.DataFrame(rows)
        if task != "detect":
            df = df.rename(columns={"unit_id": "crop_id"})
        df = df.drop_duplicates("unit_id" if task == "detect" else "crop_id", keep="last")
        n_err = int(df.error.notna().sum())
        df = df[df.error.isna()]
        df["parsed"] = df.raw.map(json.loads)
        runs[(model, task, int(p[1:]))] = (df, n_err)
    return runs


def crop_sizes(gold):
    z = zipfile.ZipFile(GOLD_ZIP)
    side = {}
    for n in z.namelist():
        if n.endswith(".jpg"):
            w, h = Image.open(io.BytesIO(z.read(n))).size
            side[n.split("/")[-1][:-4]] = min(w, h)
    return gold.crop_id.map(side)


def score_name(df, gold, domains, labels):
    d = gold.merge(df[["crop_id", "parsed", "in_tok", "out_tok", "think_tok"]], on="crop_id")
    d["pred"] = d.parsed.map(lambda p: str(p.get("cdi_word", "none")).strip().lower())
    d["off_list"] = (~d.pred.isin(set(labels) | {"none"})).astype(float)  # schema no longer enforces the list
    d["open_label"] = d.parsed.map(lambda p: str(p.get("open_label", "")).lower())
    d["strict"] = (d.pred == d.category).astype(float)
    d["lenient"] = [float(related(p, c)) for p, c in zip(d.pred, d.category)]
    d["domain_ok"] = (d.pred.map(domains) == d.domain).astype(float)
    d["none"] = (d.pred == "none").astype(float)
    d["open_has_gold"] = [float(c in o) for c, o in zip(d.category, d.open_label)]
    return d


IOU_MIN = 0.5


def iou(a, b):
    """Boxes as (xmin, ymin, xmax, ymax)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gemini_boxes(parsed, w, h):
    """box_2d is [ymin, xmin, ymax, xmax] on 0-1000 of the image sent (same aspect as the frame)."""
    out = []
    for o in parsed.get("objects", []):
        b = o.get("box_2d", [])
        if len(b) != 4:
            continue
        y0, x0, y1, x1 = (min(1000, max(0, v)) for v in b)
        if x1 > x0 and y1 > y0:
            out.append((o.get("label", ""), (x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000)))
    return out


def score_detect(df, gf, vocab, frame_w=512, frame_h=910):
    pred = dict(zip(df.unit_id, df.parsed))
    d = gf[gf.unit_id.isin(pred)].copy()
    rows = []
    for r in d.itertuples():
        boxes = gemini_boxes(pred[r.unit_id], frame_w, frame_h)
        gold_box = (r.xmin, r.ymin, r.xmax, r.ymax)
        lab = [iou(b, gold_box) for l, b in boxes if related(l, r.category)]
        anyb = [iou(b, gold_box) for _, b in boxes]
        rows.append({"off_list": float(any(l not in vocab for l, _ in boxes)), "label_found": float(bool(lab)), "best_iou": max(lab) if lab else 0.0,
                     "found": float(any(v >= IOU_MIN for v in lab)),
                     "box_found": float(any(v >= IOU_MIN for v in anyb)), "n_objects": len(boxes)})
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    tok = df.set_index("unit_id")[["in_tok", "out_tok", "think_tok"]]
    return d.join(tok, on="unit_id")


def score_verify(df, gold):
    d = gold.merge(df[["crop_id", "parsed", "in_tok", "out_tok", "think_tok"]], on="crop_id")

    def lookup(p, w):
        for a in p.get("answers", []):
            if str(a.get("word", "")).strip().lower() == w:
                return float(bool(a.get("is_object")))
        return np.nan
    d["hit"] = [lookup(p, c) for p, c in zip(d.parsed, d.category)]
    d["fa_same"] = [lookup(p, w) for p, w in zip(d.parsed, d.foil_same)]
    d["fa_other"] = [lookup(p, w) for p, w in zip(d.parsed, d.foil_other)]
    d["all_correct"] = ((d.hit == 1) & (d.fa_same == 0) & (d.fa_other == 0)).astype(float)
    return d


def cat_boot(d, col, rng):
    """Mean of col with categories resampled (equal weight per crop)."""
    g = d.groupby("category")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = rng.integers(0, len(g), (N_BOOT, len(g)))
    return s[idx].sum(1) / c[idx].sum(1)


def main():
    RES.mkdir(exist_ok=True)
    labels, domains = load_vocab()
    gold = pd.read_csv(OUT_DIR / "gold_manifest.csv")
    gold["short_side"] = crop_sizes(gold)
    gf_path = OUT_DIR / "private" / "gold_frames.csv"
    gf = pd.read_csv(gf_path, dtype={"frame_id": str}) if gf_path.exists() else None
    if gf is not None:
        gf["unit_id"] = gf.video + "/" + gf.frame_id
        gf["domain"] = gf.category.map(domains)
        gf["short_side"] = np.minimum(gf.xmax - gf.xmin, gf.ymax - gf.ymin)
    runs = read_runs()
    if not runs:
        raise SystemExit("No runs in out/ yet.")

    scored, summary = {}, []
    rng = np.random.default_rng(0)
    for (model, task, p), (df, n_err) in runs.items():
        if task == "detect":
            d = score_detect(df, gf, set(labels))
        else:
            d = score_name(df, gold, domains, labels) if task == "name" else score_verify(df, gold)
        scored[(model, task, p)] = d
        cols = TASK_COLS[task] + (["none", "open_has_gold"] if task == "name" else []) + (["off_list"] if task != "verify" else [])
        row = {"model": model, "task": task, "pass": p, "n": len(d), "failed": n_err,
               "in_tok_mean": d.in_tok.mean(), "out_tok_mean": d.out_tok.mean(),
               "think_tok_mean": d.think_tok.mean()}
        if task == "detect":
            row["objects_per_frame"] = d.drop_duplicates("unit_id").n_objects.mean()
        for c in cols:
            b = cat_boot(d.dropna(subset=[c]), c, rng)
            row[c] = d[c].mean()
            row[f"{c}_lo"], row[f"{c}_hi"] = np.percentile(b, [2.5, 97.5])
        if model in PRICES:
            pi, po = PRICES[model]
            row["usd_per_1k_calls_ondemand"] = (d.in_tok.mean() * pi + (d.out_tok.mean() + d.think_tok.mean()) * po) / 1e3
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(RES / "summary.csv", index=False)

    # Per category and by crop size (pass 1).
    per_cat, by_size = [], []
    for (model, task, p), d in scored.items():
        if p != 1:
            continue
        cols = TASK_COLS[task]
        g = d.groupby(["category", "domain"])[cols].mean().reset_index()
        g.insert(0, "model", model); g.insert(1, "task", task)
        per_cat.append(g)
        d = d.assign(size_bin=pd.cut(d.short_side, [0, 32, 64, 128, 10_000], labels=["<32", "32-63", "64-127", "128+"]))
        s = d.groupby("size_bin", observed=True)[cols].mean().reset_index()
        s.insert(0, "model", model); s.insert(1, "task", task)
        s["n"] = d.groupby("size_bin", observed=True).size().values
        by_size.append(s)
    pd.concat(per_cat).to_csv(RES / "per_category.csv", index=False)
    pd.concat(by_size).to_csv(RES / "by_size.csv", index=False)

    # Paired model differences, pass 1, crops both models answered.
    diffs = []
    models = sorted({m for m, _, _ in scored})
    for task, cols in TASK_COLS.items():
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                if (a, task, 1) not in scored or (b, task, 1) not in scored:
                    continue
                m = scored[(a, task, 1)].merge(scored[(b, task, 1)][["crop_id"] + cols], on="crop_id", suffixes=("_a", "_b"))
                for c in cols:
                    mm = m.dropna(subset=[f"{c}_a", f"{c}_b"]).assign(diff=lambda x: x[f"{c}_b"] - x[f"{c}_a"])
                    bt = cat_boot(mm, "diff", rng)
                    diffs.append({"task": task, "measure": c, "model_a": a, "model_b": b, "n": len(mm),
                                  "a": mm[f"{c}_a"].mean(), "b": mm[f"{c}_b"].mean(), "b_minus_a": mm["diff"].mean(),
                                  "lo": np.percentile(bt, 2.5), "hi": np.percentile(bt, 97.5),
                                  "p_two_sided": min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean()))})
    pd.DataFrame(diffs).to_csv(RES / "model_differences.csv", index=False)

    # Retest agreement (pass 1 vs 2).
    retest = []
    for (model, task, p), d2 in scored.items():
        if task == "detect" or p != 2 or (model, task, 1) not in scored:
            continue
        d1 = scored[(model, task, 1)]
        key = "pred" if task == "name" else "all_correct"
        m = d1[["crop_id", key]].merge(d2[["crop_id", key]], on="crop_id")
        retest.append({"model": model, "task": task, "n": len(m), "agreement": (m[f"{key}_x"] == m[f"{key}_y"]).mean()})
    pd.DataFrame(retest).to_csv(RES / "retest.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(pd.DataFrame(diffs).round(3).to_string(index=False))
    print(pd.DataFrame(retest).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
