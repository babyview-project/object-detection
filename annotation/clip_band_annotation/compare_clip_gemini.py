#!/usr/bin/env python3
"""CLIP filter vs Gemini verify on the CLIP-band sample (run_gemini_verify.py outputs).

Gemini "accepts" an item when it says YOLOE's label is the object (verify hit). The
two foil words give its false-alarm rate in every band.

Tables (results/, aggregate only: no item ids, subjects or videos)
  by_band.csv          per CLIP band: Gemini acceptance (crop / context) with category-
                       bootstrap 95% CIs, foil false alarms, crop->context changes
  by_threshold.csv     for CLIP cut-offs 0.26 / 0.27 / 0.28: CLIP x Gemini agreement,
                       weighted to all 13.2 M scored 2025.2 crops (band_population.csv)
  by_category.csv      per category: Gemini acceptance below 0.26 vs at/above 0.28, and
                       Gemini (context) acceptance of crops CLIP drops at 0.27, weighted
  category_clip_population.csv  per category over all scored crops: median CLIP and
                       pass rate at each cut-off (from crops_index.csv.gz; cached)
  vs_human.csv         only if out/private/human_labels.csv exists (item_id,
                       human_crop, human_context as 0/1): population-weighted precision
                       and recall of CLIP at each cut-off and of Gemini, against the
                       human crop and human context judgements
Per-item merge: out/private/gemini/items_scored.csv (private).

Weights: each item stands for n_population / n_sampled crops of its category x band
cell, so population estimates are not skewed by sampling 8 per cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
GEM = OUT / "gemini"
RES = HERE / "results"
MODEL = "gemini-2.5-flash-lite"
CONDS = ("crop", "context")
THRESHOLDS = (0.26, 0.27, 0.28)
BAND_ORDER = ["lt0.22", "0.22-0.24", "0.24-0.26", "0.26-0.27", "0.27-0.28", "0.28-0.30", "ge0.30"]
N_BOOT = 2000


def lookup(raw, word):
    try:
        answers = json.loads(raw).get("answers", [])
    except (TypeError, ValueError):
        return np.nan
    for a in answers:
        if str(a.get("word", "")).strip().lower() == word:
            return float(bool(a.get("is_object")))
    return np.nan


def load():
    s = pd.read_csv(OUT / "items.csv")
    man = pd.read_csv(GEM / "verify_manifest.csv")[["item_id", "foil_same", "foil_other"]]
    s = s.merge(man, on="item_id")
    for c in CONDS:
        r = pd.read_parquet(GEM / f"{MODEL}__verify_{c}.parquet")
        r = r[r.error.isna()].set_index("unit_id").raw
        raw = s.item_id.map(r)
        s[f"g_{c}"] = [lookup(x, w) for x, w in zip(raw, s.class_name)]
        s[f"fa_same_{c}"] = [lookup(x, w) for x, w in zip(raw, s.foil_same)]
        s[f"fa_other_{c}"] = [lookup(x, w) for x, w in zip(raw, s.foil_other)]
    pop = pd.read_csv(OUT / "band_population.csv")
    s = s.merge(pop, on=["class_name", "band"], how="left")
    s["w"] = s.n_population / s.groupby(["class_name", "band"]).item_id.transform("count")
    return s


def boot_ci(d, col, rng, w=None):
    """Category-bootstrap 95% CI of the (weighted) mean of col."""
    d = d.dropna(subset=[col])
    ww = d[w] if w else pd.Series(1.0, index=d.index)
    g = pd.DataFrame({"s": d[col] * ww, "c": ww, "k": d.class_name}).groupby("k")[["s", "c"]].sum()
    idx = rng.integers(0, len(g), (N_BOOT, len(g)))
    b = g.s.values[idx].sum(1) / g.c.values[idx].sum(1)
    return np.percentile(b, [2.5, 97.5])


def wmean(x, w):
    m = x.notna()
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan


def by_band(s, rng):
    rows = []
    for b in BAND_ORDER:
        d = s[s.band == b]
        r = {"band": b, "n": len(d), "n_categories": d.class_name.nunique(),
             "population": int(d.groupby("class_name").n_population.first().sum())}
        for c in CONDS:
            r[f"accept_{c}"] = d[f"g_{c}"].mean()
            r[f"accept_{c}_lo"], r[f"accept_{c}_hi"] = boot_ci(d, f"g_{c}", rng)
            r[f"fa_same_{c}"] = d[f"fa_same_{c}"].mean()
            r[f"fa_other_{c}"] = d[f"fa_other_{c}"].mean()
        both = d.dropna(subset=["g_crop", "g_context"])
        r["rescued_by_context"] = ((both.g_crop == 0) & (both.g_context == 1)).mean()
        r["dropped_by_context"] = ((both.g_crop == 1) & (both.g_context == 0)).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def kappa(a, b, w):
    m = a.notna() & b.notna()
    a, b, w = a[m], b[m], w[m]
    po = wmean((a == b).astype(float), w)
    pa, pb = wmean(a, w), wmean(b, w)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


def by_threshold(s):
    rows = []
    for t in THRESHOLDS:
        clip = (s.clip_sim >= t).astype(float)
        for c in CONDS:
            g = s[f"g_{c}"]
            m = g.notna()
            w = s.w[m]
            cp, gg = clip[m], g[m]
            tot = w.sum()
            rows.append({
                "threshold": t, "gemini": c,
                "clip_pass_pop": float((w * cp).sum() / tot),
                "gemini_accept_pop": float((w * gg).sum() / tot),
                "both_accept": float((w * cp * gg).sum() / tot),
                "clip_only": float((w * cp * (1 - gg)).sum() / tot),
                "gemini_only": float((w * (1 - cp) * gg).sum() / tot),
                "both_reject": float((w * (1 - cp) * (1 - gg)).sum() / tot),
                "gemini_accepts_of_clip_rejected": wmean(gg[cp == 0], w[cp == 0]),
                "gemini_rejects_of_clip_passed": 1 - wmean(gg[cp == 1], w[cp == 1]),
                "agreement": wmean((cp == gg).astype(float), w),
                "kappa": kappa(cp, gg, w),
            })
    return pd.DataFrame(rows)


def category_population():
    """Median CLIP and pass rates per category over the whole population (cached)."""
    p = RES / "category_clip_population.csv"
    if p.exists():
        return pd.read_csv(p, index_col="class_name")
    from sample_clip_bands import CATS_FILE, INDEX
    cats = [c.strip() for c in CATS_FILE.read_text().splitlines() if c.strip()]
    d = pd.read_csv(INDEX, usecols=["class_name", "clip_sim"], dtype={"class_name": "category"})
    d = d[d.class_name.isin(cats)]
    g = d.groupby("class_name", observed=True).clip_sim
    t = pd.DataFrame({"n_population": g.size(), "median_clip_pop": g.median()})
    for th in THRESHOLDS:
        t[f"pass_{th}_pop"] = g.apply(lambda x: (x >= th).mean())
    t.to_csv(p)
    return t


def by_category(s):
    pop = category_population()
    rej = s[(s.clip_sim < 0.27) & s.g_context.notna()]
    acc_rej = rej.groupby("class_name").apply(lambda d: wmean(d.g_context, d.w), include_groups=False)
    lo = s[s.clip_sim < 0.26].groupby("class_name")[["g_crop", "g_context"]].mean().add_suffix("_below_0.26")
    hi = s[s.clip_sim >= 0.28].groupby("class_name")[["g_crop", "g_context"]].mean().add_suffix("_from_0.28")
    n = s.groupby("class_name").agg(n=("item_id", "count"))
    n = n.join(pop).join(acc_rej.rename("g_context_of_clip_rejected_0.27"))
    return n.join(lo).join(hi).sort_values("g_context_below_0.26", ascending=False)


def vs_human(s):
    p = OUT / "human_labels.csv"
    if not p.exists():
        return None
    h = pd.read_csv(p)[["item_id", "human_crop", "human_context"]]
    d = s.merge(h, on="item_id")
    rows = []
    preds = {f"clip>={t}": (d.clip_sim >= t).astype(float) for t in THRESHOLDS}
    preds |= {f"gemini_{c}": d[f"g_{c}"] for c in CONDS}
    for truth in ("human_crop", "human_context"):
        y = d[truth]
        for name, p_ in preds.items():
            m = y.notna() & p_.notna()
            yy, pp, w = y[m], p_[m], d.w[m]
            tp = (w * yy * pp).sum()
            rows.append({"truth": truth, "filter": name, "n": int(m.sum()),
                         "precision": tp / (w * pp).sum(), "recall": tp / (w * yy).sum(),
                         "accuracy": wmean((yy == pp).astype(float), w),
                         "human_valid_pop": wmean(yy, w)})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(0)
    RES.mkdir(exist_ok=True)
    s = load()
    print(f"{len(s)} items; Gemini answered crop {s.g_crop.notna().sum()}, context {s.g_context.notna().sum()}")
    s.to_csv(GEM / "items_scored.csv", index=False)
    pd.set_option("display.width", 200)
    tables = {"by_band": by_band(s, rng), "by_threshold": by_threshold(s), "by_category": by_category(s)}
    h = vs_human(s)
    if h is not None:
        tables["vs_human"] = h
    for k, t in tables.items():
        t.to_csv(RES / f"{k}.csv", index=(k == "by_category"))
        print(f"\n== {k}\n" + (t.head(15) if k == "by_category" else t).round(3).to_string())


if __name__ == "__main__":
    main()
