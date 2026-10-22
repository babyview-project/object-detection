#!/usr/bin/env python3
"""Which human-labelled categories do the open-vocabulary models recover? (preliminary)

Human-labelled frames that already have open-vocabulary output from both models:
  montage72  72 gold frames, one rater-validated CDI object each (gold_frames.csv)
             Qwen: detect_open_format (setup B); Gemini 2.5 Flash: detect_open
  depiction  80 frames, Mira / Dora boxes with a CDI label (+ format) (depiction_boxes.csv)
             Qwen: detect_open_format; Gemini 2.5 Flash: detect_open (asks for formats)

For each human object: the model's best-overlapping box (IoU >= 0.5) and whether its open
label maps to the human category: same word, a near-synonym (vocab_groups.related), or the
head noun after dropping depiction words ("toy car" -> car, "picture of a giraffe" ->
giraffe, "sippy cup" -> cup). Also: the category named anywhere in the frame (no box), and
the open labels that map to no CDI word at all (what the open vocabulary adds).

Outputs (results/, aggregate only): recovery_overall.csv, recovery_by_category.csv,
naming_pairs.csv (human category -> model label on matched boxes), beyond_cdi_labels.csv.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "gemini_pilot"
sys.path.insert(0, str(PILOT))
import run_gold_pilot as gp  # noqa: E402
from vocab_groups import related  # noqa: E402
from compare_qwen_gemini import iou, load, objects  # noqa: E402

PRIV = PILOT / "out" / "private"
RES = HERE / "results"
W, H = 512, 910
VOCAB = set(gp.load_vocab()[0])
DEPICT = re.compile(r"^(toy|stuffed|plush|plastic|wooden|rubber|cartoon|picture of (a |an |the )?|drawing of (a |an |the )?|"
                    r"photo of (a |an |the )?|image of (a |an |the )?|painting of (a |an |the )?|baby )+")
MODELS = {
    "montage72": {"Qwen3.8-27B": PRIV / "qwen3.8-27b-fp8__detect_open_format__p1_montage72.jsonl",
                  "Gemini 2.5 Flash": PRIV / "gemini-2.5-flash__detect_open__p1_montage72.jsonl"},
    "depiction": {"Qwen3.8-27B": PRIV / "qwen3.8-27b-fp8__detect_open_format__p1_depiction_frames.jsonl",
                  "Gemini 2.5 Flash": PRIV / "gemini-2.5-flash__detect_open__p1_depiction_frames.jsonl"},
}


def singular(w):
    for suf, rep in (("ies", "y"), ("ves", "f"), ("ses", "s"), ("s", "")):
        if w.endswith(suf) and len(w) > len(suf) + 2 and not w.endswith("ss"):
            return w[: -len(suf)] + rep
    return w


# Everyday names the CDI list spells differently (only where the meaning is the same).
SYNONYMS = {"television": "tv", "tv screen": "tv", "sofa": "couch", "phone": "telephone", "cell phone": "telephone",
            "cellphone": "telephone", "mobile phone": "telephone", "smartphone": "telephone", "fridge": "refrigerator",
            "teddy bear": "teddybear", "eyeglasses": "glasses", "pram": "stroller", "buggy": "stroller"}


def to_cdi(label: str) -> str | None:
    """Map an open label to a CDI word, or None."""
    lab = re.sub(r"[^a-z ]", " ", label.lower()).strip()
    lab = re.sub(r"\s+", " ", lab)
    core = DEPICT.sub("", lab).strip() or lab
    for cand in (lab, core, singular(core)):          # whole name first: synonyms, joined spelling
        if cand in SYNONYMS:
            return SYNONYMS[cand]
        if cand.replace(" ", "") in VOCAB:
            return cand.replace(" ", "")
    for cand in (lab, core, singular(core), core.split()[-1] if core else "", singular(core.split()[-1]) if core else ""):
        if cand in SYNONYMS:
            return SYNONYMS[cand]
        if cand in VOCAB:
            return cand
        hit = next((v for v in VOCAB if related(cand, v)), None)
        if hit:
            return hit
    return None


def matches(label, category):
    c = to_cdi(label)
    return c is not None and (c == category or related(c, category))


def human_objects():
    g = pd.read_csv(PRIV / "gold_frames.csv", dtype={"frame_id": str})
    g["unit_id"] = g.video + "/" + g.frame_id
    keep = {l.strip() for l in open(PRIV / "montage72.txt") if l.strip()}
    g = g[g.unit_id.isin(keep)].drop_duplicates("unit_id")
    m72 = pd.DataFrame({"set": "montage72", "unit_id": g.unit_id, "category": g.category,
                        "box": [[r.ymin / H * 1000, r.xmin / W * 1000, r.ymax / H * 1000, r.xmax / W * 1000] for r in g.itertuples()]})
    d = pd.read_csv(PRIV / "depiction_boxes.csv")
    d = d[d.label.isin(VOCAB)]
    dep = pd.DataFrame({"set": "depiction", "unit_id": d.unit_id, "category": d.label,
                        "box": [[r.ymin / H * 1000, r.xmin / W * 1000, r.ymax / H * 1000, r.xmax / W * 1000] for r in d.itertuples()]})
    return pd.concat([m72, dep], ignore_index=True)


def main():
    RES.mkdir(exist_ok=True)
    hum = human_objects()
    rows, pairs, beyond = [], [], collections.defaultdict(collections.Counter)
    for s, models in MODELS.items():
        for name, path in models.items():
            recs = load(path)
            for u, r in recs.items():           # labels with no CDI word, counted once per frame
                if s == "depiction" or u in set(hum.unit_id):
                    for lab in {x["label"].strip().lower() for x in objects(r) or []}:
                        if to_cdi(lab) is None:
                            beyond[name][lab] += 1
            for h in hum[hum.set == s].itertuples():
                o = objects(recs.get(h.unit_id)) or []
                best = max(((iou(h.box, x["box_2d"]), x) for x in o), key=lambda t: t[0], default=(0, None))
                boxed = best[0] >= 0.5
                named_on_box = boxed and matches(best[1]["label"], h.category)
                named_anywhere = any(matches(x["label"], h.category) for x in o)
                rows.append({"set": s, "model": name, "unit_id": h.unit_id, "category": h.category,
                             "boxed": boxed, "recovered": named_on_box, "named_anywhere": named_anywhere})
                if boxed:
                    pairs.append({"model": name, "category": h.category, "label": best[1]["label"].strip().lower(),
                                  "maps_to": to_cdi(best[1]["label"]) or "(none)"})
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    ov = d.groupby(["set", "model"]).agg(objects=("recovered", "size"), box_on_object=("boxed", "mean"),
                                         recovered=("recovered", "mean"), named_anywhere=("named_anywhere", "mean"))
    ov.round(3).to_csv(RES / "recovery_overall.csv")
    print("== overall (box = IoU >= 0.5 with the human box; recovered = box AND label maps to the category)\n",
          ov.round(3).to_string())
    pv = d.pivot_table(index="category", columns="model", values="recovered", aggfunc="mean")
    pv["n"] = d[d.model == "Qwen3.8-27B"].groupby("category").size()
    both = d.pivot_table(index=["set", "unit_id", "category"], columns="model", values="recovered").reset_index()
    q, g = both["Qwen3.8-27B"].astype(bool), both["Gemini 2.5 Flash"].astype(bool)
    print(f"\n== agreement per object: both {int((q & g).sum())}, Qwen only {int((q & ~g).sum())}, "
          f"Gemini only {int((~q & g).sum())}, neither {int((~q & ~g).sum())} (of {len(both)})")
    pv.sort_values("n", ascending=False).round(2).to_csv(RES / "recovery_by_category.csv")
    print("\n== by category (share recovered; n = human objects)\n", pv.sort_values("n", ascending=False).round(2).head(40).to_string())
    p = pd.DataFrame(pairs)
    np_ = p.groupby(["model", "category", "label", "maps_to"]).size().rename("n").reset_index().sort_values(["category", "model"])
    np_.to_csv(RES / "naming_pairs.csv", index=False)
    mism = np_[(np_.maps_to != np_.category) & ~np_.apply(lambda r: related(r.maps_to, r.category), axis=1)]
    print("\n== boxes on the object whose label does NOT map to the human category (naming differences)\n",
          mism.head(40).to_string(index=False))
    b = pd.DataFrame([{"model": m, "label": l, "frames": n} for m, c in beyond.items() for l, n in c.items()])
    b.sort_values(["model", "frames"], ascending=[True, False]).to_csv(RES / "beyond_cdi_labels.csv", index=False)
    for m in beyond:
        top = beyond[m].most_common(30)
        print(f"\n== {m}: labels with no CDI word ({len(beyond[m])} distinct) — top 30 by frames\n  " +
              ", ".join(f"{l} ({n})" for l, n in top))


if __name__ == "__main__":
    main()
