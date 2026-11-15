"""Load Qwen open-vocabulary detections (and YOLOE + CLIP on the same frames) as tables.

    import qwen_data as qd
    rec = qd.load_records("detect_open_format", "subj_S00320001_t1_pp0")   # main file + shard files
    obj = qd.objects_table(rec)          # one row per box: label, format, box, cdi word, ...
    fr = qd.frames_table(rec)            # one row per frame: video, date, age in months, objects, loops
    yo = qd.yoloe_table(fr)              # YOLOE + CLIP detections on exactly these frames

Box coordinates are Gemini-style box_2d on 0-1000 (ymin, xmin, ymax, xmax). `label` is Qwen's
label lowercased; `cat` is Qwen's category with light normalization only (norm_label: plurals,
spacing, a few exact synonyms, depiction words dropped since `format` carries them: "toy cars" ->
car, "picture of a dog" -> dog; bare "toy" stays toy); `domain` groups categories into the CDI semantic
domains (assign_domains: the CDI word's domain, else the majority of its 5 nearest CDI words by CLIP text
embedding); `cdi` maps a
label to a CDI word (category_recovery.to_cdi) or NaN.
Within a frame, same-label boxes overlapping a kept box at IoU > 0.7 are flagged `dup` (the
near-duplicates loops produce); `dedup()` drops them. Data stay private (unit ids name
videos): keep derived tables and executed notebooks in out/private/.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRIV = HERE.parent / "qwen_pilot" / "out" / "runs"
OD_ROOT = HERE.parents[1]
MODEL_TAG = "qwen3.8-27b-fp8"
CROPS_INDEX = Path("/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full/crops_index.csv.gz")
METADATA = OD_ROOT / "data" / "babyview-videos-metadata-2025.1.csv"


# Exact synonyms only, onto the spelling Qwen uses most; plurals that are the everyday singular kept.
SYNONYMS = {"couch": "sofa", "tv": "television", "tv screen": "television", "fridge": "refrigerator",
            "cell phone": "phone", "cellphone": "phone", "mobile phone": "phone", "smartphone": "phone",
            "teddy": "teddy bear", "eyeglasses": "glasses", "t shirt": "t-shirt", "tshirt": "t-shirt"}
KEEP_PLURAL = {"glasses", "sunglasses", "pants", "shorts", "jeans", "leggings", "tights", "pajamas", "overalls",
               "clothes", "scissors", "headphones", "binoculars", "tongs", "stairs", "blinds", "goggles", "earphones"}
DEPICT = re.compile(r"^(toy|stuffed|plush|plastic|wooden|rubber|cartoon|picture of|drawing of|photo of|image of|"
                    r"painting of|illustration of) +")


@lru_cache(maxsize=None)
def norm_label(label: str) -> str:
    from nltk.corpus import wordnet as wn
    lab = re.sub(r"[^a-z\- ]", " ", str(label).lower())
    lab = re.sub(r"^(a|an|the) ", "", re.sub(r"\s+", " ", lab).strip())
    while True:                                   # "toy plastic car" -> car; bare "toy" stays toy
        core = DEPICT.sub("", lab, count=1)
        if core == lab or not core:
            break
        lab = re.sub(r"^(a|an|the) ", "", core)
    words = lab.split()
    if words and words[-1] not in KEEP_PLURAL and lab not in KEEP_PLURAL:
        sing = wn.morphy(words[-1], "n")
        if sing and sing != words[-1] and words[-1].endswith("s"):
            words[-1] = sing
    lab = " ".join(words)
    return SYNONYMS.get(lab, lab)


# Frequent labels whose nearest CDI words point to the wrong group (checked on S00320001).
DOMAIN_OVERRIDES = {"shelf": "furniture_rooms", "ceiling": "furniture_rooms", "floor": "furniture_rooms",
                    "wall": "furniture_rooms", "doorway": "furniture_rooms", "counter": "furniture_rooms",
                    "puzzle": "toys", "puzzle piece": "toys", "block": "toys", "building block": "toys",
                    "rug": "furniture_rooms", "blind": "furniture_rooms", "bookshelf": "furniture_rooms",
                    "mug": "household", "knob": "household", "coffee maker": "household", "sponge": "household",
                    "tumbler": "household", "appliance": "household", "toaster": "household", "tablet": "household",
                    "laptop": "household", "keyboard": "household", "screen": "household", "magnet": "household",
                    "painting": "household", "drawing": "household", "picture frame": "household", "frame": "household",
                    "poster": "household", "wall hanging": "household", "piano": "household", "hanger": "household",
                    "pacifier": "household", "backpack": "clothing", "sunglasses": "clothing", "helmet": "clothing",
                    "hair clip": "clothing", "headband": "clothing", "mask": "clothing",
                    # 2026-10-29: open-category domain fixes (toys = playthings; buildings are places; computer mice)
                    "building": "outside", "house": "outside", "book": "household", "magazine": "household",
                    "notebook": "household", "pen": "household", "pencil": "household", "marker": "household",
                    "card": "household", "computer mouse": "household", "mouse": "animals"}
PEOPLE = ["person", "woman", "man", "baby", "child", "girl", "boy", "mom", "dad", "adult"]
DOMAIN_CACHE = HERE / "results" / "category_domains.csv"


def domain_of(cat: str) -> str:
    return assign_domains([cat])[cat]


@lru_cache(maxsize=1)
def hierarchy() -> pd.DataFrame:
    """Union of results/category_hierarchy_*.csv (qwen_hierarchy.py: Qwen reads the whole name -> basic level,
    domain, generic). Empty if none yet."""
    parts = [pd.read_csv(p) for p in sorted((HERE / "results").glob("category_hierarchy_*.csv"))]
    if not parts:
        return pd.DataFrame(columns=["cat", "qwen_basic", "domain", "qwen_generic"]).set_index("cat")
    return pd.concat(parts).drop_duplicates("cat").set_index("cat")


def assign_domains(cats, k: int = 5) -> dict[str, str]:
    """Category -> CDI semantic domain: override; else the hierarchy table (Qwen judged the whole name, CDI
    convention for CDI words); else, for categories not in it yet, the older automatic rule: the CDI word's domain
    when the category maps to one, else the majority domain of its k nearest CDI words by CLIP ViT-B/32 text
    embedding (cached in results/category_domains.csv). Labels only."""
    import category_recovery as cr
    cache = pd.read_csv(DOMAIN_CACHE).set_index("cat").domain.to_dict() if DOMAIN_CACHE.exists() else {}
    hier = hierarchy().domain.dropna().to_dict()
    out, todo = {}, []
    for c in dict.fromkeys(cats):
        if c in DOMAIN_OVERRIDES:
            out[c] = DOMAIN_OVERRIDES[c]
        elif c in hier:
            out[c] = hier[c]
        elif c in cache:
            out[c] = cache[c]
        elif (w := cr.to_cdi(c)) is not None and _cdi_domains().get(w):
            out[c] = _cdi_domains()[w]
        else:
            todo.append(c)
    if todo:
        anchors = {w: d for w, d in _cdi_domains().items()} | {w: "people" for w in PEOPLE}
        names = list(anchors)
        A, T = _clip_text(names), _clip_text(todo)
        sim = T @ A.T
        for c, row in zip(todo, sim):
            nn = [anchors[names[i]] for i in np.argsort(-row)[:k]]
            out[c] = max(set(nn), key=lambda d: (nn.count(d), -nn.index(d)))
        new = pd.DataFrame({"cat": todo, "domain": [out[c] for c in todo]})
        DOMAIN_CACHE.parent.mkdir(exist_ok=True)
        pd.concat([pd.DataFrame({"cat": list(cache), "domain": list(cache.values())}), new]) \
            .drop_duplicates("cat").sort_values("cat").to_csv(DOMAIN_CACHE, index=False)
    return out


def _clip_text(names: list[str]) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    vecs = []
    with torch.inference_mode():
        for i in range(0, len(names), 256):
            t = tok([f"a photo of a {n}" for n in names[i:i + 256]], padding=True, truncation=True, return_tensors="pt")
            e = m.get_text_features(**t)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            vecs.append(torch.nn.functional.normalize(e.float(), dim=-1).numpy())
    return np.concatenate(vecs)


@lru_cache(maxsize=1)
def _cdi_domains() -> dict[str, str]:
    d = pd.read_csv(OD_ROOT / "data" / "cdi_words.csv", usecols=["uni_lemma", "category"])
    d["u"] = d.uni_lemma.astype(str).str.strip().str.lower()
    return d.drop_duplicates("u").set_index("u").category.to_dict()


def load_records(task: str, tag: str) -> dict[str, dict]:
    """Records from <model>__<task>__p1_<tag>.jsonl and its unmerged shard files (last parsed record wins)."""
    base = PRIV / f"{MODEL_TAG}__{task}__p1_{tag}.jsonl"
    paths = [base] + sorted(base.parent.glob(base.name + ".shard*-of-*"))
    recs: dict[str, dict] = {}
    for p in paths:
        if not p.exists() or p.name.endswith(".merged"):
            continue
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:          # a shard may be mid-write
                    continue
                if r.get("error") is None or r["unit_id"] not in recs:
                    recs[r["unit_id"]] = r
    return recs


def _iou(a, b):
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def objects_table(recs: dict[str, dict]) -> pd.DataFrame:
    import category_recovery as cr
    rows = []
    for u, r in recs.items():
        if r.get("error") is not None:
            continue
        kept: list[tuple[str, list]] = []
        for i, o in enumerate(json.loads(r["raw"]).get("objects", [])):
            b = o.get("box_2d", [])
            if len(b) != 4:
                continue
            y0, x0, y1, x1 = (min(1000, max(0, v)) for v in b)
            if x1 <= x0 or y1 <= y0:
                continue
            lab = str(o.get("label", "")).strip().lower()
            box = [y0, x0, y1, x1]
            dup = any(l == lab and _iou(box, k) > 0.7 for l, k in kept)
            if not dup:
                kept.append((lab, box))
            rows.append({"unit_id": u, "obj": i, "label": lab, "format": o.get("format", ""),
                         "ymin": y0, "xmin": x0, "ymax": y1, "xmax": x1,
                         "area": (y1 - y0) * (x1 - x0) / 1e6, "dup": dup})
    obj = pd.DataFrame(rows)
    cdi = {l: cr.to_cdi(l) for l in obj.label.unique()}
    obj["cdi"] = obj.label.map(cdi)
    cat = {l: norm_label(l) for l in obj.label.unique()}
    obj["cat"] = obj.label.map(cat)
    obj["domain"] = obj.cat.map(assign_domains(list(obj.cat.unique())))
    h = hierarchy()
    obj["basic"] = obj.cat.map(h.qwen_basic.dropna().to_dict()).fillna(obj.cat)
    obj["generic"] = obj.cat.map(h.qwen_generic.dropna().astype(bool).to_dict()).fillna(False).astype(bool)
    obj["domain_source"] = np.where(obj.cat.isin(h.index), "hierarchy", "auto")
    obj["video_id"] = obj.unit_id.str.rsplit("/", n=1).str[0]
    return obj


def dedup(obj: pd.DataFrame) -> pd.DataFrame:
    return obj[~obj.dup]


def _marching(objs: list[dict]) -> int:
    best = run = 1
    for a, b in zip(objs, objs[1:]):
        A, B = a["box_2d"], b["box_2d"]
        hA, wA, hB, wB = A[2] - A[0], A[3] - A[1], B[2] - B[0], B[3] - B[1]
        same = str(a.get("label", "")).strip().lower() == str(b.get("label", "")).strip().lower()
        run = run + 1 if same and abs(hA - hB) <= max(8, .1 * hA) and abs(wA - wB) <= max(8, .1 * wA) else 1
        best = max(best, run)
    return best if objs else 0


def birth_dates() -> dict[str, pd.Timestamp]:
    m = pd.read_csv(METADATA, encoding="utf-8-sig", usecols=["subject_id", "date_birth_rounded"]).dropna()
    m["birth"] = pd.to_datetime(m.date_birth_rounded, format="%m/%d/%Y", errors="coerce")
    return m.dropna().groupby("subject_id").birth.first().to_dict()


def frames_table(recs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for u, r in recs.items():
        objs = [o for o in json.loads(r["raw"]).get("objects", []) if len(o.get("box_2d", [])) == 4] \
            if r.get("error") is None else []
        rows.append({"unit_id": u, "parsed": r.get("error") is None, "n_objects": len(objs),
                     "at_cap": len(objs) >= 30, "marching": _marching(objs) >= 8, "out_tok": r.get("out_tok")})
    fr = pd.DataFrame(rows)
    fr["video_id"] = fr.unit_id.str.rsplit("/", n=1).str[0]
    fr["frame"] = fr.unit_id.str.rsplit("/", n=1).str[1].astype(int)
    fr["subject"] = fr.video_id.str.split("_").str[0]
    fr["date"] = pd.to_datetime(fr.video_id.str.split("_").str[1], errors="coerce")
    birth = birth_dates()
    fr["age_months"] = [(d - birth[s]).days / 30.44 if s in birth and pd.notna(d) else np.nan
                        for s, d in zip(fr.subject, fr.date)]
    return fr


def yoloe_table(fr: pd.DataFrame, chunksize: int = 2_000_000) -> pd.DataFrame:
    """YOLOE + CLIP rows of crops_index.csv.gz on exactly the frames in `fr` (one pass, ~1-2 min)."""
    keys = set(fr.unit_id)
    videos = set(fr.video_id)
    parts = []
    cols = ["video_id", "frame_number", "class_name", "clip_sim", "passed", "xmin", "ymin", "xmax", "ymax"]
    for ch in pd.read_csv(CROPS_INDEX, usecols=cols, chunksize=chunksize):
        ch = ch[ch.video_id.isin(videos)]
        if ch.empty:
            continue
        ch = ch.assign(unit_id=ch.video_id + "/" + ch.frame_number.astype(int).astype(str).str.zfill(5))
        parts.append(ch[ch.unit_id.isin(keys)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols + ["unit_id"])
