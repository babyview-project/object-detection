#!/usr/bin/env python3
"""Merge YOLOE+CLIP and Gemini boxes into one pre-fill and write CVAT task bundles.

Open vocabulary: RAs box every discrete object in view, not only the valid129 categories.
CVAT has one label, "object", with a free-text attribute "name" and a select attribute
"format". Names are mapped to categories after annotation.

Pre-fill per frame
  YOLOE+CLIP  boxes that pass CLIP >= 0.27 (prefill_yoloe_clip.py): the production output,
              valid129 classes only. Name = the class.
  YOLOE       person boxes as YOLOE output them (no CLIP step), source "yoloe_person".
  Gemini      every gemini-2.5-flash detect_open_format box (open label + format, --max-obj 30;
              chosen over detect_combo after compare_prefill.py), name = Gemini's open label,
              except labels in NOT_DISCRETE (walls, floors, rooms, ...) and PARTS (handles,
              knobs, ...). Match class = "person" if the label is person, else the label or its
              last word if that is a valid129 class (cdi_word, when a run has one), else none.
              The match class only pairs Gemini with YOLOE boxes in merged.csv; categories for
              analysis come from mapping RA names after annotation. Gemini's duplicate boxes
              (same name or class, IoU > DUP_IOU) are collapsed.
  Merge       same match class and IoU >= MERGE_IOU -> one row (YOLOE coordinates in xmin..,
              Gemini's in g_xmin.., Gemini's name), source "both". Everything else is kept
              with its own source. Saved as merged.csv, the key for post-hoc comparisons.
RAs see Gemini's boxes only (sources in GEMINI_SOURCES, Gemini coordinates); YOLOE-only
boxes are hidden from them and kept in merged.csv. The format attribute is pre-filled with
Gemini's format (photo and drawing -> picture) so RAs correct it rather than label it from
scratch; boxes RAs draw start as "unset".

Per rater and batch: one CVAT task folder
  out/private/cvat/<rater>_batch<k>/
    images/<frame_uid>.jpg     frames renamed to opaque ids (no subject or video ids)
    annotations.zip            CVAT for images 1.1 (annotations.xml) -> Actions > Upload annotations
    labels.json                one label "object" (attributes name, format): paste into the
                               task's label constructor (Raw) before uploading
  Shared frames go into both raters' batch 1, mixed in and not flagged.
out/private/prefill/merged.csv  every pre-filled box with its source (the key; RAs never get it)

Usage
  python build_cvat_tasks.py [--zip]
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRIV = HERE / "out" / "private"
FRAMES = PRIV / "frames.csv"
YOLOE_CLIP = PRIV / "prefill" / "yoloe_clip.csv"
GEMINI = HERE.parent / "gemini_pilot" / "out" / "private" / "gemini-2.5-flash__detect_open_format__p1_gold_v2_frames_cap30.jsonl"
VALID129 = REPO / "data" / "included_categories_valid129.txt"
EXCLUDE = {"person", "picture"}   # not analysis classes
# Not discrete objects, or parts of one: not pre-filled, and RAs do not box them
NOT_DISCRETE = {"wall", "walls", "floor", "flooring", "ceiling", "sky", "ground", "room", "kitchen",
                "bedroom", "bathroom", "living room", "hallway", "yard", "backyard", "road", "street",
                "brick wall", "none", ""}
PARTS = {"handle", "door handle", "knob", "door knob", "doorknob", "wheel", "trim", "door frame",
         "lid", "button", "label"}
YOLOE_RAW = PRIV / "prefill" / "yoloe_raw.csv"
FORMATS = ["unset", "real", "toy", "picture", "screen"]
GEMINI_FORMAT = {"real": "real", "toy": "toy", "photo": "picture", "drawing": "picture", "screen": "screen"}
MERGE_IOU, DUP_IOU = 0.5, 0.8
GEMINI_SOURCES = {"gemini", "both", "both_person"}   # pre-filled for RAs; the rest stay in merged.csv


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def load_gemini(frames: pd.DataFrame, classes: set[str], path: Path = GEMINI) -> pd.DataFrame:
    size = {r.frame_uid: Image.open(r.frame_path).size for r in frames.itertuples()}
    rows, n_err, n_drop = [], 0, 0
    recs = [json.loads(l) for l in open(path)]
    recs = {r["unit_id"]: r for r in recs}            # last record per frame wins (retries)
    missing = set(frames.frame_uid) - set(recs)
    assert not missing, f"Gemini has no output for {len(missing)} frames; rerun the pre-fill"
    for uid, rec in recs.items():
        if uid not in size:
            continue
        try:
            objs = json.loads(rec["raw"])["objects"] if not rec.get("error") else []
        except (TypeError, ValueError, KeyError):
            objs, n_err = [], n_err + 1
        w, h = size[uid]
        for o in objs:
            name = " ".join(str(o.get("label", "")).strip().lower().split())
            word = str(o.get("cdi_word", "")).strip().lower()
            b = o.get("box_2d") or []
            if name in NOT_DISCRETE or name in PARTS or len(b) != 4:
                n_drop += 1
                continue
            # detect_open_format has no cdi_word: use the label, or its last word ("toy car" -> car)
            head = name.split()[-1] if name else ""
            cls = ("person" if "person" in (name, word) else word if word in classes
                   else name if name in classes else head if head in classes else "")
            y0, x0, y1, x1 = (float(v) for v in b)
            box = (min(x0, x1) * w / 1000, min(y0, y1) * h / 1000, max(x0, x1) * w / 1000, max(y0, y1) * h / 1000)
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            rows.append(dict(frame_uid=uid, class_name=cls, name=name, xmin=box[0], ymin=box[1], xmax=box[2],
                             ymax=box[3], gemini_cdi_word=word,
                             gemini_format=GEMINI_FORMAT.get(str(o.get("format", "")).lower(), "")))
    print(f"Gemini: {len(recs)} frames, {n_err} unparsable responses, {len(rows)} boxes kept, "
          f"{n_drop} not discrete or parts")
    return pd.DataFrame(rows)


def dedup(g: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for r in g.itertuples():
        b = (r.xmin, r.ymin, r.xmax, r.ymax)
        if not any((k.name == r.name or (r.class_name and k.class_name == r.class_name))
                   and iou(b, (k.xmin, k.ymin, k.xmax, k.ymax)) > DUP_IOU for k in keep):
            keep.append(r)
    return pd.DataFrame(keep).drop(columns="Index") if keep else g.iloc[0:0]


def merge(y: pd.DataFrame, g: pd.DataFrame, uids) -> pd.DataFrame:
    out = []
    for uid in uids:
        yy = y[y.frame_uid == uid]
        gg = dedup(g[g.frame_uid == uid])
        used = set()
        for r in yy.itertuples():
            a = (r.xmin, r.ymin, r.xmax, r.ymax)
            best, bj = 0.0, None
            for j, s in gg.iterrows():
                if j in used or s.class_name != r.class_name:
                    continue
                v = iou(a, (s.xmin, s.ymin, s.xmax, s.ymax))
                if v > best:
                    best, bj = v, j
            src = r.ysrc
            row = dict(frame_uid=uid, class_name=r.class_name, name=r.class_name, xmin=r.xmin, ymin=r.ymin,
                       xmax=r.xmax, ymax=r.ymax, clip_sim=r.clip_sim, yolo_conf=r.yolo_conf,
                       gemini_name="", gemini_cdi_word="", gemini_format="", gemini_iou=np.nan)
            if bj is not None and best >= MERGE_IOU:
                used.add(bj)
                src = "both" if src == "yoloe_clip" else "both_person"
                row.update(name=gg.at[bj, "name"], gemini_name=gg.at[bj, "name"],
                           gemini_cdi_word=gg.at[bj, "gemini_cdi_word"],
                           gemini_format=gg.at[bj, "gemini_format"], gemini_iou=best,
                           **{f"g_{c}": gg.at[bj, c] for c in ("xmin", "ymin", "xmax", "ymax")})
            out.append({**row, "source": src})
        for j, s in gg.iterrows():
            if j not in used:
                out.append(dict(frame_uid=uid, class_name=s.class_name, name=s["name"], xmin=s.xmin,
                                ymin=s.ymin, xmax=s.xmax, ymax=s.ymax, clip_sim=np.nan, yolo_conf=np.nan,
                                gemini_name=s["name"], gemini_cdi_word=s.gemini_cdi_word,
                                gemini_format=s.gemini_format, gemini_iou=np.nan, g_xmin=s.xmin, g_ymin=s.ymin,
                                g_xmax=s.xmax, g_ymax=s.ymax, source="gemini"))
    m = pd.DataFrame(out)
    m["gemini_format"] = m.gemini_format.fillna("")
    m.insert(0, "box_id", [f"{u}_{i:03d}" for u, i in zip(m.frame_uid, m.groupby("frame_uid").cumcount())])
    return m


def gemini_prefill(m: pd.DataFrame) -> pd.DataFrame:
    """What RAs see: Gemini's boxes only, in Gemini's coordinates (YOLOE-only boxes left out)."""
    p = m[m.source.isin(GEMINI_SOURCES)].copy()
    for c in ("xmin", "ymin", "xmax", "ymax"):
        p[c] = p[f"g_{c}"].astype(float)
    return p


def labels_json() -> list[dict]:
    name = {"name": "name", "mutable": False, "input_type": "text", "default_value": "", "values": [""]}
    fmt = {"name": "format", "mutable": False, "input_type": "select", "default_value": "unset",
           "values": FORMATS}
    return [{"name": "object", "attributes": [name, fmt]}]


def cvat_xml(frames: pd.DataFrame, boxes: pd.DataFrame) -> str:
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    for i, r in enumerate(frames.sort_values("frame_uid").itertuples()):
        w, h = Image.open(r.frame_path).size
        lines.append(f'  <image id="{i}" name="{r.frame_uid}.jpg" width="{w}" height="{h}">')
        for b in boxes[boxes.frame_uid == r.frame_uid].itertuples():
            lines.append(f"    <box label=\"object\" source=\"auto\" occluded=\"0\" "
                         f"xtl=\"{b.xmin:.2f}\" ytl=\"{b.ymin:.2f}\" xbr=\"{b.xmax:.2f}\" ybr=\"{b.ymax:.2f}\" z_order=\"0\">")
            lines.append(f'      <attribute name="name">{escape(b.name)}</attribute>')
            lines.append(f'      <attribute name="format">{escape(b.gemini_format or "unset")}</attribute>')
            lines.append("    </box>")
        lines.append("  </image>")
    lines.append("</annotations>")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="also zip each task folder")
    args = ap.parse_args()
    classes = sorted({l.strip().lower() for l in VALID129.read_text().splitlines() if l.strip()} - EXCLUDE)
    frames = pd.read_csv(FRAMES)
    y = pd.read_csv(YOLOE_CLIP)
    y = y[y.passed.astype(bool) & y.class_name.isin(classes)].assign(ysrc="yoloe_clip")
    yp = pd.read_csv(YOLOE_RAW)
    yp = yp[yp.class_name.str.strip().str.lower() == "person"].assign(
        class_name="person", clip_sim=np.nan, passed=False, ysrc="yoloe_person")
    y = pd.concat([y, yp], ignore_index=True)
    g = load_gemini(frames, set(classes))
    m = merge(y, g, frames.frame_uid)
    (PRIV / "prefill").mkdir(parents=True, exist_ok=True)
    m.to_csv(PRIV / "prefill" / "merged.csv", index=False)
    p = gemini_prefill(m)

    raters = sorted(r for r in frames.rater.unique() if r != "shared")
    per_frame = p.groupby("frame_uid").size().reindex(frames.frame_uid, fill_value=0)
    summary = {"gemini_run": GEMINI.name, "merged_boxes": len(m), "by_source": m.source.value_counts().to_dict(),
               "prefilled_boxes": len(p), "boxes_in_valid129_or_person": int((p.class_name != "").sum()),
               "distinct_names": int(p.name.nunique()),
               "prefilled_format": p.gemini_format.replace("", "unset").value_counts().to_dict(),
               "boxes_per_frame_median_max": [float(per_frame.median()), int(per_frame.max())],
               "frames_without_boxes": int((per_frame == 0).sum()), "tasks": {}}
    root = PRIV / "cvat"
    for r in raters:
        for k in sorted(frames[frames.rater == r].batch.unique()):
            f = frames[((frames.rater == r) & (frames.batch == k))
                       | ((frames.rater == "shared") & (frames.batch == k))]
            d = root / f"{r}_batch{k}"
            if d.exists():
                shutil.rmtree(d)
            (d / "images").mkdir(parents=True)
            for fr in f.itertuples():
                shutil.copy2(fr.frame_path, d / "images" / f"{fr.frame_uid}.jpg")
            with zipfile.ZipFile(d / "annotations.zip", "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("annotations.xml", cvat_xml(f, p[p.frame_uid.isin(f.frame_uid)]))
            (d / "labels.json").write_text(json.dumps(labels_json(), indent=1) + "\n")
            if args.zip:
                shutil.make_archive(str(d), "zip", root_dir=d)
            summary["tasks"][d.name] = {"frames": len(f), "shared": int((f.rater == "shared").sum()),
                                        "boxes": int(p.frame_uid.isin(f.frame_uid).sum())}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
