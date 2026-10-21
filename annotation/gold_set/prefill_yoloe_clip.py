#!/usr/bin/env python3
"""YOLOE + CLIP boxes for the gold set v2 frames, with the production 2025.2 settings.

Why rerun: production 2025.2 YOLOE output covers only videos that are not in the 868-hour
release (those were run earlier under 868-hour names, with frames extracted separately),
so 658 of the 1,000 sampled frames have no 2025.2 boxes. One run on all frames keeps the
pre-fill uniform; `--stage check` compares it with production on the frames it covers.

Settings copied from production
  YOLOE  yoloe/run_yoloe.py: pretrain/yoloe-v8l-seg.pt, set_classes(CDI nouns + "person",
         yoloe/predict_frames.py cdi_list()), model.predict defaults, every box kept.
  CLIP   image-embedding/run_cdi_crop_pipeline.py: openai/clip-vit-base-patch32 in bfloat16,
         crop = clamp_box() integer box, cosine(L2 image, L2 "a photo of a {label}") >= 0.27,
         person/picture dropped, valid129 classes only.

Stages (two environments: YOLOE needs the YOLOE build of ultralytics)
  yoloe  -> out/private/prefill/yoloe_raw.csv
  clip   -> out/private/prefill/yoloe_clip.csv   (all valid129 boxes, clip_sim, passed)
  check  -> out/private/prefill/check_vs_production.json

Usage
  YOLOE_PY=/data/tanawm/miniconda3/envs/yoloe/bin/python   # read-only use of a lab env
  CUDA_VISIBLE_DEVICES=<gpu> $YOLOE_PY prefill_yoloe_clip.py --stage yoloe
  CUDA_VISIBLE_DEVICES=<gpu> python prefill_yoloe_clip.py --stage clip     # vislearnlabpy
  python prefill_yoloe_clip.py --stage check
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out" / "private" / "prefill"
FRAMES = HERE / "out" / "private" / "frames.csv"
YOLOE_DIR = REPO / "yoloe"
VALID129 = REPO / "data" / "included_categories_valid129.txt"
EXCLUDE = {"person", "picture"}
CLIP_MODEL = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo of a {label}"
THRESHOLD = 0.27
PROD_INDEX = Path("/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full/crops_index.csv.gz")


def cdi_list() -> list[str]:
    """Same as yoloe/predict_frames.py: CDI nouns without spaces, plus person."""
    df = pd.read_csv(YOLOE_DIR / "tools" / "MCDI_items_with_AoA.csv")
    df = df[df["lexical_category"] == "nouns"]
    df = df[~df["uni_lemma"].str.contains(" ", na=False)]
    return df["uni_lemma"].astype(str).tolist() + ["person"]


def stage_yoloe():
    from PIL import Image
    from ultralytics import YOLOE

    names = cdi_list()
    model = YOLOE(str(YOLOE_DIR / "pretrain" / "yoloe-v8l-seg.pt"))
    model.set_classes(names, model.get_text_pe(names))
    model.to("cuda")
    frames = pd.read_csv(FRAMES)
    rows = []
    for r in frames.itertuples():
        res = model.predict(Image.open(r.frame_path).convert("RGB"), verbose=False)[0]
        b = res.boxes
        for xyxy, conf, cls in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
            rows.append(dict(frame_uid=r.frame_uid, xmin=xyxy[0], ymin=xyxy[1], xmax=xyxy[2],
                             ymax=xyxy[3], yolo_conf=float(conf), class_name=res.names[int(cls)]))
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "yoloe_raw.csv", index=False)
    print(f"{len(d)} boxes on {d.frame_uid.nunique()} of {len(frames)} frames")


def clamp_box(xmin, ymin, xmax, ymax, w, h):
    x0, x1 = sorted((max(0, min(w, float(xmin))), max(0, min(w, float(xmax)))))
    y0, y1 = sorted((max(0, min(h, float(ymin))), max(0, min(h, float(ymax)))))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def stage_clip():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    valid = {ln.strip().lower() for ln in VALID129.read_text().splitlines() if ln.strip()}
    d = pd.read_csv(OUT / "yoloe_raw.csv")
    d["class_name"] = d.class_name.str.strip().str.lower()
    d = d[d.class_name.isin(valid) & ~d.class_name.isin(EXCLUDE)].reset_index(drop=True)
    paths = pd.read_csv(FRAMES).set_index("frame_uid").frame_path

    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to("cuda").eval()

    def unwrap(o):
        return o if torch.is_tensor(o) else (o.pooler_output if hasattr(o, "pooler_output") else o[0])

    names = sorted(d.class_name.unique())
    with torch.inference_mode():
        t = proc(text=[TEMPLATE.format(label=n) for n in names], return_tensors="pt",
                 padding=True, truncation=True)
        t = {k: v.to("cuda") for k, v in t.items()}
        tv = torch.nn.functional.normalize(unwrap(model.get_text_features(**t)).float(), dim=-1).cpu().numpy()
    text = dict(zip(names, tv))

    sims = np.full(len(d), np.nan)
    for uid, g in d.groupby("frame_uid"):
        rgb = np.asarray(Image.open(paths[uid]).convert("RGB"))
        h, w = rgb.shape[:2]
        crops, idx = [], []
        for i, r in g.iterrows():
            b = clamp_box(r.xmin, r.ymin, r.xmax, r.ymax, w, h)
            if b:
                crops.append(rgb[b[1]:b[3], b[0]:b[2]])
                idx.append(i)
        if not crops:
            continue
        with torch.inference_mode():
            x = proc(images=crops, return_tensors="pt")
            x = {k: v.to("cuda") for k, v in x.items() if torch.is_tensor(v)}
            e = unwrap(model.get_image_features(**x)).float().cpu().numpy()
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        for j, i in enumerate(idx):
            sims[i] = float(e[j] @ text[d.at[i, "class_name"]])
    d["clip_sim"] = sims
    d["passed"] = d.clip_sim >= THRESHOLD
    d.to_csv(OUT / "yoloe_clip.csv", index=False)
    print(f"{len(d)} valid129 boxes, {int(d.passed.sum())} pass CLIP >= {THRESHOLD} "
          f"on {d[d.passed].frame_uid.nunique()} frames")


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def stage_check():
    """Our boxes vs production crops_index on the sampled frames it covers."""
    f = pd.read_csv(FRAMES)
    ours = pd.read_csv(OUT / "yoloe_clip.csv")
    vids = set(f.video_id)
    prod = []
    for chunk in pd.read_csv(PROD_INDEX, chunksize=2_000_000,
                             usecols=["video_id", "frame_number", "class_name", "xmin", "ymin",
                                      "xmax", "ymax", "yolo_conf", "clip_sim", "passed"]):
        prod.append(chunk[chunk.video_id.isin(vids)])
    prod = pd.concat(prod).merge(f[["frame_uid", "video_id", "frame_number"]],
                                 on=["video_id", "frame_number"])
    covered = set(prod.frame_uid)
    # covered frames also include frames where production found no valid129 box; take the
    # frames of videos production processed
    ours_c = ours[ours.frame_uid.isin(covered)]
    matched, dsim, dconf, flips = 0, [], [], 0
    for uid, g in prod.groupby("frame_uid"):
        o = ours_c[ours_c.frame_uid == uid]
        for r in g.itertuples():
            cand = o[o.class_name == r.class_name]
            best = max(((iou((r.xmin, r.ymin, r.xmax, r.ymax), (c.xmin, c.ymin, c.xmax, c.ymax)), c)
                        for c in cand.itertuples()), default=(0, None), key=lambda t: t[0])
            if best[0] >= 0.9:
                matched += 1
                dsim.append(best[1].clip_sim - r.clip_sim)
                dconf.append(best[1].yolo_conf - r.yolo_conf)
                flips += int(bool(best[1].passed) != bool(r.passed))
    out = {
        "frames_compared": len(covered), "production_boxes": len(prod),
        "our_boxes_on_those_frames": len(ours_c),
        "production_boxes_matched_iou90_same_class": matched,
        "match_rate": round(matched / max(1, len(prod)), 4),
        "clip_sim_abs_diff_median_max": [float(np.median(np.abs(dsim))), float(np.max(np.abs(dsim)))] if dsim else None,
        "yolo_conf_abs_diff_median_max": [float(np.median(np.abs(dconf))), float(np.max(np.abs(dconf)))] if dconf else None,
        "pass_flag_disagreements": flips,
    }
    (OUT / "check_vs_production.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["yoloe", "clip", "check"], required=True)
    {"yoloe": stage_yoloe, "clip": stage_clip, "check": stage_check}[ap.parse_args().stage]()
