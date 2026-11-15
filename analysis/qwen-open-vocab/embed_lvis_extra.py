#!/usr/bin/env python3
"""CLIP + DINOv3 embeddings of LVIS v1 box crops for the LVIS categories mapped to Tier 1 that have no crop embeddings yet.

Same as analysis/comparison-datasets/data_prep/run_lvis_crop_embeddings.sh: ground-truth boxes (train + val), exact box
crop from the COCO image, no padding, no CLIP or size filter (boxes under 1 px dropped); CLIP ViT-B/32
get_image_features and DINOv3 ViT-B/16 pooler_output.

Output (out/crossdataset/):
  lvis_extra_category_embeddings_{clip,dinov3}.npz   category_ids, names, counts, sums (sum of L2-normalized vectors)
  lvis_extra_crops_sample.npz                          up to 300 crops per category: ann_id, category_id, path, bbox,
                                                       clip, dinov3 (float16)  (format probe, montages)

  CUDA_VISIBLE_DEVICES=7 python embed_lvis_extra.py      # qwen env
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "crossdataset"
LVIS = Path("/data2/j7yang/lvis")
SAMPLE_PER_CAT, BATCH = 300, 256


def manifest(cat_ids: set[int]) -> pd.DataFrame:
    rows = []
    for split in ("train", "val"):
        d = json.load(open(LVIS / "annotations" / f"lvis_v1_{split}.json"))
        img = {i["id"]: i["coco_url"].split("/")[-2:] for i in d["images"]}
        for a in d["annotations"]:
            if a["category_id"] in cat_ids:
                x, y, w, h = a["bbox"]
                if w >= 1 and h >= 1:
                    folder, fn = img[a["image_id"]]
                    rows.append((a["id"], a["category_id"], str(LVIS / "extracted" / folder / fn), x, y, x + w, y + h))
        del d
    return pd.DataFrame(rows, columns=["ann_id", "category_id", "path", "xmin", "ymin", "xmax", "ymax"])


def main():
    os.environ.setdefault("HF_HOME", "/ccn2/u/khaiaw/.cache/huggingface")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    torch.set_num_threads(2)
    need = set(pd.read_csv(OUT / "lvis_categories_to_embed.csv").lvis_category_id.astype(int))
    m = manifest(need).sort_values(["path", "ann_id"]).reset_index(drop=True)
    names = {c["id"]: c["name"] for c in json.load(open(LVIS / "annotations" / "lvis_v1_val.json"))["categories"]}
    rng = np.random.default_rng(0)
    m["sample"] = False
    for c, g in m.groupby("category_id"):
        m.loc[rng.choice(g.index, min(SAMPLE_PER_CAT, len(g)), replace=False), "sample"] = True
    print(f"{len(m):,} crops in {m.category_id.nunique()} categories, {m.path.nunique():,} images", flush=True)
    dev = "cuda:0"
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    dp = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
    dino = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m", dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    cats = sorted(m.category_id.unique()); ci = {c: i for i, c in enumerate(cats)}
    S = {"clip": np.zeros((len(cats), 512)), "dinov3": np.zeros((len(cats), 768))}
    N = np.zeros(len(cats), np.int64)
    samp = {"ann_id": [], "category_id": [], "path": [], "bbox": [], "clip": [], "dinov3": []}

    def crops_of(path_group):
        path, g = path_group
        im = Image.open(path).convert("RGB")
        return [(r, im.crop((r.xmin, r.ymin, r.xmax, r.ymax))) for r in g.itertuples()]

    pool = ThreadPoolExecutor(6)
    buf = []
    def flush():
        rows, ims = zip(*buf)
        with torch.inference_mode():
            c = clip.get_image_features(**{k: v.to(dev) for k, v in cp(images=list(ims), return_tensors="pt").items()})
            c = (c.pooler_output if hasattr(c, "pooler_output") else c).float().cpu().numpy()
            d = dino(**{k: v.to(dev) for k, v in dp(images=list(ims), return_tensors="pt").items()}).pooler_output.float().cpu().numpy()
        for model, X in (("clip", c), ("dinov3", d)):
            Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            np.add.at(S[model], [ci[r.category_id] for r in rows], Xn)
        np.add.at(N, [ci[r.category_id] for r in rows], 1)
        for k, r in enumerate(rows):
            if r.sample:
                samp["ann_id"].append(r.ann_id); samp["category_id"].append(r.category_id); samp["path"].append(r.path)
                samp["bbox"].append((r.xmin, r.ymin, r.xmax, r.ymax)); samp["clip"].append(c[k]); samp["dinov3"].append(d[k])
        buf.clear()
    groups = list(m.groupby("path", sort=False))
    for i, crops in enumerate(pool.map(crops_of, groups)):
        buf.extend(crops)
        if len(buf) >= BATCH:
            flush()
        if i % 20000 == 0:
            print(f"  {i:,}/{len(groups):,} images, {int(N.sum()):,} crops", flush=True)
    if buf:
        flush()
    for model in S:
        np.savez(OUT / f"lvis_extra_category_embeddings_{model}.npz", category_ids=np.array(cats),
                 names=np.array([names[c] for c in cats], dtype=object), counts=N, sums=S[model])
    np.savez(OUT / "lvis_extra_crops_sample.npz", ann_id=np.array(samp["ann_id"]), category_id=np.array(samp["category_id"]),
             path=np.array(samp["path"]), bbox=np.array(samp["bbox"], np.float32),
             clip=np.array(samp["clip"], np.float16), dinov3=np.array(samp["dinov3"], np.float16))
    print(f"done: {int(N.sum()):,} crops, {len(cats)} categories")


if __name__ == "__main__":
    main()
