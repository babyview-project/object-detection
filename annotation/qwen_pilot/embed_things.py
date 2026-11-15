#!/usr/bin/env python3
"""CLIP + DINOv3 embeddings of every THINGS image (1,854 concepts, 26,107 images), as the manuscript embeds THINGS.

Same models and readout as the BabyView crops (CLIP ViT-B/32 get_image_features 512-d, DINOv3 ViT-B/16 pooler_output
768-d; whole image through each model's processor). Concept means as the manuscript's THINGS tables: L2-normalize
each image's vector, average per concept (z-score across the categories compared happens at analysis time).

Output (THINGS_ROOT/embeddings/, THINGS_ROOT = /data2/dataset/babyview/related_data/things):
  things_images_{clip,dinov3}.npz    concept, path, emb (float16, n_images x dim)
  things_{clip,dinov3}_concept_mean_raw.csv   concept x dim (mean of L2-normalized image vectors) + n_images

  CUDA_VISIBLE_DEVICES=6 python embed_things.py [--batch 64]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

THINGS_ROOT = Path("/data2/dataset/babyview/related_data/things")
CLIP_MODEL = "openai/clip-vit-base-patch32"
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/ccn2/u/khaiaw/.cache/huggingface")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    torch.set_num_threads(2)
    paths = sorted((THINGS_ROOT / "object_images").glob("*/*.jpg"))
    concepts = [p.parent.name for p in paths]
    print(f"{len(paths):,} images, {len(set(concepts)):,} concepts", flush=True)
    dev = "cuda:0"
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModel.from_pretrained(DINO_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    C, D = [], []
    for s in range(0, len(paths), args.batch):
        ims = [Image.open(p).convert("RGB") for p in paths[s:s + args.batch]]
        with torch.inference_mode():
            c = clip.get_image_features(**{k: v.to(dev) for k, v in clip_proc(images=ims, return_tensors="pt").items()})
            c = c.pooler_output if hasattr(c, "pooler_output") else c
            d = dino(**{k: v.to(dev) for k, v in dino_proc(images=ims, return_tensors="pt").items()}).pooler_output
        C.append(c.float().cpu().numpy()); D.append(d.float().cpu().numpy())
        if (s // args.batch) % 50 == 0:
            print(f"  {s + len(ims):,}/{len(paths):,}", flush=True)
    out = THINGS_ROOT / "embeddings"
    out.mkdir(exist_ok=True)
    for model, X in (("clip", np.concatenate(C)), ("dinov3", np.concatenate(D))):
        np.savez(out / f"things_images_{model}.npz", concept=np.array(concepts), path=np.array([str(p) for p in paths]),
                 emb=X.astype(np.float16))
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        df = pd.DataFrame(Xn, columns=[f"dim_{i}" for i in range(X.shape[1])])
        df.insert(0, "concept", concepts)
        m = df.groupby("concept").mean()
        m["n_images"] = df.groupby("concept").size()
        m.reset_index().to_csv(out / f"things_{model}_concept_mean_raw.csv", index=False)
        print(f"{model}: {X.shape} -> {out}", flush=True)


if __name__ == "__main__":
    main()
