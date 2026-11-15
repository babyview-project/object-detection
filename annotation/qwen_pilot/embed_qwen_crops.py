#!/usr/bin/env python3
"""CLIP and DINOv3 embeddings of Qwen's boxes, averaged per Qwen category (manuscript recipe).

Same models and readout as the 2025.2 YOLOE + CLIP crops (image-embedding/run_cdi_crop_pipeline.py):
openai/clip-vit-base-patch32 get_image_features (512-d) and facebook/dinov3-vitb16-pretrain-lvd1689m
pooler_output (768-d). Category means as in exemplar_set_zscore_embeddings.py: L2-normalize each crop,
average per category, z-score each dimension across categories.

Boxes: Qwen's own categories (qwen_data `cat`: Qwen's label with plurals / exact synonyms / depiction
words normalized) seen in >= --min-frames frames, near-duplicates dropped, person excluded, sides
>= 8 px, at most --max-per-cat boxes per category (random; one draw per frame first). The format is
recorded, so means can be split by format.

  # 1) box table (needs nltk WordNet: run in the analysis env, e.g. vislearnlabpy)
  python embed_qwen_crops.py --prepare
  # 2) GPU pass (one A40, ~3 GB; reads the frame JPEGs; qwen env)
  python embed_qwen_crops.py --device cuda:0
  # CPU only: this child's YOLOE + CLIP category means from the packed 2025.2 store, for comparison
  python embed_qwen_crops.py --yoloe-only

Outputs (out/private/, private: unit ids name videos):
  qwen_boxes_<tag>.csv.gz                               boxes to embed (from --prepare)
  qwen_crops_<tag>.npz                                  per crop: unit_id, label, cat, domain, format, area, clip, dino
  qwen_<model>_exemplar_avg_{raw,zscore}_<tag>_{all,real}.csv     Qwen category x dim (+ n_crops column in raw)
  yoloe_<model>_exemplar_avg_{raw,zscore}_S<subject>.csv          same child, YOLOE + CLIP crops (CLIP >= 0.27)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OD_ROOT = HERE.parents[1]
OUT = HERE / "out" / "private"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "archive" / "gemini_pilot"))
sys.path.insert(0, str(OD_ROOT / "analysis" / "manuscript-2026" / "scripts"))

CLIP_MODEL = "openai/clip-vit-base-patch32"
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
HF_HOME_2025_2 = "/ccn2/u/khaiaw/.cache/huggingface"          # where the 2025.2 run cached both models
MIN_SIDE = 8


def valid129() -> list[str]:
    return [l.strip() for l in open(OD_ROOT / "data" / "included_categories_valid129.txt") if l.strip()]


def category_means(cats: np.ndarray, X: np.ndarray, keep: list[str]):
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    names = [c for c in keep if (cats == c).any()]
    M = np.stack([X[cats == c].mean(axis=0) for c in names])
    n = [int((cats == c).sum()) for c in names]
    return names, M, n


def save(path: Path, names, M, n=None):
    d = pd.DataFrame(M, columns=[f"dim_{i}" for i in range(M.shape[1])])
    d.insert(0, "category", names)
    if n is not None:
        d["n_crops"] = n
    d.to_csv(path, index=False)


def zscore_rows(X, eps=1e-10):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + eps)


def write_means(out: Path, prefix: str, suffix: str, cats, X, keep):
    names, M, n = category_means(cats, X, keep)
    save(out / f"{prefix}_exemplar_avg_raw_{suffix}.csv", names, M, n)
    save(out / f"{prefix}_exemplar_avg_zscore_{suffix}.csv", names, zscore_rows(M))
    return len(names)


def prepare(args) -> None:
    import qwen_data as qd
    o = qd.dedup(qd.objects_table(qd.load_records(args.task, args.tag)))
    o = o[o.cat != "person"]
    nfr = o.groupby("cat").unit_id.nunique()
    o = o[o.cat.isin(nfr[nfr >= args.min_frames].index)]
    o = o.sample(frac=1, random_state=0)
    o["_first"] = ~o.duplicated(["cat", "unit_id"])          # spread over frames before taking repeats
    o = o.sort_values("_first", ascending=False, kind="stable").groupby("cat").head(args.max_per_cat).drop(columns="_first")
    m = pd.read_csv(HERE.parent / "gemini_full_run" / "out" / "private" / "manifest.csv", usecols=["unit_id", "frame_path"])
    o = o.merge(m, on="unit_id").sort_values(["unit_id", "obj"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    o[["unit_id", "obj", "label", "cat", "domain", "format", "area", "ymin", "xmin", "ymax", "xmax", "frame_path"]] \
        .to_csv(OUT / f"qwen_boxes_{args.tag}.csv.gz", index=False)
    print(f"{len(o):,} boxes, {o.cat.nunique()} categories (>= {args.min_frames} frames, <= {args.max_per_cat} each), "
          f"{o.unit_id.nunique():,} frames")


def embed(args):
    os.environ.setdefault("HF_HOME", HF_HOME_2025_2)
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    torch.set_num_threads(1)

    b = pd.read_csv(OUT / f"qwen_boxes_{args.tag}.csv.gz")
    out = OUT
    if args.limit:                                   # quick check: first N frames, written to out/private/check/
        b = b[b.unit_id.isin(b.unit_id.drop_duplicates().head(args.limit))]
        out = OUT / "check"
    print(f"{len(b):,} boxes, {b.cat.nunique()} categories, {b.unit_id.nunique():,} frames", flush=True)
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()
    dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModel.from_pretrained(DINO_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(args.device).eval()

    def run(crops):
        with torch.inference_mode():
            ci = {k: v.to(args.device) for k, v in clip_proc(images=crops, return_tensors="pt").items()}
            c = clip.get_image_features(**ci)
            c = c.pooler_output if hasattr(c, "pooler_output") else c      # as _unwrap_embeds in the 2025.2 run
            di = {k: v.to(args.device) for k, v in dino_proc(images=crops, return_tensors="pt").items()}
            d = dino(**di).pooler_output
        return c.float().cpu().numpy().astype(np.float16), d.float().cpu().numpy().astype(np.float16)

    rows, C, D, crops, pend = [], [], [], [], []
    t0 = time.time()
    for k, (u, g) in enumerate(b.groupby("unit_id", sort=False)):
        im = np.asarray(Image.open(g.frame_path.iloc[0]).convert("RGB"))
        h, w = im.shape[:2]
        for r in g.itertuples():
            x0, x1 = int(r.xmin / 1000 * w), int(np.ceil(r.xmax / 1000 * w))
            y0, y1 = int(r.ymin / 1000 * h), int(np.ceil(r.ymax / 1000 * h))
            if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
                continue
            crops.append(im[y0:y1, x0:x1])
            pend.append((u, r.obj, r.label, r.cat, r.domain, r.format, r.area))
        if len(crops) >= args.batch:
            c, d = run(crops); C.append(c); D.append(d); rows += pend; crops, pend = [], []
        if k % 2000 == 0:
            print(f"  {k:,} frames, {len(rows):,} crops, {time.time() - t0:.0f}s", flush=True)
    if crops:
        c, d = run(crops); C.append(c); D.append(d); rows += pend
    meta = pd.DataFrame(rows, columns=["unit_id", "obj", "label", "cat", "domain", "format", "area"])
    C, D = np.concatenate(C), np.concatenate(D)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"qwen_crops_{args.tag}.npz", clip=C, dino=D,
             **{k: meta[k].astype(str).values for k in ("unit_id", "label", "cat", "domain", "format")},
             obj=meta.obj.values, area=meta.area.values)
    cats, real = meta.cat.values, (meta.format == "real").values
    order = list(meta.cat.value_counts().index)
    for model, X in (("clip", C), ("dinov3", D)):
        X = X.astype(np.float64)
        n_all = write_means(out, f"qwen_{model}", f"{args.tag}_all", cats, X, order)
        n_real = write_means(out, f"qwen_{model}", f"{args.tag}_real", cats[real], X[real], order)
        print(f"{model}: {len(X):,} crops; {n_all} categories (all formats), {n_real} (real only)")
    print(f"done in {time.time() - t0:.0f}s")


def yoloe(args):
    from packed_crop_embeddings import DEFAULT_PACKED_ROOT, subject_category_means
    sid = args.subject.lstrip("S")
    keep = set(valid129()) - {"person", "picture"}
    for model in ("clip", "dinov3"):
        means = subject_category_means(DEFAULT_PACKED_ROOT, keep, model=model, subjects={sid}, min_sim=0.27)[sid]
        names = [c for c in valid129() if c in means]
        M = np.stack([means[c] for c in names])
        save(OUT / f"yoloe_{model}_exemplar_avg_raw_S{sid}.csv", names, M)
        save(OUT / f"yoloe_{model}_exemplar_avg_zscore_S{sid}.csv", names, zscore_rows(M))
        print(f"YOLOE + CLIP {model}: {len(names)} categories for S{sid}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="detect_open_format")
    ap.add_argument("--tag", default="subj_S00320001_t1_pp0")
    ap.add_argument("--subject", default="S00320001")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (check run, out/private/check/)")
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--max-per-cat", type=int, default=2000)
    ap.add_argument("--prepare", action="store_true", help="write the box table (analysis env, needs WordNet)")
    ap.add_argument("--yoloe-only", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare(args)
    elif args.yoloe_only:
        yoloe(args)
    else:
        embed(args)


if __name__ == "__main__":
    main()
