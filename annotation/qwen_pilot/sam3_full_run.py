#!/usr/bin/env python3
"""SAM 3 check of every Qwen box (full frame, label-aware), BabyView and VEDB full runs.

Per frame: the image is encoded once (get_vision_features) and SAM 3 is prompted with each distinct label on the frame
(the renamed label, make_renamed_index.py; text embeddings computed once per label for the whole run). A box's score is
the highest score among its label's SAM 3 instances that overlap it with IoU >= 0.5 (0 if none); the matched instance's
box is stored as a corrected box. On the RA gold set this score separates boxes the RAs deleted from kept ones with
AUC 0.86 (sam3_box_check.py, frame mode). Runs locally (Meta SAM License: acknowledge SAM in publications).

Output: <store>/sam3/<video_id>.csv.gz  pos (row within the video in boxes_index), sam3_score, sam3_iou, sam3_n,
        sam3_x0, sam3_y0, sam3_x1, sam3_y1 (px, frame coordinates); resumable per video.
  CUDA_VISIBLE_DEVICES=0 python sam3_full_run.py --store bv --worker 0/8       # qwen env, one per GPU
  CUDA_VISIBLE_DEVICES=0 python sam3_full_run.py --gold                        # check against sam3_box_check_frame.csv
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL, CACHE = "facebook/sam3", "/data/j7yang/models/hf/hub"
STORES = {"bv": Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1"),
          "vedb": Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv")}
BV_FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")
VEDB_FRAMES = Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps")
MATCH_IOU, MIN_SCORE, LABEL_BATCH = 0.5, 0.05, 32


def iou_m(A, B):
    A, B = np.asarray(A, float).reshape(-1, 4), np.asarray(B, float).reshape(-1, 4)
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    i = ix * iy; ar = lambda X: (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    return i / np.maximum(ar(A)[:, None] + ar(B)[None, :] - i, 1e-9)


class Scorer:
    def __init__(self, dev="cuda:0"):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch, self.dev = torch, dev
        self.proc = Sam3Processor.from_pretrained(MODEL, cache_dir=CACHE)
        self.model = Sam3Model.from_pretrained(MODEL, cache_dir=CACHE, dtype=torch.bfloat16).to(dev).eval()
        self.text = {}

    def text_embeds(self, labels):
        """(input_ids, attention_mask, text_embeds) for labels, cached per label."""
        torch = self.torch
        new = [l for l in labels if l not in self.text]                # tokenized once per label (text encoder is cheap)
        if new:
            t = self.proc(text=new, return_tensors="pt", padding="max_length").to(self.dev)
            for k, l in enumerate(new):
                self.text[l] = (t["input_ids"][k], t["attention_mask"][k], None)
        ids = torch.stack([self.text[l][0] for l in labels]); am = torch.stack([self.text[l][1] for l in labels])
        return ids, am, None

    @staticmethod
    def _repeat(x, n):
        """Repeat every tensor in a (nested) model output n times along the batch dimension."""
        import torch
        if isinstance(x, torch.Tensor):
            return x.expand(n, *x.shape[1:]) if x.shape[0] == 1 else x
        if isinstance(x, (list, tuple)):
            return type(x)(Scorer._repeat(v, n) for v in x)
        if hasattr(x, "keys"):                      # ModelOutput (a dict subclass) or dict: keep the type
            return type(x)(**{k: Scorer._repeat(v, n) for k, v in x.items()})
        return x

    def score_frame(self, im, boxes, labels):
        """boxes (n, 4) px, labels list[str] -> (score, iou, n_instances, matched box) per box."""
        torch = self.torch
        uniq = sorted(set(labels))
        pix = self.proc(images=im, return_tensors="pt")
        size = pix["original_sizes"] if "original_sizes" in pix else torch.tensor([im.size[::-1]])
        with torch.inference_mode():
            vis = self.model.get_vision_features(pixel_values=pix["pixel_values"].to(self.dev, torch.bfloat16))
        inst = {}
        for s in range(0, len(uniq), LABEL_BATCH):
            labs = uniq[s:s + LABEL_BATCH]
            ids, am, te = self.text_embeds(labs)
            with torch.inference_mode():
                out = self.model(vision_embeds=self._repeat(vis, len(labs)), input_ids=ids, attention_mask=am)
            res = self.proc.post_process_instance_segmentation(out, threshold=MIN_SCORE, mask_threshold=0.5,
                                                               target_sizes=[im.size[::-1]] * len(labs))
            for l, r in zip(labs, res):
                inst[l] = (r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy())
        outv = []
        for bx, l in zip(boxes, labels):
            b, sc = inst[l]
            ious = iou_m([bx], b)[0] if len(b) else np.array([])
            ok = ious >= MATCH_IOU
            if ok.any():
                j = int(np.argmax(np.where(ok, sc, -1)))
                outv.append((float(sc[j]), float(ious[j]), len(b), b[j]))
            else:
                outv.append((0.0, float(ious.max()) if len(ious) else 0.0, len(b), [np.nan] * 4))
        return outv


def frame_path(store, video_id, frame):
    return (VEDB_FRAMES / video_id / f"{frame:08d}.jpg") if store == "vedb" else (BV_FRAMES / video_id / f"{frame:05d}.jpg")


def gold_check():
    from PIL import Image
    gold = HERE.parent / "gold_set" / "out" / "private"
    b = pd.read_csv(gold / "ra_scores" / "boxes.csv"); q = b[b.outcome != "added"]
    fr = pd.read_csv(gold / "frames_qwen_t1.csv").drop_duplicates("frame_uid").set_index("frame_uid").frame_path
    ref = pd.read_csv(gold / "ra_scores" / "sam3_box_check_frame.csv").set_index("row").sam3_score
    sc = Scorer(); rows = []; t0 = time.time()
    for n, (uid, g) in enumerate(q.groupby("frame_uid")):
        im = Image.open(fr[uid]).convert("RGB")
        labs = [str(x) if isinstance(x, str) and x else "object" for x in g.q_name]
        for idx, (s, i, k, _) in zip(g.index, sc.score_frame(im, g[["q_x0", "q_y0", "q_x1", "q_y1"]].values, labs)):
            rows.append((idx, s))
    new = pd.Series(dict(rows))
    d = (new - ref.reindex(new.index)).abs()
    print(f"{len(new)} boxes in {time.time() - t0:.0f} s ({(time.time() - t0) / q.frame_uid.nunique():.2f} s/frame); "
          f"|new - reference| median {d.median():.4f}, 95th pct {d.quantile(.95):.4f}, max {d.max():.4f}; "
          f"rank correlation {new.rank().corr(ref.reindex(new.index).rank()):.4f}")


def worker(store, k, n):
    from PIL import Image
    root = STORES[store]; out = root / "sam3"; out.mkdir(exist_ok=True)
    ix = pd.read_csv(root / "boxes_index_renamed.csv.gz", usecols=["video_id", "frame_number", "label_renamed", "xmin", "ymin", "xmax", "ymax"])
    ix["pos"] = ix.groupby("video_id").cumcount()
    vids = sorted(ix.video_id.unique())[k::n]
    todo = [v for v in vids if not (out / f"{v}.csv.gz").exists()]
    print(f"{store} worker {k}/{n}: {len(vids)} videos, {len(todo)} to do", flush=True)
    sc = Scorer(); pool = ThreadPoolExecutor(4); t0, nb = time.time(), 0
    by = dict(tuple(ix[ix.video_id.isin(set(todo))].groupby("video_id")))
    for vi, v in enumerate(todo):
        g = by[v]; frames = list(g.groupby("frame_number"))
        ims = pool.map(lambda f: Image.open(frame_path(store, v, int(f[0]))).convert("RGB"), frames)
        rows = []
        for (fnum, gg), im in zip(frames, ims):
            labs = [str(x) if isinstance(x, str) and x else "object" for x in gg.label_renamed]
            for p, (s, i, m, bb) in zip(gg.pos, sc.score_frame(im, gg[["xmin", "ymin", "xmax", "ymax"]].values, labs)):
                rows.append((p, s, i, m, *bb))
        tmp = out / f"{v}.tmp.csv.gz"
        pd.DataFrame(rows, columns=["pos", "sam3_score", "sam3_iou", "sam3_n", "sam3_x0", "sam3_y0", "sam3_x1", "sam3_y1"]).round(4).to_csv(tmp, index=False)
        os.replace(tmp, out / f"{v}.csv.gz")
        nb += len(rows)
        if vi % 10 == 0:
            el = time.time() - t0
            print(f"  {vi + 1}/{len(todo)} videos, {nb:,} boxes, {nb / el:.1f} boxes/s, {el / 3600:.1f} h", flush=True)
    print(f"{store} worker {k}/{n} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", choices=list(STORES))
    ap.add_argument("--worker", default="0/1")
    ap.add_argument("--gold", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    torch.set_num_threads(2)
    if args.gold:
        gold_check(); return
    k, n = map(int, args.worker.split("/"))
    worker(args.store, k, n)


if __name__ == "__main__":
    main()
