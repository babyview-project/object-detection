#!/usr/bin/env python3
"""Second pass on generic labels: what is the "toy" / "animal" / "character" ... in this box?

Each box whose category (qwen_data `cat`) is generic is cropped with context (box outlined in red) and
sent back to Qwen3.8-27B (locally) with Qwen's own label and format: name the specific thing it is or
depicts, "none" for a toy that does not represent anything, "unclear" if it cannot tell.

Output (private: unit ids name videos): out/private/qwen_generic_crops_<tag>.jsonl[.shardK-of-N]
  unit_id, obj, label, cat, format, name, error

  CUDA_VISIBLE_DEVICES=4 python qwen_generic_crops.py --shard 0/4      # one process per GPU, then --merge-shards 4

Full runs (--store bv / vedb): the box list comes from the embedding store's boxes_index.csv.gz (every embedded box,
repeat boxes already dropped), tag fullrun_<store>; obj = the box's obj_idx in its frame record.
  python qwen_generic_crops.py --prepare --store bv                     # analysis env
  CUDA_VISIBLE_DEVICES=0 python qwen_generic_crops.py --tag fullrun_bv --shard 0/8
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"
GENERIC = ["toy", "animal", "character", "figure", "vehicle", "food", "object", "item", "thing"]
PAD, SIDE = 0.25, 448
PROMPT = """This crop is from a young child's head-camera video. An annotator labelled the object outlined in red as "{label}" (format: {format}).
What specific thing is it, or what does it depict? Return JSON with "name": its everyday name, lowercase (for example "car", "dog", "giraffe", "block", "rattle", "stacking rings", "doll", "elmo"); "none" if it is a toy that does not represent anything specific; "unclear" if you cannot tell."""
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False}


def crop_url(path: str, y0, x0, y1, x1) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    bx0, bx1, by0, by1 = x0 / 1000 * w, x1 / 1000 * w, y0 / 1000 * h, y1 / 1000 * h
    side = max(bx1 - bx0, by1 - by0) * (1 + 2 * PAD)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    L, T = max(0, cx - side / 2), max(0, cy - side / 2)
    R, B = min(w, cx + side / 2), min(h, cy + side / 2)
    c = im.crop((round(L), round(T), round(R), round(B)))
    ImageDraw.Draw(c).rectangle([bx0 - L, by0 - T, bx1 - L, by1 - T], outline=(230, 30, 30), width=max(2, round(c.width / 150)))
    s = SIDE / max(c.size)
    if s < 1:
        c = c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    c.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def boxes(tag: str) -> pd.DataFrame:
    import qwen_data as qd
    o = qd.dedup(qd.objects_table(qd.load_records("detect_open_format", tag)))
    o = o[o.cat.isin(GENERIC)]
    m = pd.read_csv(HERE.parent / "gemini_full_run" / "out" / "private" / "manifest.csv", usecols=["unit_id", "frame_path"])
    return o.merge(m, on="unit_id")[["unit_id", "obj", "label", "cat", "format", "ymin", "xmin", "ymax", "xmax", "frame_path"]]


STORES = {"bv": ("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1", (512, 910)),
          "vedb": ("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv", (2048, 1536))}
BV_FRAMES = "/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps"


def store_boxes(store: str) -> pd.DataFrame:
    """Generic boxes of a full-run embedding store, in crop_url's 0-1000 coordinates."""
    import qwen_data as qd
    root, (W, H) = STORES[store]
    ix = pd.read_csv(Path(root) / "boxes_index.csv.gz",
                     usecols=["video_id", "frame_number", "obj_idx", "label", "format", "xmin", "ymin", "xmax", "ymax"])
    ix["label"] = ix.label.astype(str)
    nm = {l: qd.norm_label(l) for l in ix.label.unique()}
    ix["cat"] = ix.label.map(nm)
    b = ix[ix.cat.isin(GENERIC)].copy()
    if store == "bv":
        b["unit_id"] = b.video_id + "/" + b.frame_number.map("{:05d}".format)
        b["frame_path"] = BV_FRAMES + "/" + b.unit_id + ".jpg"
    else:
        m = pd.read_csv(HERE / "out" / "private" / "vedb_manifest.csv", usecols=["video_id", "frame", "unit_id", "frame_path"])
        b = b.merge(m, left_on=["video_id", "frame_number"], right_on=["video_id", "frame"])
    b["xmin"], b["xmax"] = b.xmin / W * 1000, b.xmax / W * 1000
    b["ymin"], b["ymax"] = b.ymin / H * 1000, b.ymax / H * 1000
    return b.rename(columns={"obj_idx": "obj"})[["unit_id", "obj", "label", "cat", "format", "ymin", "xmin", "ymax", "xmax", "frame_path"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="subj_S00320001_t1_pp0")
    ap.add_argument("--prepare", action="store_true", help="write the box list (analysis env, needs WordNet)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge-shards", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--store", choices=list(STORES), help="--prepare from a full-run embedding store (tag fullrun_<store>)")
    ap.add_argument("--threads", type=int, default=8, help="crop-building threads")
    args = ap.parse_args()
    if args.store:
        args.tag = f"fullrun_{args.store}"
    lst = PRIV / f"qwen_generic_boxes_{args.tag}.csv.gz"
    out = PRIV / f"qwen_generic_crops_{args.tag}.jsonl"
    if args.prepare:
        b = store_boxes(args.store) if args.store else boxes(args.tag)
        b.to_csv(lst, index=False)
        print(f"{len(b):,} generic boxes:", b.cat.value_counts().to_dict())
        return
    if args.merge_shards:
        parts = sorted(PRIV.glob(f"{out.name}.shard*-of-{args.merge_shards}"))
        with open(out, "a") as f:
            for p in parts:
                f.write(p.read_text())
        print(f"merged {len(parts)} shards -> {out}")
        return
    k, n = map(int, args.shard.split("/"))
    b = pd.read_csv(lst)
    b = b[b.index % n == k].reset_index(drop=True)
    sout = out.with_name(f"{out.name}.shard{k}-of-{n}")
    done = set()
    if sout.exists():
        done = {(r["unit_id"], r["obj"]) for r in map(json.loads, open(sout))}
    b = b[[(u, o) not in done for u, o in zip(b.unit_id, b.obj)]].reset_index(drop=True)
    print(f"{sout.name}: {len(b):,} to do ({len(done):,} done)", flush=True)

    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from run_qwen_pilot import make_llm
    llm = make_llm(SimpleNamespace(mtp=0, tp=1, max_model_len=4096, gpu_mem=0.9, max_num_seqs=128, enforce_eager=False))
    sp = SamplingParams(temperature=0.0, max_tokens=32, seed=0, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    from concurrent.futures import ThreadPoolExecutor
    pool, pre = ThreadPoolExecutor(args.threads), ThreadPoolExecutor(1)

    def build(ch):                       # crops in parallel threads; the next batch is built while the GPU runs
        urls = list(pool.map(lambda r: crop_url(r.frame_path, r.ymin, r.xmin, r.ymax, r.xmax), list(ch.itertuples())))
        return [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                              {"type": "text", "text": PROMPT.format(label=r.label, format=r.format)}]}]
                for u, r in zip(urls, ch.itertuples())]
    starts = list(range(0, len(b), args.batch))
    nxt = pre.submit(build, b.iloc[0:args.batch]) if starts else None
    for i, s in enumerate(starts):
        ch = b.iloc[s:s + args.batch]
        msgs = nxt.result()
        if i + 1 < len(starts):
            nxt = pre.submit(build, b.iloc[starts[i + 1]:starts[i + 1] + args.batch])
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
        with open(sout, "a") as f:
            for r, o in zip(ch.itertuples(), outs):
                try:
                    name, err = str(json.loads(o.outputs[0].text)["name"]).strip().lower(), None
                except Exception as e:
                    name, err = None, repr(e)[:200]
                f.write(json.dumps({"unit_id": r.unit_id, "obj": int(r.obj), "label": r.label, "cat": r.cat,
                                    "format": r.format, "name": name, "error": err}) + "\n")
        print(f"  {min(s + args.batch, len(b)):,}/{len(b):,}", flush=True)


if __name__ == "__main__":
    main()
