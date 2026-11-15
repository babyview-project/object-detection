#!/usr/bin/env python3
"""What kind of 2D depiction is it? Fine-grained format pass on Qwen's "drawing", "photo" and "screen" boxes (BabyView).

Each box is cropped with context (box outlined in red) and sent back to Qwen3.8-27B (locally), which answers two things:
  surface: paper or card (books, cards, posters, framed prints) / screen (TV, phone, tablet, computer) / drawn or painted on
           a surface (chalk, wall mural, whiteboard) / fabric or object surface (clothing, blankets, packaging, toy boxes) /
           not a picture (a real object or a toy) / unclear
  style:   line drawing / colour illustration or animation / painting / photograph or live action / text or interface /
           none (not a picture) / unclear
The prompt names neither Qwen's format nor its label (the --tag sample pilot did, which anchored the answer; that pilot
used a single style field, PROMPT_V1).

Output (private: unit ids name videos): out/private/qwen_picture_style_<tag>.jsonl[.shardK-of-N]
  python qwen_picture_style.py --prepare --n 0 --tag all2d                # analysis env: every drawing/photo/screen box >= 48 px
  CUDA_VISIBLE_DEVICES=2 python qwen_picture_style.py --tag all2d --shard 0/8   # qwen venv, one process per GPU
  python qwen_picture_style.py --tag all2d --merge-shards 8
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
FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")
SURFACES = ["paper or card", "screen", "drawn or painted on a surface", "fabric or object surface", "not a picture", "unclear"]
STYLES = ["line drawing", "colour illustration or animation", "painting", "photograph or live action", "text or interface",
          "none", "unclear"]
FORMATS = ["drawing", "photo", "screen"]
PAD, SIDE = 0.25, 448
PROMPT = """This crop is from a young child's head-camera video. Describe the object outlined in red as a picture.
Return JSON with two fields.
"surface", where the picture is:
  "paper or card" (book, board book, card, poster, framed print, puzzle piece),
  "screen" (TV, phone, tablet, computer, any lit display),
  "drawn or painted on a surface" (chalk on pavement, wall mural, whiteboard or chalkboard),
  "fabric or object surface" (printed on clothing, a blanket, packaging, a box or a toy),
  "not a picture" (a real object or a 3D toy),
  "unclear".
"style", what the picture looks like:
  "line drawing" (outlines or a sketch, little or no colour fill: colouring pages, pencil, pen, chalk or crayon lines),
  "colour illustration or animation" (coloured drawn or graphic images: cartoons, picture-book art, stickers, animated video),
  "painting" (painted artwork with visible brush strokes or texture),
  "photograph or live action" (a photo or live-action video of real scenes, people, animals or objects),
  "text or interface" (mostly writing, numbers, an app or menu screen),
  "none" (not a picture),
  "unclear"."""
PROMPT_V1 = """This crop is from a young child's head-camera video. The object outlined in red was labelled a picture (format: {format}), named "{label}".
What kind of picture is it? (pilot sample only, single "style" field: line drawing / colour illustration / painting /
photograph / not a picture / unclear; the format and name in the prompt anchored the answer)"""
SCHEMA = {"type": "object", "properties": {"surface": {"type": "string", "enum": SURFACES},
                                           "style": {"type": "string", "enum": STYLES}},
          "required": ["surface", "style"], "additionalProperties": False}


def crop_url(path, x0, y0, x1, y1) -> str:
    """Pixel box -> padded square-ish crop with the box outlined in red, longest side <= SIDE, as a data URL."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L, T = max(0, cx - side / 2), max(0, cy - side / 2)
    R, B = min(w, cx + side / 2), min(h, cy + side / 2)
    c = im.crop((round(L), round(T), round(R), round(B)))
    ImageDraw.Draw(c).rectangle([x0 - L, y0 - T, x1 - L, y1 - T], outline=(230, 30, 30), width=max(2, round(c.width / 150)))
    s = SIDE / max(c.size)
    if s < 1:
        c = c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    c.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def prepare(n: int, tag: str):
    import sys
    import numpy as np
    sys.path.insert(0, str(HERE.parents[1] / "analysis" / "qwen-open-vocab"))
    import variant as V
    store = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1")
    ix = V.read_index(store, usecols=["video_id", "frame_number", "label", "format", "xmin", "ymin", "xmax", "ymax"])
    ix = ix[ix.format.isin(FORMATS) & ((ix.xmax - ix.xmin) >= 48) & ((ix.ymax - ix.ymin) >= 48)]
    out = []
    for f, g in ix.groupby("format"):
        g = g.sample(frac=1, random_state=0).copy()
        g["subj"] = g.video_id.str[:9]
        g["v_rank"] = g.groupby("video_id").cumcount(); g = g.sort_values("v_rank", kind="stable")
        g["s_rank"] = g.groupby("subj").cumcount()
        g = g.sort_values(["s_rank", "v_rank"], kind="stable")
        out.append(g.head(n) if n else g)
    b = pd.concat(out).drop(columns=["v_rank", "s_rank"]).reset_index(drop=True)
    b["frame_path"] = [str(FRAMES / v / f"{int(k):05d}.jpg") for v, k in zip(b.video_id, b.frame_number)]
    b["uid"] = np.arange(len(b))
    b.to_csv(PRIV / f"qwen_picture_style_boxes_{tag}.csv.gz", index=False)
    print(f"{len(b):,} boxes:", b.format.value_counts().to_dict(), "| subjects", b.subj.nunique(), "| videos", b.video_id.nunique())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="sample")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--n", type=int, default=2000, help="boxes per format for --prepare")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge-shards", type=int, default=0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    lst = PRIV / f"qwen_picture_style_boxes_{args.tag}.csv.gz"
    out = PRIV / f"qwen_picture_style_{args.tag}.jsonl"
    if args.prepare:
        prepare(args.n, args.tag); return
    if args.merge_shards:
        parts = sorted(PRIV.glob(f"{out.name}.shard*-of-{args.merge_shards}"))
        out.write_text("".join(p.read_text() for p in parts))
        print(f"merged {len(parts)} shards -> {out}"); return
    k, n = map(int, args.shard.split("/"))
    b = pd.read_csv(lst)
    b = b[b.uid % n == k].reset_index(drop=True)
    sout = out.with_name(f"{out.name}.shard{k}-of-{n}")
    done = {r["uid"] for r in map(json.loads, open(sout))} if sout.exists() else set()
    b = b[~b.uid.isin(done)].reset_index(drop=True)
    print(f"{sout.name}: {len(b):,} to do ({len(done):,} done)", flush=True)

    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from run_qwen_pilot import make_llm
    llm = make_llm(SimpleNamespace(mtp=0, tp=1, max_model_len=4096, gpu_mem=0.9, max_num_seqs=128, enforce_eager=False))
    sp = SamplingParams(temperature=0.0, max_tokens=64, seed=0, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(args.threads)
    for s in range(0, len(b), args.batch):
        ch = b.iloc[s:s + args.batch]
        urls = list(pool.map(lambda r: crop_url(r.frame_path, r.xmin, r.ymin, r.xmax, r.ymax), list(ch.itertuples())))
        msgs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}},
                                              {"type": "text", "text": PROMPT}]}]
                for u, r in zip(urls, ch.itertuples())]
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
        with open(sout, "a") as f:
            for r, o in zip(ch.itertuples(), outs):
                try:
                    j = json.loads(o.outputs[0].text); su, st, err = j["surface"], j["style"], None
                except Exception as e:
                    su, st, err = None, None, repr(e)[:200]
                f.write(json.dumps({"uid": int(r.uid), "format": r.format, "label": r.label, "surface": su, "style": st,
                                    "error": err}) + "\n")
        print(f"  {min(s + args.batch, len(b)):,}/{len(b):,}", flush=True)


if __name__ == "__main__":
    main()
