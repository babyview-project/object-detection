#!/usr/bin/env python3
"""Qwen3.8-27B (open weights, run locally with vLLM) on the Gemini pilot's frame sets.

Same tasks and prompts as gemini_pilot/run_gold_pilot.py, so results compare directly:
  detect              CDI word list (295), every visible listed object with a box
  detect_open_format  setup B of the full Gemini run: open label + format + box
Only the box line differs: Qwen-VL models are trained to give "bbox_2d": [x1, y1, x2, y2] on
0-1000, so the prompt asks for that (--box-prompt native, default) and the output is
converted to Gemini's "box_2d": [ymin, xmin, ymax, xmax]. --box-prompt gemini sends the
Gemini prompt unchanged.

Records go to ../gemini_pilot/out/private/<model>__<task>__p1_<frames>[_<sampling>].jsonl in
the Gemini pilot's format (raw = converted JSON, raw_model = Qwen's own text), so
score_gold_pilot.py, score_depiction.py, compare_boxes_montage.py etc. read them unchanged.

Frame sets (--frames):
  gold        5,919 frames of the valid5921 gold crops (gemini_pilot/out/private/gold_frames.csv)
  montage72   the 72 frames of the box-comparison pages
  depiction   80 frames with human format tags (gemini_pilot/out/private/depiction_frames.csv)
  gold_v2     1,000 gold set v2 frames (gold_set/out/private/gold_v2_frames.csv)
  fullrun500  the 500-frame batch pilot of the full Gemini run

Sampling (--sampling): greedy = temperature 0 (as Gemini); qwen = Qwen's recommended
non-thinking settings (temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5).
Thinking is off (enable_thinking=False). Frames go in at native 512x910. Data never leave
the machine. Runs only with GPUs: see run_qwen_pilot.sh for picking free ones.

  python run_qwen_pilot.py --task detect_open_format --frames depiction --dry-run
  CUDA_VISIBLE_DEVICES=0 python run_qwen_pilot.py --task detect_open_format --frames depiction
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "gemini_pilot"
sys.path.insert(0, str(PILOT))
import run_gold_pilot as gp  # noqa: E402  (prompts, vocabulary, frame lists)

MODEL_DIR = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.8-27B-FP8")
MODEL_TAG = "qwen3.8-27b-fp8"
OUT = PILOT / "out" / "private"
MAX_PX = 910
GEMINI_BOX_LINES = [
    '- "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000.',
    '"box_2d", [ymin, xmin, ymax, xmax] normalized to 0-1000.',
]
NATIVE_BOX_LINES = [
    '- "bbox_2d": [x1, y1, x2, y2], the box corners normalized to 0-1000.',
    '"bbox_2d", [x1, y1, x2, y2], the box corners normalized to 0-1000.',
]
SAMPLING = {"greedy": dict(temperature=0.0),
            "qwen": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)}


def frame_set(name: str) -> pd.DataFrame:
    if name == "gold":
        return gp.load_frames()
    if name == "montage72":
        keep = {l.strip() for l in open(OUT / "montage72.txt") if l.strip()}
        f = gp.load_frames()
        return f[f.unit_id.isin(keep)].reset_index(drop=True)
    if name == "depiction":
        return gp.load_frames(OUT / "depiction_frames.csv")
    if name == "gold_v2":
        return gp.load_frames(HERE.parent / "gold_set" / "out" / "private" / "gold_v2_frames.csv")
    if name == "fullrun500":
        fr = HERE.parent / "gemini_full_run" / "out" / "private"
        units = [json.loads(l)["unit_id"] for l in open(fr / "pilot_results" / "results_pilot.jsonl")]
        m = pd.read_csv(fr / "manifest.csv")
        m = m[m.unit_id.isin(units)].rename(columns={"video_id": "video"})
        m["frame_id"] = m.frame.astype(str).str.zfill(5)
        m["category"] = ""
        return m[["unit_id", "video", "frame_id", "frame_path", "category"]].reset_index(drop=True)
    if name.startswith("subject:"):                  # every 5th 1 fps frame of one child (full-run manifest)
        subj = name.split(":", 1)[1]
        m = pd.read_csv(HERE.parent / "gemini_full_run" / "out" / "private" / "manifest.csv")
        m = m[m.video_id.str.startswith(subj + "_")].rename(columns={"video_id": "video"})
        m["frame_id"] = m.frame.astype(str).str.zfill(5)
        m["category"] = ""
        return m[["unit_id", "video", "frame_id", "frame_path", "category"]].sort_values("unit_id").reset_index(drop=True)
    raise SystemExit(f"unknown frame set {name}")


def prompt_for(task: str, labels: list[str], box_prompt: str, max_obj: int) -> str:
    gp.MAX_OBJ = max_obj
    p = gp.prompt_for(task, None, labels)
    if box_prompt == "native":
        n = 0
        for g, q in zip(GEMINI_BOX_LINES, NATIVE_BOX_LINES):
            n += p.count(g)
            p = p.replace(g, q)
        assert n == 1, f"box line not found once in the {task} prompt"
    return p


def schema(task: str, box_prompt: str, max_obj: int) -> dict:
    key = "bbox_2d" if box_prompt == "native" else "box_2d"
    props = {"label": {"type": "string"}}
    order = ["label"]
    if task == "detect_open_format":
        props["format"] = {"type": "string", "enum": gp.FORMATS}
        order.append("format")
    props[key] = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
    order.append(key)
    obj = {"type": "object", "properties": props, "required": order, "additionalProperties": False}
    return {"type": "object", "properties": {"objects": {"type": "array", "items": obj, "maxItems": max_obj}},
            "required": ["objects"], "additionalProperties": False}


def to_gemini(text: str, box_prompt: str) -> str:
    """Qwen JSON -> the Gemini pilot's JSON (box_2d = [ymin, xmin, ymax, xmax] on 0-1000)."""
    d = json.loads(text)
    if box_prompt == "native":
        for o in d.get("objects", []):
            b = o.pop("bbox_2d", None)
            if isinstance(b, list) and len(b) == 4:
                x1, y1, x2, y2 = b
                o["box_2d"] = [y1, x1, y2, x2]
    return json.dumps(d)


def data_url(path: str) -> str:
    raw = open(path, "rb").read()
    im = Image.open(io.BytesIO(raw))
    if max(im.size) > MAX_PX:
        s = MAX_PX / max(im.size)
        im = im.convert("RGB").resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        raw = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def messages(frame_path: str, prompt: str, task: str) -> list[dict]:
    system = gp.DETECT_SYSTEM if task.startswith("detect") else gp.SYSTEM
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(frame_path)}},
                                         {"type": "text", "text": prompt}]}]


def make_llm(args):
    from vllm import LLM
    extra = {"speculative_config": {"method": "qwen3_5_mtp", "num_speculative_tokens": args.mtp}} if args.mtp else {}
    return LLM(model=MODEL_DIR, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
               gpu_memory_utilization=args.gpu_mem, limit_mm_per_prompt={"image": 1},
               max_num_seqs=args.max_num_seqs, enforce_eager=args.enforce_eager, seed=0, **extra)


def sampling_params(args, sch: dict):
    from vllm import SamplingParams
    kw = dict(SAMPLING[args.sampling], max_tokens=args.max_tokens, seed=0)
    if args.temperature is not None or args.presence_penalty is not None:
        kw = dict(temperature=args.temperature or 0.0, max_tokens=args.max_tokens, seed=0)
        if kw["temperature"] > 0:          # Qwen's non-thinking recommendation for the rest
            kw.update(top_p=args.top_p or 0.8, top_k=args.top_k or 20, min_p=0.0)
        if args.presence_penalty:
            kw["presence_penalty"] = args.presence_penalty
    try:                                    # vLLM >= 0.11
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(**kw, structured_outputs=StructuredOutputsParams(json=sch))
    except ImportError:                     # older vLLM
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(**kw, guided_decoding=GuidedDecodingParams(json=sch))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["detect", "detect_open_format"], required=True)
    ap.add_argument("--frames", required=True,
                    help="gold | montage72 | depiction | gold_v2 | fullrun500 | subject:<id> (e.g. subject:S00320001)")
    ap.add_argument("--sampling", choices=list(SAMPLING), default="greedy")
    ap.add_argument("--box-prompt", choices=["native", "gemini"], default="native")
    ap.add_argument("--temperature", type=float, help="override --sampling: temperature")
    ap.add_argument("--presence-penalty", type=float, default=None, help="override: presence penalty (Qwen: 1.5)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--mtp", type=int, default=0, help="speculative tokens with Qwen's MTP head (setting D: 3)")
    ap.add_argument("--max-obj", type=int, default=0, help="0: 25 for detect, 30 for detect_open_format (as Gemini)")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    ap.add_argument("--batch", type=int, default=64, help="frames per vLLM call (checkpoint interval)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no model: print the prompt, schema and one request")
    ap.add_argument("--shard", default="", help="K/N: run every N-th remaining frame from K, into <out>.shardK-of-N "
                                                 "(merge with --merge-shards N afterwards)")
    ap.add_argument("--merge-shards", type=int, default=0, help="append <out>.shard*-of-N files to <out> and exit")
    args = ap.parse_args()
    max_obj = args.max_obj or (30 if args.task == "detect_open_format" else 25)
    labels, _ = gp.load_vocab()
    prompt = prompt_for(args.task, labels, args.box_prompt, max_obj)
    sch = schema(args.task, args.box_prompt, max_obj)
    frames = frame_set(args.frames)
    if args.limit:
        frames = frames.head(args.limit)
    # file names follow the Gemini runs on the same frames, so the pilot scorers pick them up
    base = {"gold": "", "montage72": "_montage72", "depiction": "_depiction_frames",
            "gold_v2": "_gold_v2_frames_cap30", "fullrun500": "_fullrun500"}.get(args.frames)
    if base is None:
        base = "_subj_" + args.frames.split(":", 1)[1]
    custom = args.temperature is not None or args.presence_penalty is not None
    samp_tag = (f"_t{args.temperature if args.temperature is not None else 0:g}"
                f"_pp{args.presence_penalty or 0:g}") if custom else ("" if args.sampling == "greedy" else f"_{args.sampling}")
    tag = base + samp_tag \
        + ("" if args.box_prompt == "native" else "_geminibox") + (f"_lim{args.limit}" if args.limit else "")
    out = OUT / f"{MODEL_TAG}__{args.task}__p1{tag}.jsonl"
    if args.merge_shards:
        parts = sorted(out.parent.glob(f"{out.name}.shard*-of-{args.merge_shards}"))
        with open(out, "a") as f:
            for p in parts:
                f.write(p.read_text())
        print(f"merged {len(parts)} shards -> {out}")
        for p in parts:
            p.rename(p.with_name(p.name + ".merged"))
        return
    done = set()
    for p in [out, *out.parent.glob(f"{out.name}.shard*")]:   # main file and any shard files count as done
        if p.exists() and not p.name.endswith(".merged"):
            done |= {json.loads(l)["unit_id"] for l in open(p) if json.loads(l)["error"] is None}
    todo = frames[~frames.unit_id.isin(done)].reset_index(drop=True)
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        todo = todo[todo.index % n == k].reset_index(drop=True)
        out = out.with_name(f"{out.name}.shard{k}-of-{n}")
    print(f"{out.name}: {len(frames)} frames | {len(done)} done | {len(todo)} to do", flush=True)

    if args.dry_run:
        r = todo.iloc[0] if len(todo) else frames.iloc[0]
        m = messages(r.frame_path, prompt, args.task)
        print("--- system ---\n" + m[0]["content"] + "\n--- prompt ---\n" + prompt[:900] + (" ..." if len(prompt) > 900 else ""))
        print("--- schema ---\n" + json.dumps(sch)[:600])
        print("--- image ---", Image.open(r.frame_path).size, f"{len(m[1]['content'][0]['image_url']['url']) // 1024} KB as data URL")
        fake = json.dumps({"objects": [{"label": "cup", **({"format": "real"} if args.task == "detect_open_format" else {}),
                                        ("bbox_2d" if args.box_prompt == "native" else "box_2d"): [100, 200, 300, 400]}]})
        print("--- conversion check ---", fake, "->", to_gemini(fake, args.box_prompt))
        return

    llm = make_llm(args)
    sp = sampling_params(args, sch)
    meta = {"model": MODEL_TAG, "task": args.task, "pass": 1, "endpoint": f"local vllm {MODEL_DIR}",
            "sampling": args.sampling if not (args.temperature is not None or args.presence_penalty is not None)
            else f"t{args.temperature or 0:g}_pp{args.presence_penalty or 0:g}", "mtp": args.mtp,
            "box_prompt": args.box_prompt, "thinking": "off", "max_obj": max_obj,
            "max_px": MAX_PX}
    t_all = time.time()
    for i in range(0, len(todo), args.batch):
        chunk = todo.iloc[i:i + args.batch]
        convs = [messages(p, prompt, args.task) for p in chunk.frame_path]
        t0 = time.time()
        outs = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        dt = (time.time() - t0) / max(1, len(chunk))
        with open(out, "a") as f:
            for u, o in zip(chunk.unit_id, outs):
                c = o.outputs[0]
                rec = {"unit_id": u, **meta, "error": None, "raw_model": c.text, "latency": round(dt, 3),
                       "in_tok": len(o.prompt_token_ids or []), "out_tok": len(c.token_ids), "think_tok": 0,
                       "finish": c.finish_reason}
                try:
                    rec["raw"] = to_gemini(c.text, args.box_prompt)
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                f.write(json.dumps(rec) + "\n")
        n = i + len(chunk)
        print(f"  {n}/{len(todo)} frames  {n / (time.time() - t_all):.2f} frames/s", flush=True)
    print(f"DONE {out}")


if __name__ == "__main__":
    main()
