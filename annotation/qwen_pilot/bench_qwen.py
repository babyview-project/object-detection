#!/usr/bin/env python3
"""Throughput test for the full Qwen run: same 200 frames, one setting per process.

Settings (setup B prompt, temperature 0, one A40):
  A_current   max_num_seqs 32, batches of 64 frames, pretty-printed JSON (the pilot)
  B_inflight  max_num_seqs 96, all frames submitted at once
  C_compact   B + compact JSON (structured outputs with disable_any_whitespace)
  D_mtp       C + multi-token prediction (Qwen3.8's MTP head, 3 speculative tokens)
Each run writes out/private/bench/<setting>.jsonl (records in the pilot format) and
<setting>.json (load time, generation time, frames/s, tokens). --report compares every
setting with A_current on the same frames: frames/s, objects per frame, and how many of
A's boxes come back with the same label (IoU >= 0.5) and the same format.

  CUDA_VISIBLE_DEVICES=1 python bench_qwen.py --setting A_current
  python bench_qwen.py --report
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import run_qwen_pilot as rq

OUT = Path(__file__).resolve().parent / "out" / "private" / "bench"
SETTINGS = {
    "A_current": dict(max_num_seqs=32, batch=64, compact=False, mtp=0),
    "B_inflight": dict(max_num_seqs=96, batch=10_000, compact=False, mtp=0),
    "C_compact": dict(max_num_seqs=96, batch=10_000, compact=True, mtp=0),
    "D_mtp": dict(max_num_seqs=96, batch=10_000, compact=True, mtp=3),
    # second round: compact JSON is an engine-level option in vLLM 0.32 (the per-request flag
    # used by C/D had no effect, so C ran as B and D as B + MTP)
    "C2_compact": dict(max_num_seqs=96, batch=10_000, compact="engine", mtp=0),
    "E_noschema": dict(max_num_seqs=96, batch=10_000, compact=False, mtp=0, schema=False),
    "F_compact_mtp": dict(max_num_seqs=96, batch=10_000, compact="engine", mtp=3),
    # third round: full-precision (bf16) weights split over 2 GPUs, no FP8->16-bit conversion
    # (A40s have no FP8), with setting D's options; compare frames/s per GPU with D
    "G_bf16_tp2_mtp": dict(max_num_seqs=96, batch=10_000, compact=False, mtp=3, tp=2, model="Qwen/Qwen3.8-27B"),
}
N_FRAMES = 200


def frames():
    f = rq.frame_set("fullrun500").sort_values("unit_id")
    return f.sample(N_FRAMES, random_state=20261022).reset_index(drop=True)


def run(setting: str):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    cfg = SETTINGS[setting]
    labels, _ = rq.gp.load_vocab()
    prompt = rq.prompt_for("detect_open_format", labels, "native", 30)
    sch = rq.schema("detect_open_format", "native", 30)
    fr = frames()
    model = rq.MODEL_DIR if "model" not in cfg else __import__("huggingface_hub").snapshot_download(cfg["model"], local_files_only=True)
    kw = dict(model=model, tensor_parallel_size=cfg.get("tp", 1), max_model_len=16384, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 1}, max_num_seqs=cfg["max_num_seqs"], seed=0)
    if cfg["compact"] == "engine":
        kw["structured_outputs_config"] = {"backend": "xgrammar", "disable_any_whitespace": True}
    if cfg["mtp"]:
        kw["speculative_config"] = {"method": "qwen3_5_mtp", "num_speculative_tokens": cfg["mtp"]}
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    so = None if cfg.get("schema", True) is False else \
        StructuredOutputsParams(json=sch, disable_any_whitespace=bool(cfg["compact"]))
    sp = SamplingParams(temperature=0.0, max_tokens=6000, seed=0, structured_outputs=so)
    convs = [rq.messages(p, prompt, "detect_open_format") for p in fr.frame_path]
    llm.chat(convs[:4], sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})   # warm-up, not timed
    OUT.mkdir(parents=True, exist_ok=True)
    recs, t1 = [], time.time()
    for i in range(0, len(convs), cfg["batch"]):
        outs = llm.chat(convs[i:i + cfg["batch"]], sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        for u, o in zip(fr.unit_id[i:i + cfg["batch"]], outs):
            c = o.outputs[0]
            rec = {"unit_id": u, "setting": setting, "error": None, "raw_model": c.text,
                   "out_tok": len(c.token_ids), "in_tok": len(o.prompt_token_ids or []), "finish": c.finish_reason}
            try:
                rec["raw"] = rq.to_gemini(c.text, "native")
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
            recs.append(rec)
    gen_s = time.time() - t1
    with open(OUT / f"{setting}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = {"setting": setting, **cfg, "frames": len(recs), "load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
               "frames_per_s": round(len(recs) / gen_s, 3), "gpus": cfg.get("tp", 1),
               "frames_per_s_per_gpu": round(len(recs) / gen_s / cfg.get("tp", 1), 3), "out_tok_mean": sum(r["out_tok"] for r in recs) / len(recs),
               "parse_failures": sum(r["error"] is not None for r in recs)}
    (OUT / f"{setting}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def report():
    import pandas as pd
    from compare_qwen_gemini import iou, load, looping, objects
    base = load(OUT / "A_current.jsonl")
    rows = []
    for s in SETTINGS:
        p = OUT / f"{s}.json"
        if not p.exists():
            rows.append({"setting": s, "status": "not run / failed"}); continue
        m = json.loads(p.read_text())
        recs = load(OUT / f"{s}.jsonl")
        same_lab = same_fmt = found = total = 0
        n_obj = []
        for u, r in recs.items():
            o, b = objects(r), objects(base.get(u))
            if o is None or b is None:
                continue
            n_obj.append(len(o))
            for x in b:
                total += 1
                best = max(((iou(x["box_2d"], y["box_2d"]), y) for y in o), key=lambda t: t[0], default=(0, None))
                if best[0] >= 0.5:
                    found += 1
                    same_lab += best[1]["label"].strip().lower() == x["label"].strip().lower()
                    same_fmt += best[1].get("format") == x.get("format")
        rows.append({"setting": s, "gpus": m.get("gpus", 1), "frames_per_s": m["frames_per_s"],
                     "frames_per_s_per_gpu": m.get("frames_per_s_per_gpu", m["frames_per_s"]), "speedup": None, "load_s": m["load_s"],
                     "out_tok_mean": round(m["out_tok_mean"], 1), "parse_failures": m["parse_failures"],
                     "objects_per_frame": sum(n_obj) / max(1, len(n_obj)),
                     "looping_frames": sum(looping(objects(r) or []) for r in recs.values()),
                     "A_boxes_found": found / max(1, total), "same_label": same_lab / max(1, found),
                     "same_format": same_fmt / max(1, found)})
    t = pd.DataFrame(rows)
    if "frames_per_s" in t and t.frames_per_s.notna().any():
        a = t.loc[t.setting == "A_current", "frames_per_s"]
        if len(a) and a.notna().all():
            t["speedup"] = t.frames_per_s_per_gpu / float(a.iloc[0])   # per GPU, vs A on one GPU
    t.to_csv(OUT / "report.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=list(SETTINGS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.setting)
