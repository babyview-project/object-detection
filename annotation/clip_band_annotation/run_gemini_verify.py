#!/usr/bin/env python3
"""Gemini verify on the CLIP-band sample: the like-for-like comparison with the CLIP filter.

Every item in items.csv (7,155 YOLOE crops, 129 categories x 7 CLIP bands) is judged
with the gold-pilot verify task: three shuffled words (YOLOE's label, a same-domain
foil, an other-domain foil), "is the object that kind of thing?" for each. Foils give
Gemini's false-alarm rate in every band, so a high acceptance rate in a low CLIP band
can be told apart from a model that accepts everything.

Two conditions, mirroring the two human steps:
  crop     the crop alone (what CLIP scores), pilot prompt unchanged
  context  the crop, then the full frame with the same object outlined in magenta
Crops are sent at native size (<= 512 px, as in the pilot); frames at their native
512x910 (--frame-px). Both conditions use media_resolution MEDIUM (one 258-token tile
per image): by default Vertex tiles a lone image at high resolution (a large crop ~1,290
tokens, a frame 1,806) but drops every image to one tile when a call has two, and HIGH
is refused for multi-image calls. MEDIUM keeps the crop identical across conditions, so
only the context differs. (The gold pilot used the default; for crops <= ~384 px, most
of this sample, that was one tile anyway.)

Vertex only, same rules as gemini_pilot/run_gold_pilot.py (whose client, config, call
and consolidate functions this reuses). Outputs are keyed by item_id and stay in
out/private/gemini/ (git-ignored) with the rest of the sample.

  set -a; source ~/.secrets/hs-hs-langcog-gemini.env; set +a
  python run_gemini_verify.py --cond crop --sample 50       # sanity pass
  python run_gemini_verify.py --cond crop
  python run_gemini_verify.py --cond context
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gemini_pilot"))
import run_gold_pilot as gp  # noqa: E402

OUT = HERE / "out" / "private"
IMG = OUT / "images"
GEM = OUT / "gemini"

CONTEXT_SYSTEM = (
    "You annotate egocentric video from a head-mounted camera worn by a young child. "
    "You get two images: first, a crop around one object detected in a video frame; "
    "second, the full frame, with that same object outlined by a magenta box."
)
CONTEXT_PROMPT = gp.VERIFY_PROMPT.replace(
    "the main object in the crop", "the object in the crop (outlined in the frame)")
assert CONTEXT_PROMPT != gp.VERIFY_PROMPT


def load_items():
    labels, domains = gp.load_vocab()
    s = pd.read_csv(OUT / "items.csv")
    missing = sorted(set(s.class_name) - set(labels))
    assert not missing, f"categories not in the vocabulary: {missing}"
    s = s.rename(columns={"class_name": "category"})
    s["crop_id"] = s["item_id"]  # add_foils seeds each item's foils by crop_id
    s["domain"] = s.category.map(domains)
    s = gp.add_foils(s, labels, domains)
    s["unit_id"] = s["item_id"]
    return s, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", choices=["crop", "context"], required=True)
    ap.add_argument("--model", default="gemini-2.5-flash-lite", choices=gp.ALLOWED_MODELS)
    ap.add_argument("--sample", type=int, default=0, help="random N items (sanity pass)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--crop-px", type=int, default=512, help="longest crop edge sent (pilot: 512)")
    ap.add_argument("--frame-px", type=int, default=910, help="longest frame edge sent (native 910)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.max_output_tokens = 512

    items, labels = load_items()
    GEM.mkdir(parents=True, exist_ok=True)
    items[["item_id", "category", "domain", "foil_same", "foil_other", "word_order"]].to_csv(
        GEM / "verify_manifest.csv", index=False)
    if args.sample:
        items = items.sample(min(args.sample, len(items)), random_state=gp.SEED).reset_index(drop=True)

    config = gp.build_config("verify", labels, args.model, args)
    from google.genai import types
    update = {"media_resolution": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM}
    if args.cond == "context":
        update["system_instruction"] = CONTEXT_SYSTEM
    config = config.model_copy(update=update)
    prompt_tpl = gp.VERIFY_PROMPT if args.cond == "crop" else CONTEXT_PROMPT

    def prompt(r):
        words = [r.category, r.foil_same, r.foil_other]
        return prompt_tpl.format(words=", ".join(words[i] for i in r.word_order))

    def images(r):
        crop = gp.to_jpeg(open(IMG / f"{r.item_id}_crop.jpg", "rb").read(), args.crop_px)
        if args.cond == "crop":
            return crop
        return [crop, gp.to_jpeg(open(IMG / f"{r.item_id}_frame.jpg", "rb").read(), args.frame_px)]

    if args.dry_run:
        r = items.iloc[0]
        print(f"{len(items)} items, {items.category.nunique()} categories")
        print("--- system ---\n" + config.system_instruction + "\n--- prompt ---\n" + prompt(r))
        ims = images(r)
        print("images (bytes):", [len(x) for x in (ims if isinstance(ims, list) else [ims])])
        return

    client = gp.make_client()
    tag = f"_s{args.sample}" if args.sample else ""
    out = GEM / f"{args.model}__verify_{args.cond}{tag}.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            d = json.loads(line)
            if d.get("error") is None:
                done.add(d["unit_id"])
    todo = items[~items.unit_id.isin(done)]
    print(f"{out.name}: {len(items)} items | {len(done)} already scored | {len(todo)} to do", flush=True)

    meta = {"model": args.model, "task": f"verify_{args.cond}", "pass": 1,
            "endpoint": f"vertexai/{client._api_client.location}", "temperature": 0.0,
            "thinking": "budget0", "media_resolution": "MEDIUM", "crop_px": args.crop_px,
            "frame_px": args.frame_px if args.cond == "context" else None}
    wlock = threading.Lock()

    def work(r):
        rec = {"unit_id": r.unit_id, **meta, "error": None}
        t0 = time.time()
        try:
            resp = gp.call(client, args.model, config, prompt(r), images(r))
            u = resp.usage_metadata
            rec.update(raw=resp.text, latency=round(time.time() - t0, 2),
                       in_tok=u.prompt_token_count, out_tok=u.candidates_token_count or 0,
                       think_tok=getattr(u, "thoughts_token_count", None) or 0,
                       finish=str(resp.candidates[0].finish_reason) if resp.candidates else None,
                       model_version=getattr(resp, "model_version", None))
            json.loads(resp.text)
        except Exception as e:  # logged; retried on relaunch
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        with wlock, open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec["error"] is None

    ok = bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, r) for r in todo.itertuples(index=False)]
        for i, f in enumerate(as_completed(futs), 1):
            ok, bad = (ok + 1, bad) if f.result() else (ok, bad + 1)
            if i % 200 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} ok={ok} failed={bad} {i / (time.time() - t0):.1f}/s", flush=True)
    if out.exists():
        u = gp.consolidate(out, args, meta)
        print(f"DONE {out.name} scored={u['scored']} failures={u['failures']} "
              f"tokens in/out/think={u['input_tokens']}/{u['output_tokens']}/{u['thinking_tokens']}", flush=True)


if __name__ == "__main__":
    main()
