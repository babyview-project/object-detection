#!/usr/bin/env python3
"""Does temperature or a group rule stop Gemini's duplicate-box loops? (pilot frames)

Frames: from the 500-frame batch pilot (out/private/pilot_results/results_pilot.jsonl),
every looping frame (one label >= 10 times with >= 3 near-duplicate box pairs, IoU > 0.7,
or a reply cut off at the token cap) plus 100 other frames (seeded).

Variants, online Vertex calls with the full run's request (run_full_batch.request: same
system instruction, prompt, schema, 30-object cap, token cap, thinking off, full frame):
  t0         temperature 0 (the full-run setting; re-run online to see run-to-run change)
  t05        temperature 0.5 (Gemini cookbook: "to prevent the model from repeating itself")
  group      temperature 0 + GROUP_RULE
  t05_group  temperature 0.5 + GROUP_RULE

Measures per variant, separately for looping and other frames: looping rate (same rule),
parse failures, objects per frame, frames at the cap, output tokens, distinct labels per
frame, and on the other frames how many of the batch pilot's boxes are found again
(IoU >= 0.5, any label) and with the same label.

Outputs: out/private/loop_test/<variant>.jsonl (resumable), summary.csv.

  set -a; source ~/.secrets/hs-hs-langcog-gemini.env; set +a
  python test_loop_variants.py [--n-other 100] [--workers 8]
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import run_full_batch as rb

OUT = rb.PRIV / "loop_test"
GROUP_RULE = ("List each object once. If many similar things form a group (for example clouds in the sky, "
              "books on a shelf, leaves on a tree), give one box around the group with a plural label "
              "instead of one box per item.")
VARIANTS = {"t0": (0.0, False), "t05": (0.5, False), "group": (0.0, True), "t05_group": (0.5, True)}


def iou(a, b):
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def objects(raw):
    return [o for o in json.loads(raw).get("objects", []) if len(o.get("box_2d", [])) == 4]


def looping(objs):
    """One label >= 10 times with >= 3 near-duplicate same-label box pairs (IoU > 0.7)."""
    by = collections.defaultdict(list)
    for o in objs:
        by[o["label"].strip().lower()].append(o["box_2d"])
    for boxes in by.values():
        if len(boxes) >= 10 and sum(iou(a, b) > 0.7 for a, b in itertools.combinations(boxes, 2)) >= 3:
            return True
    return False


def pick_frames(n_other):
    pilot = {}
    for line in open(rb.PRIV / "pilot_results" / "results_pilot.jsonl"):
        r = json.loads(line)
        pilot[r["unit_id"]] = r
    loop = [u for u, r in pilot.items() if r["error"] or looping(objects(r["raw"]))]
    rest = sorted(u for u in pilot if u not in set(loop))
    other = random.Random(20261022).sample(rest, min(n_other, len(rest)))
    return pilot, [(u, "looping") for u in sorted(loop)] + [(u, "other") for u in other]


def make_config(temp, group):
    from google.genai import types
    req = rb.request("gs://unused")["request"]
    cfg = types.GenerateContentConfig(
        system_instruction=rb.DETECT_SYSTEM, response_mime_type="application/json",
        response_schema=types.Schema.model_validate(req["generationConfig"]["responseSchema"]),
        temperature=temp, max_output_tokens=rb.MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0))
    prompt = req["contents"][0]["parts"][1]["text"]
    if group:
        prompt = prompt.rstrip("\n") + "\n" + GROUP_RULE + "\n"
    return cfg, prompt


def run(client, variant, frames, manifest, workers):
    from google.genai import types
    temp, group = VARIANTS[variant]
    cfg, prompt = make_config(temp, group)
    out = OUT / f"{variant}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["unit_id"] for l in open(out) if json.loads(l)["error"] is None}
    todo = [(u, kind) for u, kind in frames if u not in done]
    lock = threading.Lock()

    def one(item):
        u, kind = item
        rec = {"unit_id": u, "kind": kind, "variant": variant, "temperature": temp, "group_rule": group, "error": None}
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=rb.MODEL, config=cfg,
                    contents=[types.Part.from_bytes(data=rb.jpeg(manifest[u]), mime_type="image/jpeg"), prompt])
                m = resp.usage_metadata
                rec.update(raw=resp.text, in_tok=m.prompt_token_count, out_tok=m.candidates_token_count or 0,
                           finish=str(resp.candidates[0].finish_reason) if resp.candidates else None)
                json.loads(resp.text)
                rec["error"] = None
                break
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
                if not any(c in str(e) for c in ("429", "503", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                    break            # e.g. broken JSON: keep it as a failure (that is what we measure)
                time.sleep(2 ** attempt)
        with lock, open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(as_completed([ex.submit(one, it) for it in todo]))
    print(f"{variant}: {len(todo)} calls", flush=True)


def summarize(pilot, frames):
    kind = dict(frames)
    rows = []
    for variant in ["batch_t0"] + list(VARIANTS):
        if variant == "batch_t0":
            recs = {u: {**pilot[u], "kind": kind[u]} for u in kind}
        else:
            p = OUT / f"{variant}.jsonl"
            recs = {}
            for l in open(p):
                r = json.loads(l)
                recs[r["unit_id"]] = r
        for k in ("looping", "other"):
            rs = [r for u, r in recs.items() if kind.get(u) == k]
            ok = [r for r in rs if r["error"] is None]
            objs = {r["unit_id"]: objects(r["raw"]) for r in ok}
            row = {"variant": variant, "frames": k, "n": len(rs), "parse_failures": len(rs) - len(ok),
                   "looping": sum(looping(o) for o in objs.values()),
                   "objects_per_frame": sum(map(len, objs.values())) / max(1, len(objs)),
                   "at_cap": sum(len(o) >= rb.MAX_OBJ for o in objs.values()),
                   "distinct_labels_per_frame": sum(len({x["label"].strip().lower() for x in o}) for o in objs.values()) / max(1, len(objs)),
                   "out_tokens_mean": sum(r.get("out_tok", 0) for r in rs) / max(1, len(rs))}
            if k == "other" and variant != "batch_t0":
                found = same = total = 0
                for u, o in objs.items():
                    base = objects(pilot[u]["raw"]) if pilot[u]["error"] is None else []
                    for b in base:
                        total += 1
                        best = max(((iou(b["box_2d"], x["box_2d"]), x) for x in o), key=lambda t: t[0], default=(0, None))
                        if best[0] >= 0.5:
                            found += 1
                            same += best[1]["label"].strip().lower() == b["label"].strip().lower()
                row["batch_boxes_found"] = found / max(1, total)
                row["found_with_same_label"] = same / max(1, found)
            rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))
    tin = tout = 0
    for v in VARIANTS:
        for l in open(OUT / f"{v}.jsonl"):
            r = json.loads(l)
            tin += r.get("in_tok", 0) or 0
            tout += r.get("out_tok", 0) or 0
    pi, po = rb.PRICES[rb.MODEL]
    print(f"\ncost of this test (online prices): ${(tin * pi + tout * po) / 1e6:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-other", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pilot, frames = pick_frames(args.n_other)
    print(f"frames: {sum(k == 'looping' for _, k in frames)} looping, {sum(k == 'other' for _, k in frames)} other")
    if not args.summary_only:
        m = pd.read_csv(rb.PRIV / "manifest.csv")
        want = {u for u, _ in frames}
        manifest = dict(zip(m.unit_id[m.unit_id.isin(want)], m.frame_path[m.unit_id.isin(want)]))
        client = rb.make_client()
        for v in VARIANTS:
            run(client, v, frames, manifest, args.workers)
    summarize(pilot, frames)


if __name__ == "__main__":
    main()
