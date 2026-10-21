#!/usr/bin/env python3
"""Gemini model pilot on the rater-validated gold crops (valid5921_public).

Decides which model to use for the BabyView frame annotation run: the lab default
gemini-2.5-flash, or the cheaper gemini-2.5-flash-lite.
Every crop in the gold set was judged correct by raters, so it supports two tasks:

  name    blind: pick the CDI word (YOLOE's 295-label vocabulary) for the main
          object, or "none", plus a free open-vocabulary label.
  verify  three shuffled words (gold label, a same-domain foil, an other-domain
          foil): is each one the object shown? Gives hits and false alarms.
  detect  the whole source frame (no YOLOE input): every visible object from the
          vocabulary, with a box.
  detect_open  the same, open vocabulary: the most precise everyday name per object. Scored on finding the rater-validated object.
          Frames come from out/private/gold_frames.csv (build_gold_frames.py); its
          outputs go to out/private/ because they are keyed by video names.

Follows the lab guide and /home/mcfrank/bv-annotations/alignment/SETUP.md +
gemini_align.py: Vertex AI only (IRB rule; never the AI Studio endpoint), service
account, config from the four env vars only, temperature 0, thinking off, longest
image edge downsized to --max-px, JSON schema, resumable .jsonl checkpoint
consolidated to parquet, token usage written to usage_*.json.

  set -a; source ~/.secrets/hs-hs-langcog-gemini.env; set +a
  python run_gold_pilot.py --dry-run                                   # no API calls
  python run_gold_pilot.py --list-models
  python run_gold_pilot.py --model gemini-2.5-flash-lite --task name --sample 200   # sanity pass
  python run_gold_pilot.py --model gemini-2.5-flash-lite --task name               # all crops
  python run_gold_pilot.py --model gemini-2.5-flash-lite --task name --pass 2 --limit-per-cat 8
"""
import argparse
import hashlib
import io
import json
import os
import random
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

from vocab_groups import GENERAL, related

HERE = Path(__file__).resolve().parent
OD_ROOT = HERE.parents[1]
GOLD_ZIP = Path(
    "/home/j7yang/babyview-projects/ccn-2026-object-variability/data/shared_data_ccn_2026/"
    "valid5921_public/crops/valid7018_public_crops.zip"
)
LABELS_JSON = OD_ROOT / "annotation" / "cvat_cdi_labels.json"
CDI_CSV = OD_ROOT / "data" / "cdi_words.csv"
OUT_DIR = HERE / "out"
SEED = 20261016
MAX_OBJ = 25
ALLOWED_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")

SYSTEM = (
    "You annotate egocentric video from a head-mounted camera worn by a young child. "
    "Each image is a crop around one object detected in a video frame."
)

# Kept terse on purpose: short output = low output-token cost.
NAME_PROMPT = (
    "Return JSON with:\n"
    '- "cdi_word": the single best word for the main object in the crop, copied exactly '
    'from the list below; "none" only if no word on the list fits.\n'
    '- "open_label": the main object in your own words, as a short lowercase common noun '
    '(e.g. "sippy cup").\n'
    "Word list: {words}"
)

DETECT_SYSTEM = (
    "You annotate egocentric video from a head-mounted camera worn by a young child. "
    "Each image is one video frame."
)

DETECT_PROMPT = (
    "Return JSON with \"objects\": every clearly visible object in the frame whose name is "
    "on the word list below (at most {max_obj}). For each: \"label\", the word copied exactly from the list; \"box_2d\", "
    "[ymin, xmin, ymax, xmax] normalized to 0-1000. Pictures of objects (in books, on "
    "screens, toys) count.\n"
    "Word list: {words}"
)

# Open vocabulary: no word list. People are labelled "person" only (no descriptions).
DETECT_OPEN_PROMPT = (
    "Return JSON with \"objects\": every clearly visible object in the frame (at most {max_obj}). "
    "For each: \"label\", the most precise name for it that people use in everyday life, as a short "
    "lowercase common noun (e.g. \"sippy cup\" rather than \"cup\" or \"container\"); \"box_2d\", "
    "[ymin, xmin, ymax, xmax] normalized to 0-1000. Include body parts, clothing, furniture and parts "
    "of the room. Toys, pictures and screens count: name what they show (e.g. \"toy giraffe\", "
    "\"picture of a dog\"). Label any person just \"person\", without describing them.\n"
)

# Same, but says nothing about toys, pictures or screens and gives no example label, so any
# format word in a label ("stuffed", "poster", "cartoon") is unprompted.
DETECT_OPEN_NEUTRAL_PROMPT = (
    "Return JSON with \"objects\": every clearly visible object in the frame (at most {max_obj}). "
    "For each: \"label\", the most precise name for it that people use in everyday life, as a short "
    "lowercase name; \"box_2d\", [ymin, xmin, ymax, xmax] normalized to 0-1000. Include body parts, "
    "clothing, furniture and parts of the room. Label any person just \"person\", without describing them.\n"
)

# Combined: open label + closest CDI word + visual format, one call. detect_combo_nocap is the
# same prompt without "(at most N)" and without the schema's maxItems. Tried for the gold set
# v2 pre-fill (207/1,000 frames hit the 25-object cap) and dropped: without a cap some frames
# came back with far fewer objects and some looped until the token limit. The pre-fill uses
# detect_combo --max-obj 30 instead. Format is a small
# enum (Vertex accepts short enums); the word list is a reference, not a constraint.
FORMATS = ["real", "toy", "photo", "drawing", "screen"]
DETECT_COMBO_PROMPT = (
    "Return JSON with \"objects\": every clearly visible object in the frame (at most {max_obj}). For each:\n"
    "- \"label\": the most precise name for it that people use in everyday life, as a short lowercase name;\n"
    "- \"cdi_word\": the closest word from the word list below, copied exactly, or \"none\" if no word fits;\n"
    "- \"format\": \"real\" for a real object, \"toy\" for a toy version of something, \"photo\" for a "
    "photograph, \"drawing\" for a drawing, cartoon, painting or illustration, \"screen\" for an image shown on a screen;\n"
    "- \"box_2d\": [ymin, xmin, ymax, xmax] normalized to 0-1000.\n"
    "If an object is a toy, picture or screen image of something, label and cdi_word name what it shows, and "
    "format says what kind of depiction it is. Include body parts, clothing, furniture and parts of the room. "
    "Label any person just \"person\", without describing them.\n"
    "Word list: {{words}}"
)

# detect_combo without the CDI word and word list: open label + format only. Run on the gold
# set v2 frames with 2.5 Flash and Flash-Lite to compare with the detect_combo pre-fill.
DETECT_OPEN_FORMAT_PROMPT = (
    "Return JSON with \"objects\": every clearly visible object in the frame (at most {max_obj}). For each:\n"
    "- \"label\": the most precise name for it that people use in everyday life, as a short lowercase name;\n"
    "- \"format\": \"real\" for a real object, \"toy\" for a toy version of something, \"photo\" for a "
    "photograph, \"drawing\" for a drawing, cartoon, painting or illustration, \"screen\" for an image shown on a screen;\n"
    "- \"box_2d\": [ymin, xmin, ymax, xmax] normalized to 0-1000.\n"
    "If an object is a toy, picture or screen image of something, label names what it shows, and "
    "format says what kind of depiction it is. Include body parts, clothing, furniture and parts of the room. "
    "Label any person just \"person\", without describing them.\n"
)

VERIFY_PROMPT = (
    "Return JSON with \"answers\": one entry per word below, with \"word\" and \"is_object\" "
    "(true if the main object in the crop is that kind of thing, else false). Judge each "
    "word independently; more than one or none can be true.\n"
    "Words: {words}"
)


def load_vocab():
    labels = [d["name"] for d in json.load(open(LABELS_JSON))]
    cdi = pd.read_csv(CDI_CSV)
    dom = {}
    for col in ("english_gloss", "uni_lemma", "item_definition"):
        for w, c in zip(cdi[col].astype(str).str.lower(), cdi["category"]):
            dom.setdefault(w, c)
    domains = {w: dom.get(w.lower(), "unknown") for w in labels}
    return labels, domains


def load_gold(domains):
    z = zipfile.ZipFile(GOLD_ZIP)
    rows = []
    for n in z.namelist():
        if n.endswith(".jpg"):
            _, cat, fname = n.split("/")
            rows.append({"crop_id": fname[:-4], "category": cat, "zip_path": n})
    g = pd.DataFrame(rows).sort_values(["category", "crop_id"]).reset_index(drop=True)
    g["domain"] = g.category.map(domains)
    return g


def add_foils(gold, labels, domains):
    """One same-domain and one other-domain foil per crop, fixed by crop_id."""
    by_dom = {}
    for w in labels:
        by_dom.setdefault(domains[w], []).append(w)
    s_foil, o_foil, order = [], [], []
    for cid, cat, d in zip(gold.crop_id, gold.category, gold.domain):
        r = random.Random(int(hashlib.sha256(f"{SEED}:{cid}".encode()).hexdigest()[:12], 16))
        ok = lambda w: w not in GENERAL and not related(w, cat)
        s_foil.append(r.choice([w for w in by_dom[d] if ok(w)]))
        o_foil.append(r.choice([w for w in labels if ok(w) and domains[w] not in (d, "unknown")]))
        o = [0, 1, 2]
        r.shuffle(o)
        order.append(o)
    gold = gold.copy()
    gold["foil_same"], gold["foil_other"], gold["word_order"] = s_foil, o_foil, order
    return gold


def subsample(gold, k):
    if not k:
        return gold
    return (gold.sample(frac=1, random_state=SEED).groupby("category").head(k)
                .sort_values(["category", "crop_id"]).reset_index(drop=True))


def load_frames(path=None):
    """One unit per gold frame; the gold crops on it are scored against its boxes.
    path: another frame list (unit_id, video, frame_id, frame_path, category), e.g.
    out/private/depiction_frames.csv."""
    if path is not None:
        f = pd.read_csv(path, dtype={"frame_id": str})[["unit_id", "video", "frame_id", "frame_path", "category"]]
        f["crop_id"] = f.unit_id
        return f.sort_values("unit_id").reset_index(drop=True)
    g = pd.read_csv(OUT_DIR / "private" / "gold_frames.csv", dtype={"frame_id": str})
    f = g.groupby("frame_path", as_index=False).agg(video=("video", "first"), frame_id=("frame_id", "first"),
                                                     category=("category", "first"))
    f["unit_id"] = f.video + "/" + f.frame_id
    f["crop_id"] = f.unit_id  # so subsample() can sort
    return f.sort_values("unit_id").reset_index(drop=True)


def make_client():
    """Vertex config comes from env (SETUP.md); refuse anything else."""
    from google import genai
    need = ["GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS"]
    missing = [v for v in need if not os.environ.get(v)]
    if missing or os.environ["GOOGLE_GENAI_USE_VERTEXAI"].lower() != "true":
        sys.exit(f"Refusing to run: Vertex env not set ({missing or 'GOOGLE_GENAI_USE_VERTEXAI != true'}). "
                 "Source ~/.secrets/<project>.env first; never use the AI Studio endpoint.")
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        sys.exit("Refusing to run: an API key is set in the environment. Unset it; Vertex uses the service account.")
    client = genai.Client()
    assert client.vertexai, "client is not on Vertex AI"
    return client


def build_config(task, labels, model, args):
    from typing import List, Literal
    from google.genai import types
    from pydantic import BaseModel, Field, create_model

    # Words are plain strings, not an enum: Vertex rejects an enum this long (400
    # INVALID_ARGUMENT at 150+ values; the vocab has 295). The prompt lists the words and
    # score_gold_pilot.py counts answers that are not on the list.
    if task == "name":
        schema = create_model("Naming", cdi_word=(str, ...), open_label=(str, ...))
    elif task in ("detect_combo", "detect_combo_nocap"):
        Obj = create_model("Obj", label=(str, ...), cdi_word=(str, ...), format=(Literal[tuple(FORMATS)], ...),
                           box_2d=(List[int], ...))
        cap = {} if task == "detect_combo_nocap" else {"max_length": MAX_OBJ}
        schema = create_model("Detect", objects=(List[Obj], Field(..., **cap)))
    elif task == "detect_open_format":
        Obj = create_model("Obj", label=(str, ...), format=(Literal[tuple(FORMATS)], ...), box_2d=(List[int], ...))
        schema = create_model("Detect", objects=(List[Obj], Field(..., max_length=MAX_OBJ)))
    elif task in ("detect","detect_open", "detect_open_neutral"):
        Obj = create_model("Obj", label=(str, ...), box_2d=(List[int], ...))
        # maxItems stops runaway lists (Flash-Lite once repeated "toy" boxes until MAX_TOKENS)
        schema = create_model("Detect", objects=(List[Obj], Field(..., max_length=MAX_OBJ)))
    else:
        class Check(BaseModel):
            word: str
            is_object: bool

        class Verify(BaseModel):
            answers: List[Check]
        schema = Verify

    thinking = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(
        system_instruction=DETECT_SYSTEM if task.startswith("detect") else SYSTEM, response_mime_type="application/json", response_schema=schema,
        temperature=0.0, thinking_config=thinking, max_output_tokens=args.max_output_tokens)


def prompt_for(task, row, labels):
    if task == "name":
        return NAME_PROMPT.format(words=", ".join(labels))
    if task == "detect_open":
        return DETECT_OPEN_PROMPT.format(max_obj=MAX_OBJ)
    if task == "detect_combo":
        return DETECT_COMBO_PROMPT.format(max_obj=MAX_OBJ).format(words=", ".join(labels))
    if task == "detect_open_format":
        return DETECT_OPEN_FORMAT_PROMPT.format(max_obj=MAX_OBJ)
    if task == "detect_combo_nocap":
        return DETECT_COMBO_PROMPT.replace(" (at most {max_obj})", "").format(words=", ".join(labels))
    if task == "detect_open_neutral":
        return DETECT_OPEN_NEUTRAL_PROMPT.format(max_obj=MAX_OBJ)
    if task == "detect":
        return DETECT_PROMPT.format(words=", ".join(labels), max_obj=MAX_OBJ)
    words = [row.category, row.foil_same, row.foil_other]
    return VERIFY_PROMPT.format(words=", ".join(words[i] for i in row.word_order))


def to_jpeg(raw, max_px):
    """Downsize longest edge to max_px (as in gemini_align.load_jpeg); crops are usually smaller."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(im.size) <= max_px:
        return raw
    s = max_px / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def call(client, model, config, prompt, jpeg):
    """jpeg: bytes, or a list of bytes for several images (sent in order, before the prompt)."""
    from google.genai import types
    jpegs = jpeg if isinstance(jpeg, list) else [jpeg]
    parts = [types.Part.from_bytes(data=j, mime_type="image/jpeg") for j in jpegs] + [prompt]
    for attempt in range(6):
        try:
            return client.models.generate_content(model=model, contents=parts, config=config)
        except Exception as e:
            msg = str(e)
            transient = any(c in msg for c in ("429", "503", "500", "RESOURCE_EXHAUSTED",
                                               "UNAVAILABLE", "DEADLINE", "timeout"))
            if attempt == 5 or not transient:
                raise
            time.sleep(min(60, 2 ** attempt) + random.random())


def consolidate(out, args, meta):
    """jsonl checkpoint -> parquet (last record per unit) + usage_*.json."""
    recs = pd.DataFrame([json.loads(l) for l in open(out)]).drop_duplicates("unit_id", keep="last")
    recs.to_parquet(out.with_suffix(".parquet"), index=False)
    if "error" not in recs:
        recs["error"] = None
    ok = recs[recs.error.isna()].reindex(columns=[*recs.columns, "in_tok", "out_tok", "think_tok", "model_version"]
                                         ).loc[:, lambda x: ~x.columns.duplicated()]
    usage = {**meta, "date": time.strftime("%Y-%m-%d"), "units": int(len(recs)), "scored": int(len(ok)),
             "failures": int(recs.error.notna().sum()),
             "input_tokens": int(ok.in_tok.sum()), "output_tokens": int(ok.out_tok.sum()),
             "thinking_tokens": int(ok.think_tok.sum()),
             "model_versions": sorted(ok.model_version.dropna().unique().tolist())}
    json.dump(usage, open(out.parent / f"usage_{out.stem}.json", "w"), indent=1)
    return usage


def main():
    global MAX_OBJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash-lite", choices=ALLOWED_MODELS)
    ap.add_argument("--task", choices=["name", "verify", "detect", "detect_open", "detect_open_neutral", "detect_combo",
                                       "detect_combo_nocap", "detect_open_format"], default="name")
    ap.add_argument("--pass", dest="pass_", type=int, default=1, help="2 = retest pass")
    ap.add_argument("--limit-per-cat", type=int, default=0, help="crops per category (0 = all)")
    ap.add_argument("--sample", type=int, default=0, help="random N crops (sanity pass)")
    ap.add_argument("--frames", type=Path, help="detect tasks: frame list CSV instead of the gold frames")
    ap.add_argument("--units", type=Path, help="text file of unit ids to run (one per line); tags the output")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-px", type=int, default=512, help="downsize longest image edge")
    ap.add_argument("--max-output-tokens", type=int, default=0, help="0 = 512, or 3072 for detect")
    ap.add_argument("--max-obj", type=int, default=MAX_OBJ, help="detect tasks: object cap in prompt and schema")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()
    MAX_OBJ = args.max_obj
    args.max_output_tokens = args.max_output_tokens or (16384 if args.task == "detect_combo_nocap" else
                                                         max(4096, 200 * MAX_OBJ) if args.task == "detect_combo" else 3072 if args.task.startswith("detect") else 512)

    labels, domains = load_vocab()
    OUT_DIR.mkdir(exist_ok=True)
    if args.task.startswith("detect"):
        gold = subsample(load_frames(args.frames), args.limit_per_cat)
    else:
        gold = subsample(add_foils(load_gold(domains), labels, domains), args.limit_per_cat)
        if not (args.sample or args.limit_per_cat):
            gold.drop(columns="zip_path").to_csv(OUT_DIR / "gold_manifest.csv", index=False)
        gold["unit_id"] = gold.crop_id
    if args.sample:
        gold = gold.sample(min(args.sample, len(gold)), random_state=SEED).reset_index(drop=True)
    if args.units:
        keep = {l.strip() for l in open(args.units) if l.strip()}
        gold = gold[gold.unit_id.isin(keep)].reset_index(drop=True)
        assert len(gold) == len(keep), f"{len(keep) - len(gold)} unit ids not found"

    if args.dry_run:
        r = gold.iloc[0]
        print(f"{len(gold)} units, {gold.category.nunique()} categories, vocab {len(labels)} "
              f"({sum(d == 'unknown' for d in domains.values())} without a CDI domain)")
        if "domain" in gold:
            print(gold.domain.value_counts().to_dict())
        print("\n--- system ---\n" + SYSTEM)
        if not args.task.startswith("detect"):
            print("\n--- name prompt ---\n" + prompt_for("name", r, labels)[:600] + " ...")
            print("\n--- verify prompt ---\n" + prompt_for("verify", r, labels))
        else:
            print("\n--- detect system ---\n" + DETECT_SYSTEM)
            print(f"\n--- {args.task} prompt ---\n" + prompt_for(args.task, r, labels)[:700] + " ...")
            im = Image.open(io.BytesIO(to_jpeg(open(r.frame_path, "rb").read(), args.max_px)))
            print(f"frame sent as {im.size}")
        return

    client = make_client()
    if args.list_models:
        for m in client.models.list():
            if "gemini" in m.name:
                print(m.name)
        return

    config = build_config(args.task, labels, args.model, args)
    tag = f"_s{args.sample}" if args.sample else (f"_{args.units.stem}" if args.units else
                                                  f"_{args.frames.stem}" if args.frames else "")
    if args.task.startswith("detect") and MAX_OBJ != 25:
        tag += f"_cap{MAX_OBJ}"
    out_dir = OUT_DIR / "private" if args.task.startswith("detect") else OUT_DIR
    out = out_dir / f"{args.model}__{args.task}__p{args.pass_}{tag}.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            d = json.loads(line)
            if d.get("error") is None:
                done.add(d["unit_id"])
    todo = gold[~gold.unit_id.isin(done)]
    print(f"{out.name}: {len(gold)} units | {len(done)} already scored | {len(todo)} to do", flush=True)

    z = zipfile.ZipFile(GOLD_ZIP)
    zlock, wlock = threading.Lock(), threading.Lock()
    meta = {"model": args.model, "task": args.task, "pass": args.pass_,
            "endpoint": f"vertexai/{client._api_client.location}", "temperature": 0.0,
            "thinking": "budget0", "max_obj": MAX_OBJ,
            "max_px": args.max_px}

    def work(row):
        if args.task.startswith("detect"):
            raw = open(row.frame_path, "rb").read()
        else:
            with zlock:
                raw = z.read(row.zip_path)
        rec = {"unit_id": row.unit_id, **meta, "error": None}
        t0 = time.time()
        try:
            resp = call(client, args.model, config, prompt_for(args.task, row, labels), to_jpeg(raw, args.max_px))
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
        futs = [ex.submit(work, r) for r in todo.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            ok += f.result()
            bad += not f.result()
            if i % 200 == 0 or i == len(futs):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(futs)} ok={ok} failed={bad} {rate:.1f}/s", flush=True)
    if out.exists():
        u = consolidate(out, args, meta)
        print(f"DONE {out.name} scored={u['scored']} failures={u['failures']} "
              f"tokens in/out/think={u['input_tokens']}/{u['output_tokens']}/{u['thinking_tokens']}", flush=True)


if __name__ == "__main__":
    main()
