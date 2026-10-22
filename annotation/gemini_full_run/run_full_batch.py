#!/usr/bin/env python3
"""Full Gemini run on 2025.2 BV-main through Vertex batch prediction (setup B).

Setup B = the gold set v2 pre-fill: gemini-2.5-flash, detect_open_format (open label + format +
box per object, at most 30), full frame (512x910), temperature 0, thinking off. Prompt, system
instruction and formats come from ../gemini_pilot/run_gold_pilot.py, so the run matches the pilot.
Frames: every 5th 1 fps frame of each BV-main video in 2025.2 (subjects S00xx, S00270001
excluded), from /ccn2a/dataset/babyview/2025.2/extracted_frames_1fps.

Stages (run in order; each is resumable and keeps its state in out/private/):
  manifest  list the frames            -> out/private/manifest.csv (unit_id = <video_id>/<frame:05d>)
  prepare   batch request files        -> out/private/requests/shard_NNN.jsonl (--shard-size per file)
  smoke     send N prepared requests online with the image inlined (no bucket needed) to check
            the prompt, schema and config on Vertex; prints the parsed objects
  upload    frames -> gs://BUCKET/frames/<unit_id>.jpg, shards -> gs://BUCKET/requests/
  submit    one batch job per uploaded shard, at most --max-active running at once -> out/private/jobs.json
  status    job states and completion counts
  collect   download finished jobs' predictions -> out/private/results.jsonl (same record format as the
            pilot's jsonl) + usage.json; failed units -> out/private/failed_units.txt
  cleanup   delete gs://BUCKET/frames/ and requests/ (asks for --yes)
After collect, `prepare --retry` writes a shard of the failed units; upload, submit and collect it.

Frames go to Gemini only through Vertex in the grant project (IRB); the bucket must be in the
same project and region (us-central1). Delete the frames (cleanup) once results are checked.

  set -a; source ~/.secrets/hs-hs-langcog-gemini.env; set +a
  python run_full_batch.py manifest
  python run_full_batch.py prepare                   # or: prepare --limit 200 --tag pilot
  python run_full_batch.py smoke --n 5
  python run_full_batch.py upload --bucket hs-hs-langcog-gemini-bv-objects
  python run_full_batch.py submit --bucket hs-hs-langcog-gemini-bv-objects
  python run_full_batch.py status
  python run_full_batch.py collect --bucket hs-hs-langcog-gemini-bv-objects
  python run_full_batch.py cleanup --bucket hs-hs-langcog-gemini-bv-objects --yes
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gemini_pilot"))
from run_gold_pilot import DETECT_OPEN_FORMAT_PROMPT, DETECT_SYSTEM, FORMATS, make_client  # noqa: E402

FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")
PRIV = HERE / "out" / "private"
MODEL = "gemini-2.5-flash"
MAX_OBJ, MAX_PX, STEP = 30, 910, 5
MAX_OUTPUT_TOKENS = max(4096, 200 * MAX_OBJ)   # pilot used 3,072; its 5 broken-JSON replies look truncated
EXCLUDE_SUBJECTS = {"S00270001"}
PRICES = {"gemini-2.5-flash": (0.30, 2.50)}    # per 1 M tokens, online; batch is half
GCS = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"


# ---------- request ----------
def schema() -> dict:
    """Vertex response schema matching the pilot's pydantic model for detect_open_format."""
    obj = {"type": "OBJECT",
           "properties": {"label": {"type": "STRING"}, "format": {"type": "STRING", "enum": FORMATS},
                          "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
           "required": ["label", "format", "box_2d"], "propertyOrdering": ["label", "format", "box_2d"]}
    return {"type": "OBJECT", "properties": {"objects": {"type": "ARRAY", "items": obj, "maxItems": MAX_OBJ}},
            "required": ["objects"]}


def generation_config() -> dict:
    return {"temperature": 0.0, "maxOutputTokens": MAX_OUTPUT_TOKENS, "responseMimeType": "application/json",
            "responseSchema": schema(), "thinkingConfig": {"thinkingBudget": 0}}


def request(uri: str) -> dict:
    """One batch line (Vertex GenerateContentRequest), image part first as in the pilot."""
    return {"request": {
        "contents": [{"role": "user", "parts": [{"fileData": {"fileUri": uri, "mimeType": "image/jpeg"}},
                                                 {"text": DETECT_OPEN_FORMAT_PROMPT.format(max_obj=MAX_OBJ)}]}],
        "systemInstruction": {"parts": [{"text": DETECT_SYSTEM}]},
        "generationConfig": generation_config()}}


def jpeg(path: str) -> bytes:
    """Frame bytes as the pilot sent them: unchanged unless the long edge exceeds MAX_PX."""
    raw = open(path, "rb").read()
    im = Image.open(io.BytesIO(raw))
    if max(im.size) <= MAX_PX:
        return raw
    s = MAX_PX / max(im.size)
    im = im.convert("RGB").resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------- Cloud Storage (JSON API with the service-account key; no extra packages) ----------
def gcs_session():
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_write"])
    return AuthorizedSession(creds)


def gcs_list(s, bucket: str, prefix: str) -> set[str]:
    names, token = set(), None
    while True:
        r = s.get(GCS.format(bucket=bucket), params={"prefix": prefix, "fields": "items(name),nextPageToken",
                                                     "maxResults": 1000, **({"pageToken": token} if token else {})})
        r.raise_for_status()
        d = r.json()
        names |= {i["name"] for i in d.get("items", [])}
        token = d.get("nextPageToken")
        if not token:
            return names


def gcs_put(s, bucket: str, name: str, data: bytes, ctype: str):
    for attempt in range(6):
        r = s.post(GCS_UPLOAD.format(bucket=bucket), params={"uploadType": "media", "name": name},
                   data=data, headers={"Content-Type": ctype})
        if r.status_code < 300:
            return
        if r.status_code not in (408, 429, 500, 502, 503, 504) or attempt == 5:
            r.raise_for_status()
        time.sleep(min(60, 2 ** attempt))


def gcs_get(s, bucket: str, name: str) -> bytes:
    r = s.get(f"{GCS.format(bucket=bucket)}/{quote(name, safe='')}", params={"alt": "media"})
    r.raise_for_status()
    return r.content


def gcs_delete(s, bucket: str, name: str):
    r = s.delete(f"{GCS.format(bucket=bucket)}/{quote(name, safe='')}")
    if r.status_code not in (204, 404):
        r.raise_for_status()


# ---------- stages ----------
def stage_manifest(args):
    rows = []
    for v in sorted(os.listdir(FRAMES)):
        subj = v.split("_")[0]
        if not subj.startswith("S00") or subj in EXCLUDE_SUBJECTS:
            continue
        nums = sorted(int(f[:-4]) for f in os.listdir(FRAMES / v) if f.endswith(".jpg") and f[:-4].isdigit())
        rows += [(f"{v}/{n:05d}", v, n, str(FRAMES / v / f"{n:05d}.jpg")) for n in nums[::STEP]]
    m = pd.DataFrame(rows, columns=["unit_id", "video_id", "frame", "frame_path"])
    PRIV.mkdir(parents=True, exist_ok=True)
    m.to_csv(PRIV / "manifest.csv", index=False)
    print(f"manifest: {len(m):,} frames from {m.video_id.nunique():,} videos -> {PRIV / 'manifest.csv'}")


def stage_prepare(args):
    m = pd.read_csv(PRIV / "manifest.csv")
    if args.retry:
        failed = set((PRIV / "failed_units.txt").read_text().split())
        m = m[m.unit_id.isin(failed)]
        tag = f"retry{int(time.time())}"
    else:
        tag = args.tag
    if args.limit:
        m = m.sample(args.limit, random_state=20261021) if args.limit < len(m) else m
    d = PRIV / "requests"
    d.mkdir(parents=True, exist_ok=True)
    uri = f"gs://{args.bucket or 'BUCKET'}/frames/{{}}.jpg"
    n = 0
    for i in range(0, len(m), args.shard_size):
        part = m.iloc[i:i + args.shard_size]
        name = f"shard_{tag}_{i // args.shard_size:03d}.jsonl" if tag else f"shard_{i // args.shard_size:03d}.jsonl"
        with open(d / name, "w") as f:
            for u in part.unit_id:
                f.write(json.dumps(request(uri.format(u))) + "\n")
        n += 1
    print(f"prepare: {len(m):,} requests in {n} shards -> {d} (bucket placeholder is rewritten at upload)")


def stage_smoke(args):
    from google.genai import types
    client = make_client()
    m = pd.read_csv(PRIV / "manifest.csv").sample(args.n, random_state=1)
    req = request("gs://unused")["request"]
    cfg = types.GenerateContentConfig(
        system_instruction=DETECT_SYSTEM, response_mime_type="application/json",
        response_schema=types.Schema.model_validate(req["generationConfig"]["responseSchema"]),
        temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0))
    prompt = req["contents"][0]["parts"][1]["text"]
    for r in m.itertuples():
        resp = client.models.generate_content(
            model=MODEL, config=cfg,
            contents=[types.Part.from_bytes(data=jpeg(r.frame_path), mime_type="image/jpeg"), prompt])
        objs = json.loads(resp.text)["objects"]
        u = resp.usage_metadata
        print(f"{r.unit_id}: {len(objs)} objects, tokens in/out {u.prompt_token_count}/{u.candidates_token_count} | "
              + ", ".join(f"{o['label']} ({o['format']})" for o in objs[:8]))


def stage_upload(args):
    s = gcs_session()
    m = pd.read_csv(PRIV / "manifest.csv")
    shards = sorted((PRIV / "requests").glob("shard_*.jsonl"))
    wanted = set()
    for p in shards:                     # upload only frames that some prepared shard asks for
        for line in open(p):
            uri = json.loads(line)["request"]["contents"][0]["parts"][0]["fileData"]["fileUri"]
            wanted.add(uri.split("/frames/", 1)[1][:-4])
    have = {n[len("frames/"):-4] for n in gcs_list(s, args.bucket, "frames/")}
    todo = m[m.unit_id.isin(wanted - have)]
    print(f"upload: {len(wanted):,} frames needed, {len(have & wanted):,} already in the bucket, {len(todo):,} to upload")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(gcs_put, s, args.bucket, f"frames/{r.unit_id}.jpg", jpeg(r.frame_path), "image/jpeg")
                for r in todo.itertuples()]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 5000 == 0 or done == len(futs):
                print(f"  {done:,}/{len(futs):,} frames {done / (time.time() - t0):.0f}/s", flush=True)
    for p in shards:                     # shards carry the real bucket name
        text = p.read_text().replace("gs://BUCKET/", f"gs://{args.bucket}/")
        gcs_put(s, args.bucket, f"requests/{p.name}", text.encode(), "application/jsonl")
    print(f"upload: {len(shards)} request shards -> gs://{args.bucket}/requests/")


def load_jobs() -> dict:
    p = PRIV / "jobs.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_jobs(jobs: dict):
    (PRIV / "jobs.json").write_text(json.dumps(jobs, indent=1) + "\n")


ACTIVE = {"JOB_STATE_PENDING", "JOB_STATE_QUEUED", "JOB_STATE_RUNNING", "JOB_STATE_UPDATING", "JOB_STATE_CANCELLING"}


def refresh(client, jobs: dict) -> dict:
    for shard, j in jobs.items():
        if j["state"] in ACTIVE or j["state"] == "submitted":
            b = client.batches.get(name=j["name"])
            j["state"] = str(b.state.value if hasattr(b.state, "value") else b.state)
            j["dest"] = b.dest.gcs_uri if b.dest else j.get("dest")
            if b.completion_stats:
                j["stats"] = b.completion_stats.model_dump(exclude_none=True)
            if b.error:
                j["error"] = str(b.error)[:300]
    return jobs


def stage_submit(args):
    from google.genai import types
    client = make_client()
    s = gcs_session()
    uploaded = sorted(n.split("/", 1)[1] for n in gcs_list(s, args.bucket, "requests/"))
    jobs = refresh(client, load_jobs())
    active = sum(j["state"] in ACTIVE or j["state"] == "submitted" for j in jobs.values())
    for shard in uploaded:
        if shard in jobs and jobs[shard]["state"] not in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            continue
        if active >= args.max_active:
            break
        b = client.batches.create(
            model=MODEL, src=f"gs://{args.bucket}/requests/{shard}",
            config=types.CreateBatchJobConfig(display_name=f"bv-objects-{Path(shard).stem}",
                                              dest=f"gs://{args.bucket}/outputs/{Path(shard).stem}/"))
        jobs[shard] = {"name": b.name, "state": "submitted", "submitted": time.strftime("%Y-%m-%d %H:%M")}
        active += 1
        print(f"submitted {shard}: {b.name}")
    save_jobs(jobs)
    print(f"{active} jobs active (max {args.max_active}); {sum(s not in jobs for s in uploaded)} shards waiting")


def stage_status(args):
    jobs = refresh(make_client(), load_jobs())
    save_jobs(jobs)
    for shard, j in sorted(jobs.items()):
        print(f"{shard:28s} {j['state']:28s} {j.get('stats', '')} {j.get('error', '')}")


def stage_collect(args):
    s = gcs_session()
    jobs = refresh(make_client(), load_jobs())
    save_jobs(jobs)
    out_dir = PRIV / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard, j in jobs.items():
        if j["state"] not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED") or j.get("downloaded"):
            continue
        prefix = j["dest"].split(f"gs://{args.bucket}/", 1)[1]
        for name in sorted(n for n in gcs_list(s, args.bucket, prefix) if n.endswith(".jsonl")):
            (out_dir / f"{Path(shard).stem}__{Path(name).parent.name}__{Path(name).name}").write_bytes(
                gcs_get(s, args.bucket, name))
        j["downloaded"] = True
    save_jobs(jobs)

    meta = {"model": MODEL, "task": "detect_open_format", "pass": 1, "endpoint": "vertexai/batch",
            "temperature": 0.0, "thinking": "budget0", "max_obj": MAX_OBJ, "max_px": MAX_PX}
    recs = {}
    for p in sorted(out_dir.glob("*.jsonl")):          # later files (retries) overwrite earlier failures
        for line in open(p):
            d = json.loads(line)
            uri = d["request"]["contents"][0]["parts"][0]["fileData"]["fileUri"]
            uid = uri.split("/frames/", 1)[1][:-4]
            rec = {"unit_id": uid, **meta, "error": None}
            try:
                resp = d.get("response") or {}
                cand = (resp.get("candidates") or [{}])[0]
                text = "".join(pt.get("text", "") for pt in cand.get("content", {}).get("parts", []))
                u = resp.get("usageMetadata", {})
                rec.update(raw=text, in_tok=u.get("promptTokenCount", 0), out_tok=u.get("candidatesTokenCount", 0),
                           think_tok=u.get("thoughtsTokenCount", 0), finish=cand.get("finishReason"),
                           model_version=resp.get("modelVersion"))
                if d.get("status"):
                    raise RuntimeError(d["status"])
                json.loads(text)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            if uid not in recs or recs[uid]["error"] is not None:
                recs[uid] = rec
    with open(PRIV / "results.jsonl", "w") as f:
        for r in recs.values():
            f.write(json.dumps(r) + "\n")
    m = pd.read_csv(PRIV / "manifest.csv")
    ok = [r for r in recs.values() if r["error"] is None]
    failed = sorted(set(m.unit_id) - {r["unit_id"] for r in ok})
    (PRIV / "failed_units.txt").write_text("\n".join(failed) + ("\n" if failed else ""))
    pi, po = PRICES[MODEL]
    tin, tout = sum(r["in_tok"] for r in ok), sum(r["out_tok"] for r in ok)
    usage = {**meta, "date": time.strftime("%Y-%m-%d"), "units": len(m), "scored": len(ok),
             "failures": len(recs) - len(ok), "missing": len(set(m.unit_id) - set(recs)),
             "input_tokens": tin, "output_tokens": tout, "thinking_tokens": sum(r["think_tok"] for r in ok),
             "cost_batch_usd": round((tin * pi + tout * po) / 1e6 / 2, 2)}
    (PRIV / "usage.json").write_text(json.dumps(usage, indent=1) + "\n")
    print(json.dumps(usage, indent=1))
    print(f"{len(failed):,} units not scored -> failed_units.txt (prepare --retry to resubmit)")


def stage_cleanup(args):
    if not args.yes:
        sys.exit("cleanup deletes gs://BUCKET/frames/ and requests/; rerun with --yes once results are checked")
    s = gcs_session()
    names = sorted(gcs_list(s, args.bucket, "frames/") | gcs_list(s, args.bucket, "requests/"))
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda n: gcs_delete(s, args.bucket, n), names))
    print(f"deleted {len(names):,} objects from gs://{args.bucket} (outputs/ kept)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["manifest", "prepare", "smoke", "upload", "submit", "status", "collect", "cleanup"])
    ap.add_argument("--bucket", help="bucket name, without gs://")
    ap.add_argument("--shard-size", type=int, default=50000, help="requests per batch job")
    ap.add_argument("--limit", type=int, default=0, help="prepare: random N frames only (e.g. a pilot job)")
    ap.add_argument("--tag", default="", help="prepare: shard name tag, e.g. pilot")
    ap.add_argument("--retry", action="store_true", help="prepare: one shard set of the failed units")
    ap.add_argument("--n", type=int, default=5, help="smoke: requests to send online")
    ap.add_argument("--workers", type=int, default=32, help="upload / cleanup threads")
    ap.add_argument("--max-active", type=int, default=4, help="submit: batch jobs running at once")
    ap.add_argument("--yes", action="store_true", help="cleanup: confirm deletion")
    args = ap.parse_args()
    if args.stage in ("upload", "submit", "collect", "cleanup") and not args.bucket:
        sys.exit(f"{args.stage} needs --bucket")
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
