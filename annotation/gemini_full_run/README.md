# Full Gemini run on 2025.2 BV-main (Vertex batch)

Runs the gold set v2 pre-fill setup (setup B) over BabyView 2025.2 BV-main:
`gemini-2.5-flash`, `detect_open_format` (open label + format + box per object, at most 30),
full frame (512×910), temperature 0, thinking off, one frame every 5 s. Prompt, system
instruction and formats are imported from `../gemini_pilot/run_gold_pilot.py`. The output cap is
6,000 tokens (the pilot used 3,072; its 5 broken-JSON replies in 1,000 looked truncated).

| | |
|---|---|
| frames | every 5th frame of `/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps`, subjects S00xx, S00270001 excluded: **1,030,672 frames, 7,590 videos** (checked 2026-10-21: no empty folders, no gaps in frame numbering, no zero-byte files) |
| requests | 21 shards of 50,000 (about 84 MB each), one Vertex batch job per shard |
| cost | about $1,230 at batch prices (measured on the pilot: 2,059 input / 707 output tokens per frame) |
| bucket | `gs://hs-hs-langcog-gemini-bv-objects` (us-central1, grant project), requested from Mike |

## Run

```bash
set -a; source ~/.secrets/hs-hs-langcog-gemini.env; set +a
B=hs-hs-langcog-gemini-bv-objects
python run_full_batch.py manifest                        # done 2026-10-21
python run_full_batch.py prepare                         # done 2026-10-21 (21 shards)
python run_full_batch.py smoke --n 5                     # done 2026-10-21: prompt, schema and config work on Vertex

# once the bucket exists: a small pilot job first
python run_full_batch.py prepare --limit 500 --tag pilot   # writes shard_pilot_000.jsonl next to the full shards
python run_full_batch.py upload --bucket $B                # uploads frames for every local shard
python run_full_batch.py submit --bucket $B --max-active 4
python run_full_batch.py status                            # repeat until jobs finish (usually hours, up to 72 h)
python run_full_batch.py collect --bucket $B               # results.jsonl, usage.json, failed_units.txt
python run_full_batch.py prepare --retry                   # then upload, submit, collect again for failures
python run_full_batch.py cleanup --bucket $B --yes         # delete frames and requests once results are checked
```

To run the pilot job on its own, move the full `shard_0*.jsonl` files out of `out/private/requests/`
before `upload`, and put them back afterwards. `upload` sends only the frames that the local shards
ask for, and skips frames already in the bucket.

All state is in `out/private/` (git-ignored: frame ids carry subject and video ids). `results.jsonl`
has one record per frame in the pilot's format (`unit_id` = `<video_id>/<frame:05d>`, `raw` =
Gemini's JSON), so `gold_set/build_cvat_tasks.load_gemini` reads it; apply the same filters
(`NOT_DISCRETE`, `PARTS`, duplicates) when building the analysis tables.

## Data handling

Frames go to Gemini only through Vertex in the grant project (IRB-approved for annotation).
The bucket stores frames only while jobs run: `cleanup` deletes `frames/` and `requests/` and
keeps `outputs/` (model text, no images). Revoke the service-account key after the run.
Cloud Storage access uses the same service-account key through the JSON API, so no extra
packages are needed.
