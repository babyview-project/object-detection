# Gemini model pilot on the gold crops

Decides which Gemini model to use for the BabyView frame annotation run: the lab
default `gemini-2.5-flash`, or the cheaper `gemini-2.5-flash-lite`
(`gemini-3.1-flash-lite` was dropped: on this project it is served only from the
`global` endpoint, which does not keep data in a US region). The lab README reports that Flash-Lite "piles at one
value" on the 0–100 alignment score; this tests whether that matters for object
labels. Quoted cost of the recommended full run (2025.2 at 1 frame per 5 s,
CDI + open-vocab, batch): $145 on 2.5 Flash-Lite, $537 on 3.1 Flash-Lite
(high-resolution token counts; default resolution is cheaper).

Set up and run following the lab guide ("Annotating BabyView with Gemini (via
Vertex AI)") and `/home/mcfrank/bv-annotations/alignment/SETUP.md`; the code
follows `alignment/gemini_align.py`.

## Gold set

`ccn-2026-object-variability/data/shared_data_ccn_2026/valid5921_public`: 5,921
YOLOE crops in 72 CDI categories, each judged correct by raters (body parts and
glasses removed; opaque crop ids). Read directly from the zip; nothing is unpacked.
Every crop is a true positive, so the pilot measures recognition with two tasks:

| task | what Gemini sees | measures |
|---|---|---|
| `name` | crop + the 295-label YOLOE/CDI vocabulary | strict accuracy (exact word), lenient accuracy (near-synonym counts, `vocab_groups.py`), domain accuracy, "none" rate |
| `verify` | crop + 3 shuffled words: gold, a same-domain foil, an other-domain foil | hit rate, false alarms within and across domains |
| `detect` | the whole source frame (5,919 frames), no YOLOE input; returns every vocabulary object with a box (`box_2d`, 0–1000) | found (same/near-synonym label and IoU ≥ 0.5 with the gold box), label found anywhere, box found with any label, best IoU, objects per frame; by gold-object size |

Foils are fixed per crop (seeded) and never a general word (animal, toy, food…)
or a near-synonym of the gold label. Same-domain foils can still be truly visible
in a crop (a chair next to a table), so `fa_same` has a nonzero floor.

Settings, as in `gemini_align.py`: system instruction + terse prompt, JSON
response schema, one call per crop, temperature 0, thinking off
(`thinking_budget=0`), longest image edge downsized to 512 px, default media
resolution, 32 workers, output cap 512 tokens (3,072 for `detect`). Words are plain
strings, not an enum: Vertex rejects an enum of 150+ values (the vocabulary has 295),
so the scorer counts off-list answers (`off_list`). `detect` lists are capped at 25
objects in the schema (`maxItems`), after Flash-Lite once repeated boxes until the
token limit. Reliability: a second pass on 8 crops per category (576).

## Decision rule (set before running)

Use the cheapest model for which all of these hold, compared with `gemini-2.5-flash`:

1. Lenient naming accuracy is no more than 3 points below (point estimate), with
   the upper end of the 95% category-bootstrap CI of the gap below 5 points.
2. Domain accuracy is no more than 2 points below. Cross-domain errors are the
   ones that would bias the superordinate-structure analyses.
3. Other-domain false alarms ≤ 5%, and pass-1 vs pass-2 agreement ≥ 90%.

## Setup (once)

1. Get a service-account key scoped to `roles/aiplatform.user` on
   `hs-hs-langcog-gemini` from Mike (SETUP.md has his commands). Not an API key:
   the runner refuses to start if `GOOGLE_API_KEY`/`GEMINI_API_KEY` is set.
2. On ccn2: `mkdir -p ~/.secrets && chmod 700 ~/.secrets`; put the key there;
   `chmod 600 ~/.secrets/*`.
3. `~/.secrets/hs-hs-langcog-gemini.env`:
   ```bash
   export GOOGLE_GENAI_USE_VERTEXAI=true
   export GOOGLE_CLOUD_PROJECT=hs-hs-langcog-gemini
   export GOOGLE_CLOUD_LOCATION=us-central1
   export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.secrets/<your-key>.json
   ```
4. Python: `~/.venvs/gemini` (google-genai, pillow, pandas, pyarrow; built on the
   vislearnlabpy env).
5. Revoke the key when the runs are done (`gcloud iam service-accounts keys delete …`).

## Running

```bash
PY=~/.venvs/gemini/bin/python
$PY run_gold_pilot.py --dry-run                                  # prompts, no calls
set -a; source ~/.secrets/hs-hs-langcog-gemini.env; set +a
$PY run_gold_pilot.py --list-models                              # confirm model ids
for m in gemini-2.5-flash-lite gemini-2.5-flash; do
  $PY run_gold_pilot.py --model $m --task name --sample 200      # sanity pass: look at the distribution
done
setsid nohup bash run_all.sh > logs/run_all.log 2>&1 &           # full pilot, detached; DONE marker
wc -l out/*.jsonl                                                # progress
```

Tell the lab before launching (quota is per project). Estimated pilot cost:
under $20 on demand, billed to the grant project (about 58k calls; most of the
cost is 2.5 Flash, and `detect` is about half of it because of its longer output).

Outputs in `out/`: `<model>__<task>__p<pass>.jsonl` (checkpoint, one line per
crop; failures retried on relaunch), the consolidated `.parquet`,
`usage_<run>.json` (model, endpoint, date, tokens, failures, model versions),
`gold_manifest.csv` (crops, foils, word order). `score_gold_pilot.py` writes `results/`.

## Source frames for the gold crops

`build_gold_frames.py` maps each public crop back to its 868-hour frame
(`/ccn2a/dataset/babyview/868_hours/sampled_frames`, 512×910) and its YOLOE box
(`/ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_allframes_1fps/<video>/bounding_box_predictions.csv`),
via the original stems in the non-public valid7018 embeddings manifest. All
5,921 crops matched (5,919 frames, 3,375 videos, 79 subjects; box size within
2 px of the crop; pixels match on a 40-crop check). Output:
`out/private/gold_frames.csv`, git-ignored because it contains subject ids and dates.

## Limits

- `detect` measures recall of rater-validated objects only. Only one object per
  frame was validated, so Gemini's other boxes cannot be scored as right or wrong
  here; the full frames Mira and Dora boxed (`annotation/cvat_3000_frames/annotation_data/`) are the
  test for that.
- Frames are sent at 288×512 (the guide's 512 px rule), which shrinks small
  objects: a 32 px object becomes about 18 px. If small-object recall decides the
  model, rerun `detect` on a subset with `--max-px 910` (full size).
- `detect` outputs are in `out/private/` (git-ignored) because they are keyed by
  video names.
- The gold frames are keyed by the older 868-hour video names. The main run should
  be keyed by 2025.2 release names (`video_id`, `frame_idx`) and built from release
  files, per the guide's checklist, with outputs in
  `/ccn2b/dataset/babyview/2025.2/outputs/annotations/<layer>/` plus a README.
