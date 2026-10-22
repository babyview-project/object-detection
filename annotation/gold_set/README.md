# Gold set v2: BV-main frames, boxes corrected by RAs

v2 replaces the CCN 2026 gold set (v1) for the revision. RAs correct pre-filled boxes on
whole frames, add the objects nobody found, and give every box an open-vocabulary name and
a format. Every discrete object is boxed, not only the CDI / valid129 categories (the CDI
list missed about 40% of the objects Gemini found: cabinets, shelves, rugs, curtains,
mirrors, bags, ...), with no limit on boxes per frame. One annotation pass
serves three TODOs:

| What RAs produce | Used for |
|---|---|
| corrected and added boxes on whole frames | precision **and recall** of YOLOE+CLIP vs Gemini on the same frames (the explicit comparison of the two detection methods) |
| a format for every box: real / toy / picture / screen | validating Gemini's format labels (round 2: picture-vs-real table, BV vs VEDB depictions) |
| every corrected box | gold set v2 (BV-main only) |
| open names for all discrete objects | how much of the view the CDI list covers; names mapped to categories after annotation |

All outputs are in `out/private/` (git-ignored): frames show children's homes and family
members, and the tables hold subject ids and dates. Share task bundles only through
IRB-approved storage; CVAT runs locally.

## Steps

```bash
python freeze_v1.py                         # v1 frozen copy + its BV-main subset
python sample_frames.py --raters Mira Emma   # 1,000 frames -> out/private/frames.csv
# Gemini pre-fill (Vertex, gemini-2.5-flash, open label + format, ~$2.40 for 1,000 frames)
cd ../gemini_pilot && set -a && source ~/.secrets/hs-hs-langcog-gemini.env && set +a
~/.venvs/gemini/bin/python run_gold_pilot.py --model gemini-2.5-flash --task detect_open_format \
    --max-obj 30 --frames ../gold_set/out/private/gold_v2_frames.csv --max-px 910 --workers 8
cd ../gold_set
# YOLOE+CLIP pre-fill (production settings; YOLOE needs a YOLOE build of ultralytics)
(cd ../../yoloe && CUDA_VISIBLE_DEVICES=<gpu> /data/tanawm/miniconda3/envs/yoloe/bin/python \
    ../annotation/gold_set/prefill_yoloe_clip.py --stage yoloe)
CUDA_VISIBLE_DEVICES=<gpu> python prefill_yoloe_clip.py --stage clip   # vislearnlabpy env
python prefill_yoloe_clip.py --stage check
python build_cvat_tasks.py --zip            # one CVAT task per rater and batch
```

## Gold set v1 and what was dropped

v1 = `valid5921_public` (5,921 rater-validated YOLOE crops, 72 categories) + its frame
manifest. `out/private/v1_original/` is a byte-for-byte, read-only copy with checksums.

BV-Bing vs BV-main comes from the raw-video storage, `/ccn2/dataset/babyview/unzip*/
babyview_{bing,main}_storage`, where no subject appears in both. Subjects `00xx` are main
(home); `01xx` and `02xx` are Bing (preschool). v1 had 1,027 Bing crops from 48 children.
Four crops come from videos named `2023_<n>_<hash>`; their GoPro sidecar files name
`00270001`, which is excluded. **v1 BV-main = 4,890 crops, 31 children, 72 categories**
(`out/private/v1_bvmain.csv`).

v1 and v2 rows differ in kind: a v1 row is one YOLOE detection per frame that raters
confirmed (only YOLOE positives could enter), a v2 frame has every object boxed. Use both
for precision and format; use v2 frames alone for recall and method comparisons.

## Sample (`sample_frames.py`)

- Population: 1 fps frames of BV-main videos in 2025.2, excluding S00270001, videos under
  30 s, the first and last 5 s of each video, and frames within 10 s of a v1 frame in the
  same session. 7,519 videos from 35 children.
- 800 **random** frames: child quotas proportional to the square root of the child's video
  count (at least 2), one frame per video, videos spread over each child's date-sorted
  recordings, frame uniform within the video. Use this stratum alone for prevalence and recall.
- 200 **depiction** frames: from 10 s clips whose context label is reading, watching tv,
  looking at device or drawing (`merged_contexts.csv`, not detector output), same rules,
  different videos. Adds books, screens and drawings for checking formats.
- Every frame has its inclusion probability and weight.
- Result: 1,000 frames from 1,000 videos, 35 children (4-61 frames each), 5.7-36.6 months.
- 100 random-stratum frames (34 children) go to both RAs: agreement, and the double-coded
  set for the method comparison. The rest alternate between RAs within each child.
  Batch 1 = 100 shared + 100 own per RA; later batches are 100 frames.

## Pre-fill

- **YOLOE+CLIP**: production 2025.2 YOLOE output covers only videos that are not in the
  868-hour release (those were run under 868-hour names, with frames extracted
  separately), i.e. 342 of the sampled frames. `prefill_yoloe_clip.py` reruns YOLOE and
  CLIP with the production settings on all 1,000. Check against production on the 314
  covered frames with boxes: 97.8% of production boxes matched (same class, IoU ≥ 0.9),
  CLIP score difference median 0.00006 (max 0.01), 1 pass/fail flip at 0.27.
- **Gemini**: 2.5 Flash `detect_open_format` (box, open label, format; no CDI word), full
  frame at 512×910, temperature 0, thinking off, at most 30 objects per frame
  (`--max-obj 30`). Chosen over `detect_combo` (the same prompt plus the closest CDI word)
  after the side-by-side check below: more specific names, fewer generic "toy" labels.
  Earlier `detect_combo` runs used a cap of 25 (207 of 1,000 frames hit it). A run with no cap
  (`detect_combo_nocap`, tested on 80 frames) was dropped: "at most N" seems to work as a
  target, so without it some busy frames came back with far fewer objects (25 → 5), and some
  looped until the token limit (138 "light" boxes; 2 of 20 broken JSON). All boxes are kept
  except labels that are not discrete objects or are parts (`NOT_DISCRETE`, `PARTS` in
  `build_cvat_tasks.py`); the box name is Gemini's open label.
- **Person**: YOLOE person boxes as output (production drops person before CLIP) and
  Gemini boxes labelled person.
- **Merge** (`build_cvat_tasks.py`): a Gemini box matches a YOLOE box when its label, or the
  label's last word ("toy car" → car), or person is the YOLOE class and IoU ≥ 0.5 (Gemini
  gives no CDI word in this setup, so the key under-matches synonyms such as "sofa" vs
  couch; the method comparison scores both detectors against RA boxes anyway) → one row with Gemini's name and both
  sets of coordinates (YOLOE in `xmin..`, Gemini in `g_xmin..`). All rows, with their
  source, are the key in `out/private/prefill/merged.csv`, for comparing RA boxes with
  YOLOE+CLIP (or any other model) post hoc.
- **What RAs see**: Gemini's boxes only, in Gemini's coordinates; YOLOE-only boxes are
  hidden (kept in `merged.csv`). The format attribute is pre-filled with Gemini's format
  (photo and drawing → picture), so RAs correct formats rather than label them from
  scratch; boxes RAs draw start as `unset`.
- **Anchoring**: RA boxes, names and formats are corrections of Gemini's, not independent
  labels. Agreement with Gemini is an upper bound, and the method comparison favours
  Gemini (an object both found is scored against a box drawn from Gemini's). Objects only
  YOLOE found count only if RAs add them. The double-coded shared frames give
  inter-rater agreement but do not remove this bias; report it in the comparison.
- **CVAT label setup**: one label, `object`, with a text attribute `name` and a select
  attribute `format`. RAs follow naming rules (lowercase, singular, CDI word when it fits);
  names are mapped to CDI / valid129 categories after annotation.

Pre-fill result (2026-10-21, `detect_open_format`, cap 30): merged key 13,282 boxes. By
source: Gemini only 10,550, YOLOE+CLIP only 1,078, both 632; person: YOLOE only 567, both 455.
Shown to RAs: 11,637 Gemini boxes, median 10 per frame (max 30), 26 frames with none, 1,131
distinct names; 6,062 are a valid129 class or person by label. Matched boxes: median IoU of
YOLOE vs Gemini box 0.87. Gemini returned 14,415 objects (median 12 per frame); 162 frames hit
the cap of 30, and 1,678 objects were dropped as not discrete or parts (mostly walls, floors,
ceilings). Gemini run: 2.06 M input / 0.71 M output tokens ($2.39; 5 broken-JSON responses
succeeded on retry). Pre-filled formats: real 9,572, toy 1,322, picture 631, screen 112.
Per RA: Emma 548 frames / 6,172 boxes, Mira 552 / 6,584, shared frames 1,119
(`out/private/ra_assignments.csv`). The `detect_combo` runs (cap 25, cap 30) and the 80-frame
no-cap test stay in `gemini_pilot/out/private/` but are not used.

## Pre-fill comparison (2026-10-21)

Three Gemini versions on the same 1,000 frames, to choose what RAs correct and which model to
use for the full run. **Chosen: B** (2.5 Flash, open label + format), for the RA pre-fill and
the likely full run. `compare_prefill.py` writes `out/private/compare_prefill/` (summary.csv,
side-by-side images of 60 frames, index.html).

| | A: 2.5 Flash, open + CDI + format | **B: 2.5 Flash, open + format (chosen)** | C: 2.5 Flash-Lite, open + format |
|---|---|---|---|
| boxes (median per frame) | 12,833 (11) | 11,643 (10) | 12,278 (9) |
| frames at the cap of 30 / empty | 56 / 39 | 40 / 26 | 129 / 11 |
| distinct names | 730 | 1,134 | 909 |
| real / toy / picture / screen | 85 / 9 / 6 / 1% | 82 / 11 / 5 / 1% | 63 / 27 / 9 / 2% |
| cost, 1,000 frames online | $2.89 | $2.39 | $0.55 |
| full run, BV-main 1 per 5 s, batch | $1,487 | $1,227 | $282 |

Flash-Lite names 3,505 objects just "toy" (before duplicates are removed), against 625 for B
and 1,210 for A; the CDI list (which contains "toy") seems to pull A toward the generic word.
One Flash-Lite frame returned broken JSON twice and counts as empty.

## CVAT

For each folder in `out/private/cvat/`: create a task, paste `labels.json` into the label
constructor (Raw), upload `images/`, then Actions → Upload annotations → "CVAT for images
1.1" → `annotations.zip`. Export finished tasks as "CVAT for images 1.1" to
`out/private/cvat_exports/<rater>_batch<k>.zip`.

## RA instructions

`RA_INSTRUCTIONS.md` (for Mira and Emma): privacy rules, CVAT install, task setup,
pre-fill upload, annotation and naming rules, export and hand-in.
