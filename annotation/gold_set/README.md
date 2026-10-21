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
# Gemini pre-fill (Vertex, gemini-2.5-flash, ~$3-4 for 1,000 frames)
cd ../gemini_pilot && set -a && source ~/.secrets/hs-hs-langcog-gemini.env && set +a
~/.venvs/gemini/bin/python run_gold_pilot.py --model gemini-2.5-flash --task detect_combo \
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
- **Gemini**: 2.5 Flash `detect_combo` (box, open label, closest CDI word, format), full
  frame at 512×910, temperature 0, thinking off, at most 30 objects per frame
  (`--max-obj 30`). The first run used 25; 207 of 1,000 frames hit it. A run with no cap
  (`detect_combo_nocap`, tested on 80 frames) was dropped: "at most N" seems to work as a
  target, so without it some busy frames came back with far fewer objects (25 → 5), and some
  looped until the token limit (138 "light" boxes; 2 of 20 broken JSON). All boxes are kept
  except labels that are not discrete objects or are parts (`NOT_DISCRETE`, `PARTS` in
  `build_cvat_tasks.py`); the box name is Gemini's open label.
- **Person**: YOLOE person boxes as output (production drops person before CLIP) and
  Gemini boxes labelled person.
- **Merge** (`build_cvat_tasks.py`): a Gemini box matches a YOLOE box when its CDI word
  (or person) is the YOLOE class and IoU ≥ 0.5 → one box with Gemini's name. Sources are
  hidden from RAs (key in `out/private/prefill/merged.csv`). The format attribute is
  pre-filled with Gemini's format (photo and drawing → picture), so RAs correct formats
  rather than label them from scratch; YOLOE-only boxes and new boxes start as `unset`.
  RA formats are therefore corrections of Gemini's, not independent labels: agreement with
  Gemini is an upper bound (anchoring), so report it as such in the format validation.
- **CVAT label setup**: one label, `object`, with a text attribute `name` and a select
  attribute `format`. RAs follow naming rules (lowercase, singular, CDI word when it fits);
  names are mapped to CDI / valid129 categories after annotation.

Pre-fill result (2026-10-21, Gemini cap 30): 14,300 boxes, median 13 per frame (max 39),
30 frames with none, 739 distinct names; 9,738 boxes are a valid129 class or person. By
source: Gemini only 11,568, YOLOE+CLIP only 937, both 773; person: YOLOE only 530, both 492.
Gemini returned 16,081 objects (median 14 per frame); 247 frames still hit the cap of 30,
and 1,138 objects were dropped as not discrete or parts. Gemini run: 2.71 M input / 0.83 M
output tokens (about $3). Pre-filled formats: real 10,887, toy 1,099, picture 744, screen
103, unset 1,467 (the YOLOE-only boxes). The earlier cap-25 run and the 80-frame no-cap test stay in
`gemini_pilot/out/private/` but are not used.

## CVAT

For each folder in `out/private/cvat/`: create a task, paste `labels.json` into the label
constructor (Raw), upload `images/`, then Actions → Upload annotations → "CVAT for images
1.1" → `annotations.zip`. Export finished tasks as "CVAT for images 1.1" to
`out/private/cvat_exports/<rater>_batch<k>.zip`.

## RA instructions

`RA_INSTRUCTIONS.md` (for Mira and Emma): privacy rules, CVAT install, task setup,
pre-fill upload, annotation and naming rules, export and hand-in.
