# CLIP-band annotation: false rejections and scene context

Reviewers asked us to justify the CLIP threshold (we used 0.26, 0.27, 0.28). CLIP
scores sit in a narrow range: over all 13.2 M scored YOLOE crops in 2025.2 the middle
90% lie between 0.219 and 0.299 (SD 0.025), a quarter of crops fall between 0.26 and
0.28, and the pass rate drops from 43% at 0.26 to 29% at 0.27 to 18% at 0.28. A choice
inside that band needs human labels: precision and false-rejection rate at each
threshold.

The second question is context. CLIP scores the tight crop only, so it cannot use the
scene. Raters therefore judge each crop twice: alone (what CLIP sees), then with the
full frame and the box drawn. Objects that are only recognisable in context, and how
CLIP scored them, test whether the filter drops true detections for lack of context.

## Sample

| | |
|---|---|
| Population | every scored YOLOE crop in 2025.2, valid129, person/picture and subject 00270001 excluded (`crops_index.csv.gz`) |
| Strata | 129 categories × 7 CLIP bands: < 0.22, 0.22–0.24, 0.24–0.26, 0.26–0.27, 0.27–0.28, 0.28–0.30, ≥ 0.30 |
| Per stratum | 8 crops, round-robin over children, then videos; one crop per frame |
| Total | 7,155 items (902 of 903 cells; 17 cells have < 8 crops), 22 children, 2,362 videos |
| Weights | `band_population.csv`: population count per category × band |

Band edges fall on every threshold we used, so precision and recall can be computed at
0.26, 0.27, 0.28 or any other band edge. Estimates must weight each band by its
population share (bands differ in size by up to 6×).

## Steps

```bash
nice -n 19 python sample_clip_bands.py      # items.csv, band_population.csv
nice -n 19 python render_items.py           # images/<item_id>_crop.jpg, _frame.jpg
python export_gallerize.py --n-buckets 3    # gallerize/annotations.csv + img dirs
```

All outputs go to `out/private/` (git-ignored): items hold subject ids and frame paths,
and frames show family members' faces and homes.

## Annotation front ends

**objects-gallerize** (preferred; github.com/babyview-project/objects-gallerize, the
tool used for the 868 h precision study). One page per category, 15–19 crops plus an
attention-check tile; raters click invalid crops. 3 buckets × 129 pages, about 2,400
crops per rater; store.js caps each bucket at 3 workers, so each item gets 3 ratings.
File names are opaque item ids, so no subject ids or CLIP scores reach the browser.
The context step is `gallerize_context_phase.patch` (against objects-gallerize master
0a6bb7f; `git apply gallerize_context_phase.patch` in that repo). With it, after each
category's crop page (with its attention check) the same items come back as cards
showing the crop beside its frame with the box drawn (click to enlarge), in a new
order, unmarked, and are judged again. Changes:
  - `Trial.js` / `SingleCard.js`: frame step, `ContextCard`; every response carries
    `phase` (`crop` / `frame`); each finished step posts a `page_complete` row with the
    item list, so "nothing marked invalid" is recorded. The last category's clicks are
    now posted before the Prolific redirect (the original code dropped them).
  - `store.js`: `FRAME_DIR` serves `/db/frames/*` behind the same API key; items get a
    `frame_url`; `NUM_BUCKETS` comes from the env; image routes use `sendFile(..., {root})`
    so `../` requests are refused.
  - `response.model.js`: `phase`, `n_items`, `items` fields (Mongo drops unknown fields).
Run with `REACT_APP_CONTEXT_PHASE=1` at build time and, for store.js,
`DATA_SOURCE=local_csv`, `DATA_CSV_PATH=out/private/gallerize/annotations.csv`,
`IMG_DIR=out/private/gallerize/img`, `FRAME_DIR=out/private/gallerize/img_frame`,
`NUM_BUCKETS=3`. Without the build flag the study behaves as before.
Tested end to end (2026-10-17): local build, store.js on an in-memory MongoDB 6.0,
headless Chromium through consent, practice, two categories (crop + frame) and submit;
the expected crop/frame/page_complete rows were stored, `../` returned 403 and a frame
request without the key 401. Practice and check-tile images are not in the public repo
(git-ignored); the test used placeholders.

**annotate_crop_context.py** (fallback for lab raters on the server; tkinter). Same
two steps per item, plus "can't tell" and a "what is it?" label from the CDI list.
Uses `assignments_<rater>.csv` from the sampler (2 raters, 10% double-coded). Not yet
tested end to end.

## Open decisions

- Who sees full frames. Crops went to Prolific for the 868 h study; frames show faces
  and home interiors, so the frame phase may need lab raters only, or IRB sign-off.
- The 868 h study excluded body parts; this sample keeps them (valid129).

## CLIP vs Gemini (before human labels)

`run_gemini_verify.py` asks Gemini 2.5 Flash-Lite (Vertex) the gold-pilot verify question for every
item, from the crop alone and from crop + frame (box drawn), with two foil words; both conditions use
`MEDIUM` media resolution so the crop is seen identically. `compare_clip_gemini.py` writes aggregate
tables to `results/` (by band, by cut-off with population weights, by category, and `vs_human.csv`
once `out/private/human_labels.csv` exists). `build_clip_vs_gemini_nb.py` writes
`clip_vs_gemini.ipynb` (code only); it shows crops and frames, so execute it only into
`out/private/`:

```bash
jupyter nbconvert --to notebook --execute clip_vs_gemini.ipynb \
  --ExecutePreprocessor.kernel_name=vislearnlabpy --output-dir out/private \
  --output clip_vs_gemini_executed.ipynb
```

Figures (aggregate, no images) go to `results/fig_*.pdf`; montages to `out/private/montages/`.
