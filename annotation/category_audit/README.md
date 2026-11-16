# category_audit/: which categories does Qwen name correctly, in BabyView and in VEDB?

The 1,000-frame gold set (`annotation/gold_set/`) covers BabyView only, and most heavily depicted categories have
fewer than 5 boxes in it (128 of the 182 core categories). This audit checks those categories directly, in both
datasets, so categories Qwen names poorly can be excluded by a rule fixed before rating.

## Design (decided 2026-11-15)

| | |
|---|---|
| Categories | 53: every core category ≥ 30% depicted in BabyView or VEDB, plus the weakest core categories in the gold set (cat, teddy bear, coat, fish, pig, car) |
| Sample | 30 Qwen boxes per category and dataset (3,180), random over all boxes (no size or person filter), spread over children / sessions (round-robin over units, one box per video first); VEDB adults only |
| Raters | Mira and Emma; each item to one rater, 20% (stratified by dataset) to both for agreement; about 1,900 items each on 106 pages |
| Task | per page (one category, BabyView and VEDB mixed, dataset not shown): step 1, crop alone: click every box that is **not** the category (other object, empty, too unclear); step 2, the same boxes beside their frames: click every box that is **depicted** (toy, picture, screen) |
| Measures | per category × dataset: name precision (Wilson 95% CI); share depicted among correctly named boxes, by the raters and by Qwen's label; real-vs-depicted agreement with Qwen; Cohen's κ per step on shared items |
| Rule | **exclude a category if its name precision is below 0.70 in BabyView or in VEDB**; rerun the analyses with and without the excluded categories (sensitivity check) |

With 30 boxes, a precision of 0.70 has a 95% CI of about ±0.16: enough to catch clearly broken categories, not to
rank borderline ones.

## Steps

```bash
cd annotation/category_audit
/data/j7yang/anaconda3/bin/python sample_audit.py                         # out/private/items.csv + img/ (crops, frames)
/data/j7yang/anaconda3/bin/python make_audit_bundles.py --raters Mira Emma --zip   # out/private/bundles/bundle_<RA>.zip
# RAs: open web/index.html (or the hosted copy), choose the unzipped bundle folder, export answers
# put the exports in out/private/responses/, then
/data/j7yang/anaconda3/bin/python score_audit.py                          # results/audit_by_category.csv, audit_agreement.csv,
                                                                           # excluded_categories.txt
```

Bundles hold frames of children's homes and families: share them only through IRB-approved storage (never GitHub,
Slack or email). `out/` is git-ignored; `results/` holds aggregate tables only.

## Web page (`web/`)

Adapted from the object-check page of the dropped CLIP-band study (`annotation/clip_band_annotation/web`, git
`1a4a8ee`): static HTML/CSS/JS, the Content-Security-Policy forbids every network request, images load from the RA's
local folder as `blob:` URLs, answers are saved in the browser after every click and exported as CSV (`study,
app_version, rater, bundle_id, page_index, category, phase, item_id, marked, step_started, step_submitted,
step_seconds, exported_at`; `marked` = not the category in phase `crop`, depicted in phase `frame`). A step is locked
once submitted. `demo_bundle/` is synthetic (drawn shapes); `test_e2e.py` runs the whole flow in headless Chromium
(passed 2026-11-15; Playwright 1.47 needs Python ≤ 3.12 and supports this Ubuntu 20.04 host).
