# Object check (in-house annotation task)

A static web page for lab RAs to judge YOLOE detections in two steps, gallerize-style:
one page per category, click the crops that are **not** the named object; then the same
crops again, reshuffled, each beside its full frame with the box drawn in pink.

**No data lives here or is ever uploaded.** The page (HTML/CSS/JS only) can be hosted on
GitHub Pages; each RA opens their own private bundle folder from disk with a folder picker.
The Content-Security-Policy forbids every network request (`connect-src 'none'`), images
are shown from local files as `blob:` URLs, and answers are kept in the browser
(`localStorage`, saved after every click) and exported as CSV.

## Files

| | |
|---|---|
| `index.html`, `style.css`, `app.js` | the task (no build step, no dependencies) |
| `demo_bundle/` | synthetic shapes (no real data): try the task, run the test |
| `make_demo_bundle.py` | regenerates `demo_bundle/` |
| `test_e2e.py` | headless-Chromium test on the demo bundle (`pip install playwright==1.47.0`) |

## Workflow

1. `python ../make_inhouse_bundles.py --raters RA1 RA2 RA3 --zip` (in `clip_band_annotation/`)
   writes `out/private/inhouse/bundle_<RA>/` (+ `.zip`): `manifest.json`, `crop/`, `frame/`.
   Each item goes to one RA; about 15% (1,028 items) go to every RA for agreement.
   About 3,070 items per RA on 129 pages of 20-24, about 270 MB.
2. Share each zip only through IRB-approved storage (bundles show children's homes and
   family members; never GitHub, Slack or email). The RA unzips it.
3. The RA opens the page, chooses the unzipped folder, works through the pages, and
   uploads **Export backup** files to the lab drive regularly (the page reminds every 10
   pages). Closing the browser is fine: choosing the folder again resumes. To switch
   computers: export, then on the new one import the backup and choose the folder.
4. Put the exports in `out/private/inhouse/responses/` and run
   `python ../merge_inhouse_responses.py` -> `out/private/human_labels.csv` (majority on
   shared items) and `inhouse/agreement.json` (Cohen's / Fleiss' kappa per step). Then
   `python ../compare_clip_gemini.py` adds `results/vs_human.csv` (precision / recall of
   CLIP at 0.26 / 0.27 / 0.28 and of Gemini, against the crop and the in-scene judgements).

## Exported CSV

One row per item per submitted or started step: `study, app_version, rater, bundle_id,
page_index, category, phase (crop|frame), item_id, marked_not_object (0/1),
step_started, step_submitted, step_seconds, exported_at`. Item ids are opaque; the CSV
holds no subject or video ids.

## Behaviour

- The crop step is locked once the RA moves to the scene step (so crop answers are never
  influenced by the frame); there is no going back to earlier pages.
- The scene step shows the page's items in a fixed per-page shuffle, unmarked.
- Click or Enter/Space toggles a card; "Enlarge" (below each frame) opens it full screen.
- Works in current Chrome, Edge, Firefox and Safari (folder picker = `webkitdirectory`).
  A private window does not keep progress; the page warns and asks for frequent exports.

## Instructions shown to RAs

Edit the "Mark as not the object" / "Counts as the named object" lists in `index.html`.
The invalid types follow the objects-gallerize (868 h) study: other categories, empty
detections (only background), and blurry or unrecognisable detections. Added here, as the
gallerize instructions do not say: toys, pictures, drawings and screens count as the
object. Gallerize's practice trial needs example images that cannot be hosted publicly,
so RAs are trained in person instead.
