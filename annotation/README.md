# annotation/

Human annotation and model-based labelling of BabyView detections.

| Folder | What | Status |
|---|---|---|
| [`precision_868h/`](precision_868h/) | Prolific crop-validation study on the 868-hour release (raters marked invalid YOLOE crops). Source of `per_class_validation_data.csv` / `per_file_precision_data.csv`, used across the manuscript. | done |
| [`cvat_3000_frames/`](cvat_3000_frames/) | Mira / Dora CVAT boxes on 3,000 inverse-proportionally sampled 868-hour frames (one YOLOE box checked per frame, plus visual-format tags), and the animal-depiction labels | done |
| [`gemini_pilot/`](gemini_pilot/) | Gemini (Vertex) model pilot on the valid5921 gold crops: naming, verification, whole-frame detection, open vocabulary, combined label + CDI word + format; box comparisons with YOLOE + CLIP | done (2026-10) |
| [`clip_band_annotation/`](clip_band_annotation/) | CLIP-threshold sample (129 categories x 7 CLIP bands) for in-house RAs; Gemini vs CLIP comparison; the object-check web task ([vislearnlab.github.io/object-check](https://vislearnlab.github.io/object-check/)) | annotation pending |
| [`gold_set/`](gold_set/) | Gold set v2: RAs correct Gemini pre-filled boxes on whole 2025.2 frames (box, open name, format) | in progress |
| [`gemini_full_run/`](gemini_full_run/) | Vertex batch run of Gemini 2.5 Flash (open label + format) over 2025.2 BV-main, one frame per 5 s | done |

Shared files at this level:

| File | Use |
|---|---|
| `cvat_cdi_labels.json` | the 295-label YOLOE / CDI vocabulary (CVAT label set; also the Gemini word list). Built by `get_cdi_words_json.py`. |
| `word_list.json` | CDI word list |
| `annotate_crops.py` | tkinter GUI for checking crops (category, CDI semantic category) |
| `environment.yml`, `requirements.txt` | environment for the GUI (only Pillow beyond the standard library) |

## Privacy

Crops and frames show children's homes and family members. Every folder keeps images,
subject / video ids and per-frame outputs in a git-ignored `out/private/` (or untracked
data files); only code, aggregate tables and image-free figures are committed. Gemini is
used only through Vertex AI on the lab grant project (see `gemini_pilot/README.md`).

## Crop GUI setup

```bash
conda env create -f annotation/environment.yml && conda activate crop-annotation
python annotation/annotate_crops.py        # or: pip install -r annotation/requirements.txt
```
