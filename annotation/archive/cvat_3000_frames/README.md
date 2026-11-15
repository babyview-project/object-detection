# cvat_3000_frames/: CVAT box annotations on 3,000 sampled frames (868-hour)

Each of ~2,870 frames was sampled (inverse-proportionally by category,
`/ccn2/dataset/babyview/outputs_20250312/3000_inverse_proportional_sampled_frames.csv`) for
**one** YOLOE detection, pre-filled in CVAT. Mira and Dora edited only that box (checked
2026-10-19: in 99% of labelled frames the rater box matches it at IoU >= 0.9):

| Verdict | Share |
|---|---|
| label kept (YOLOE right) | ~41% |
| relabelled to the right CDI word | ~53% |
| box deleted (no recognisable CDI object): an empty label file | ~6% |

They also tagged each box's visual format (`real_format`, `toy_format`, `photo_format`,
`drawing_format`, `video_format`; class ids 295-299). Frames other than the one checked
object are **not** exhaustively labelled.

## Data: `annotation_data/`

| Folder / file | What |
|---|---|
| `{Mira,Dora}_20251001_completed*_VEDI/` | first export (YOLO 1.1 format: `obj.names`, `obj_train_data/frames/<video>_processed_<frame>.txt`) |
| `{Mira,Dora}_20251112_completed*_VEDI/` | second, smaller export |
| `7k_animal_depiction_annotation_04302026.csv` | separate study: 700 animal crops judged real-life vs depiction (manuscript notebook 10; untracked) |

## Scripts and notebooks

| File | Does |
|---|---|
| `clip_filter_precision_and_bbox_validation.ipynb` | CLIP-filter precision and box agreement vs the human boxes (frame x category level) |
| `analyze_overlapping_bboxes.ipynb` | overlapping boxes in one export |
| `error_pattern_analysis.py` | YOLOE label errors (true -> predicted) and error montages |
| `make_format_montages.py` | montages of crops per visual format |
| `analyze_mira_filtered_annotations.py` | quick summary of a filtered category-annotation CSV |
| `convert_prediction_csv_to_yolo_cvat.py` | YOLOE predictions -> CVAT YOLO import |
| `annotation_analysis.qmd` | R summary of the exports |

These look for `annotation_data/` next to themselves. Several also expect
`3000_inverse_proportional_sampled_frames*/` (frames and YOLOE pre-fill) next to them, which
is no longer in the repo; the frames are under
`/ccn2/dataset/babyview/outputs_20250312/sampled_frames/`.

Also read by `gemini_pilot/build_empty_frames.py` and `build_depiction_frames.py`.
