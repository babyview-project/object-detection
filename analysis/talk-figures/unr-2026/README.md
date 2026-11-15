# talk-figures/unr-2026 (figure helpers for the revision)

What is left of the UNR talk (Nov 4, 2026) folder after the 2026-11-15 cleanup: only the code
the manuscript and response-letter figure scripts import, its fonts, and the montages they copy.
The deck, slide figures and the other talk scripts were deleted.

| File | Used by |
|---|---|
| `talk_style.py` | Lato fonts, dataset colours; imported by the files below |
| `fig_qwen_talk.py` | `qwen-open-vocab/make_manuscript_figures.py` (`import fig_qwen_talk as Q`) |
| `fig_annotation_talk.py` | `qwen-open-vocab/make_manuscript_figures.py` (`import fig_annotation_talk as A`) |
| `make_talk_montages.py` | montages for Supplementary Figures (similar / dissimilar, child vs adult); `qwen-open-vocab/make_background_examples.py` imports it |
| `fonts/` | Lato; read by `qwen-open-vocab/fig_style.py` and every `comparison-datasets` notebook |
| `figures/private/montages/` | `make_talk_montages.py` output, copied into the manuscript by `make_manuscript_figures.py montages` |

`figures/private/` holds BabyView and VEDB frames and is git-ignored. Do not commit it.
Running `fig_qwen_talk.py` or `fig_annotation_talk.py` directly still writes talk panels to `figures/`.
