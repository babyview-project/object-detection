# video-qa/

Exploratory video question answering (VQA models, e.g. Video-LLaMA) for unconstrained object
labels; marked in progress in the code.

| File | Does |
|---|---|
| `video_llama.py`, `model.py`, `trying_prompts.py` | model wrappers and prompt trials |
| `unconstrained_questions.py` | asks open questions about clips |
| `process_unconstrained_results.ipynb`, `process_constrained_results.ipynb` | parse model answers |
| `unconstrained_objects/` | Quarto comparison with YOLOE category counts (reads `data/shared_data_manuscript_2026/vqa_detections/`) |

The Gemini object annotation that replaced this line of work is in `annotation/gemini_pilot/`
and `annotation/gemini_full_run/`.
