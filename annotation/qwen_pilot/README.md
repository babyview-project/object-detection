# qwen_pilot/: Qwen3.8-27B (open weights, local) vs Gemini on the pilot frame sets

`Qwen/Qwen3.8-27B-FP8` (Apache 2.0) run with vLLM on node14's A40s. Frames never leave the
machine. Same prompts and frame sets as `../gemini_pilot/`, so the Gemini scorers and montage
scripts read the Qwen outputs unchanged (records go to `../gemini_pilot/out/private/qwen3.8-27b-fp8__*.jsonl`).

## Setup (node14)

| | |
|---|---|
| weights | `/data/j7yang/models/hf` (`HF_HOME`); FP8 31 GB; bf16 `Qwen/Qwen3.8-27B` 56 GB for the 2-GPU test |
| env | `/data/j7yang/venvs/qwen`: vLLM 0.32.0 **+cu129** and torch 2.13.0+cu129 (the driver is CUDA 12.6, so the default CUDA 13 wheels cannot run), torchcodec CPU build, pandas |
| FP8 on A40 | no native FP8: vLLM's Marlin weight-only fallback (supports this block-wise checkpoint) |
| flags | `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer would JIT-compile without nvcc); thinking off (`enable_thinking=False`); boxes asked in Qwen's native `bbox_2d` [x1, y1, x2, y2] and converted to Gemini's `box_2d` |

## Scripts

| File | Does |
|---|---|
| `run_qwen_pilot.py` | one task on one frame set (`gold`, `montage72`, `depiction`, `gold_v2`, `fullrun500`, `subject:<id>`); `--temperature/--presence-penalty`, `--mtp`, `--max-num-seqs`, `--shard K/N` + `--merge-shards N`; resumable |
| `run_qwen_pilot.sh` | smoke test, then every pilot set on one free GPU |
| `run_shards.sh` | one set split over several GPUs (`GPUS="0 1 2 3"`), merged at the end |
| `wait_for_gpu.sh` | starts the pilot once a GPU has >= 40 GB free on 3 checks |
| `compare_qwen_gemini.py` | Qwen vs Gemini on the same frames: looping, objects, cap hits, box agreement, format vs human tags |
| `category_recovery.py` | which human-labelled categories each open-vocabulary model recovers; labels beyond the CDI list |
| `make_overlays.py` | per-frame overlays and Qwen \| Gemini side-by-side pages (private) |
| `bench_qwen.py`, `run_bench*.sh` | throughput settings (in-flight frames, compact JSON, MTP, bf16 on 2 GPUs) |
| `temp_compare.py` | sampling settings vs looping and lost objects |
| `finish_subject.sh` | final merge for the S00320001 run |

## Findings so far (2026-10-22; aggregate tables in `results/`)

- **Gold set (5,919 frames, CDI list):** gold object found (word + IoU >= 0.5) 71.4% vs 47.8%
  for Gemini 2.5 Flash (+23.5 points, 95% CI 20.8-26.2) and 34.7% for Flash-Lite. Gemini saw
  these frames at 288x512 and Qwen at 512x910, but Qwen also leads by 19 points on objects
  >= 128 px. Reference boxes are YOLOE's (raters validated labels, not boxes).
- **Setup B (open label + format), 500 full-run pilot frames:** at temperature 0, near-duplicate
  loops 0.2% (Gemini 6.9%), 15.3 objects and 10.7 distinct labels per frame (Gemini 14.1 / 8.5).
- **Formats vs Mira / Dora tags (80 frames):** toy 78% (Gemini 42%), photo 48% (21%), screen 47%
  (38%), drawing 80% (83%), real 92% (98%).
- **Throughput:** ~0.33 frames/s per A40; 96 frames in flight + MTP gives ~1.16x. Compact JSON
  makes the model return empty lists (unusable); without the JSON schema it wraps replies in
  code fences and writes more.
- **Sampling:** temperature 0 gives "marching" loops (same-size boxes stepping across the frame)
  on 4.6% of frames, mostly filling the 30-box cap. Temperature 1.0 (top_p 0.8, top_k 20) cuts
  them to 0.2% with -2.6% objects on non-looping frames; Qwen's presence penalty 1.5 removes
  ~18% of real objects. The S00320001 run (95,536 frames) uses temperature 1.0.
