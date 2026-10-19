# CodeOcean / slim-repo export (manuscript 2026)

Export a **manuscript-only** CodeOcean-ready tree (or new public GitHub repo) —
not the full ~6 GB monorepo and not Stage 0 cluster tooling.

```bash
# From monorepo root — writes next to the repo under $HOME by default
bash analysis/manuscript-2026/scripts/export_codeocean_capsule.sh --archive

# Prefer an explicit home path on shared clusters (avoid /tmp)
bash analysis/manuscript-2026/scripts/export_codeocean_capsule.sh \
  --out "$HOME/babyview-projects/vss2026/babyview-objects-manuscript-capsule" \
  --archive
```

See `--help` for flags.

## Default profile (`manuscript`)

| Include | Notes |
|---------|--------|
| Notebooks **01–05**, **08–10** (cleared outputs) | Paper analyses |
| Analysis scripts whitelist | Shuffle / RDM / long-tail / SI helpers |
| `data/shared_data_manuscript_2026/` | Anonymized Tier A (**includes** YOLOE↔VQA SI summaries by default; **excludes** video-level `unconstrained_objects.csv`) |
| `exemplar_set_embeddings/` | Z-scored tables only, from `shared_data…/embeddings/` |
| Root stubs | `included_categories_*.txt`, `cdi_words.csv`, `MCDI_…`, AP CSV |
| Docs | `README`, `REPRODUCTION`, `SCRIPTS`, `DATA_AVAILABILITY`, `requirements`, `manuscript_config` |

### Always excluded

| Path | Why |
|------|-----|
| `not_in_manuscript/`, logs, `__pycache__` | Exploratory / junk |
| Notebooks **06–07**, `00_build_…`, `run_*tmux*.sh`, `run_exemplar_…` | Stage 0 / cluster |
| `create_babydinov3_*.py`, `exemplar_set_zscore_*.py`, `check_exemplar_stage.py` | Tier C builders |
| `build_shared_public_data.py`, `export_codeocean_capsule.sh`, autofill | Packaging / drafting |
| `paths.example.env`, `online_annotation/` | Private paths / methods aside |
| Local `main_results/**/subject_*`, `*all_subjects*`, `*.npz` | Privacy / regenerable |
| `vqa_detections/.../unconstrained_objects.csv` | Video-level VQA dump (~19 M); SI uses summaries/figures instead |
| Real BabyView subject IDs / absolute `/home`, `/data2`, `/ccn2` paths | Scrubbed in the exported capsule (replaced with tokens) |
| Full monorepo `.git` | Bloated history |

### Optional flags

| Flag | Adds |
|------|------|
| `--minimal` | Drop exemplar tables (Tier A verify only) |
| `--include-figures` | PNG/PDF figure dumps (~118 M) |
| `--include-local-results` | Privacy-filtered local `results/` (no `subject_*`) |
| `--include-vqa-raw` | Video-level `unconstrained_objects.csv` (~19 M, has `video_id`) |
| `--no-vqa` | Skip YOLOE↔VQA summary tables/figures (not recommended) |
| `--include-stage0` | Notebooks 06–07 + GPU/tmux helpers (needs Tier C to run) |
| `--include-manuscript-pdf` | Submission PDF |

## Size guide

| Capsule | Approx. |
|---------|---------|
| Default manuscript-only | ~20–40 M unpacked |
| + figures | ~140–160 M |
| Old “ship everything under manuscript-2026” | ~84 M+ and leaked subject IDs — **do not use** |

## What it can / cannot do

| Goal | In capsule? |
|------|-------------|
| Check numbers vs `shared_data…/results_valid129/` | yes |
| Rerun **01–05 / 08–10** from z-scored category tables | yes (default) |
| Rebuild Stage 0 from per-crop `.npy` | **no** (BabyView access) |
| Rerun **04** from raw per-child embeddings | **no** without Tier C |

## After export

```bash
OUT="$HOME/babyview-projects/vss2026/babyview-objects-manuscript-capsule"
du -sh "$OUT" "$OUT"/*
chmod -R go-rwx "$OUT"   # already applied by the exporter when possible
```

On CodeOcean, use this directory as the capsule root (`analysis/` and `data/` siblings).
