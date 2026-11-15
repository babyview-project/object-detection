# qwen-open-vocab/: the revision's main analyses (Qwen3.8-27B open-vocabulary detections)

Every main-text analysis and figure of the revised *BabyView Objects* manuscript, and the response-letter figures.
Input: the full 2025.2 Qwen3.8-27B run on BabyView and VEDB (one frame every 5 s; `annotation/qwen_pilot/`), grouped
by Qwen's basic-level category. Methods follow the original manuscript: category means of L2-normalized crop vectors
(DINOv3 reported; CLIP computed), z-scored across the categories compared, cosine RDMs, power-law long tail, domain
clustering with a within-domain category bootstrap.

## Variants (environment variables, `variant.py`)

| Variable | Values | Effect |
|---|---|---|
| `QOV_LABELS` | `renamed` (use this), `original` | `renamed`: generic labels replaced by the renaming pass (`boxes_index_renamed.csv.gz`); folders get `_renamed`. The `original`-label outputs were removed on 2026-11-15 (only `out/private/open/categories.csv` is kept, for the WordNet mapping) |
| `QOV_CATS` | `core`, `tier1` (default) | `core`: the 182 core categories (`data/included_categories_qwen_core.txt`); folders get `_core` |
| `QOV_DOMAINS` | `cdi` (default), `things27` | `things27`: THINGS high-level categories (`data/category_domains_things27.csv`, `build_things27_domains.py`); folders get `_things27`, manuscript figures go to `figures_things27/` |

VEDB is restricted to adults (709 of 717 sessions, `analysis/vedb-1612/vedb_adult_sessions.txt`). The full rerun,
both domain schemes, is `analysis/vedb-1612/rerun_adults_only.sh`.

## Scripts

| Step | Scripts | Does |
|---|---|---|
| Category means | `build_open_means.py` (`--dataset vedb`) | open category set and per-child / per-session embedding sums → `out/private/open[_vedb]_renamed/` (private) |
| Cross-dataset mapping | `sense_pick.py`, `build_crossdataset_mapping.py` | WordNet sense per category (Qwen picks), then each THINGS / LVIS / ImageNet item mapped to a category → `mapping/map_*.csv` |
| Cross-dataset tables | `build_crossdataset_tables.py`, `embed_lvis_extra.py`, `build_format_tables.py`, `format_classifier.py` | category tables for BabyView, VEDB, THINGS, LVIS, ImageNet (`out/crossdataset_renamed/`); format shares |
| Domains | `build_things27_domains.py`, `make_things27_inputs.py` | THINGS-27 domains (hand labels in `mapping/domain_handmap_things27.csv`, column `final`), and the `*_things27` input copies |
| BabyView vs VEDB | `build_bv_vedb_data.py`, `build_bv_vedb_nb.py` → `bv_vedb_analysis.ipynb`, `scene_within_child_density.py`, `filter_bv_vedb_sample_adults.py` | scene statistics, category presence, embedding sample, age trends |
| Cross-dataset analyses | `build_crossdataset_nb.py` → `crossdataset_analysis.ipynb`, `ego_vs_curated.py`, `things_readout_check.py` | RDMs, RDM agreement, domain clustering, egocentric vs curated, THINGS readout check |
| Open categories | `build_open_nb.py` → `open_category_analysis.ipynb` | long tail, BabyView RDMs, per-child RDMs, BabyView vs THINGS |
| Co-occurrence | `build_cooccurrence_data.py`, `build_cooccurrence_nb.py` → `cooccurrence_analysis.ipynb` | co-occurrence structure, BabyView vs VEDB |
| Controls | `build_controls_data.py`, `controls_analysis.py`, `controls_figures.py`, `build_blur_control.py`, `analyze_background_control.py` | subject split, co-occurrence, room / activity, geometry, size floors, exemplar count, prevalence, blur, SAM 3 background removal |
| Development | `build_child_month_means.py`, `development_within_child.py`, `development_size_by_domain.py` | within-child random-slope checks of the age trends (Supplementary Figure S14), object size by domain (CDI only) |
| One child / early children | `build_subject_nb.py`, `build_children_cache.py`, `build_children_nb.py`, `make_subject_figures.py`, `make_depiction_figures.py`, `make_depiction_montages.py` | pilot analyses on S00320001 and the first children |
| Figures | `make_manuscript_figures.py`, `make_dinov3_manuscript_figures.py`, `background_response_figures.py`, `make_background_examples.py`, `make_crossdataset_montages.py`, `make_overview_examples_figure.py`, `fig_style.py` | manuscript figures (`figures/manuscript[_things27]/`, copied to `manuscript-2026/revision_private/manuscript_revision/figures[_things27]/`) and response-letter figures (`revision_private/figures[_things27]/`); some panels reuse `analysis/talk-figures/unr-2026/` |

## Folders

| Folder | Contents |
|---|---|
| `mapping/` | WordNet sense choices, dataset → category maps, THINGS-27 hand-mapping sheet (in git) |
| `results/` | aggregate tables per variant (`*_renamed[_things27][_core]`) |
| `figures/` | figures per variant; `manuscript[_things27]/`, `manuscript_dinov3[_things27]/` |
| `out/` | link to `/data2/.../qwen-open-vocab/out` (git-ignored): `out/private/` holds per-child tables, samples and executed notebooks; `out/crossdataset*/` the category tables |

Inputs: the Qwen box stores and embeddings (`/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1/`,
`annotation/qwen_pilot/embed_full_run.py`), the hierarchy table (`annotation/qwen_pilot/results/category_hierarchy_*.csv`),
THINGS embeddings (`/data2/dataset/babyview/related_data/things/embeddings/`, `annotation/qwen_pilot/embed_things.py`),
LVIS / ImageNet category tables (`analysis/comparison-datasets/outputs/mapping/`), SAM 3 masks
(`annotation/qwen_pilot/sam3_crop_sample.py`). `results/replication_129_qwen_vs_yoloe.csv` is a record of the earlier
129-category Qwen vs YOLOE replication; its inputs were removed on 2026-11-15.

## Running

Notebooks are written by their `build_*_nb.py` and executed into `out/private/` (they read private data):

```bash
cd analysis/qwen-open-vocab
export QOV_LABELS=renamed
python build_open_means.py --workers 4 && python build_open_means.py --dataset vedb --workers 4
python build_crossdataset_tables.py
QOV_CATS=core jupyter nbconvert --to notebook --execute crossdataset_analysis.ipynb --ExecutePreprocessor.kernel_name=vislearnlabpy \
    --ExecutePreprocessor.timeout=28800 --output-dir out/private --output crossdataset_analysis_executed_renamed_core.ipynb
```

Controls (after the means, `build_crossdataset_tables.py` and `build_bv_vedb_data.py`):

```bash
QOV_LABELS=renamed nice -n 19 python build_controls_data.py --workers 16
QOV_LABELS=renamed nice -n 19 python controls_analysis.py --sections split cooc geom size count perm prev
for m in clip dinov3; do for k in room act; do                # room is slow: four slices in parallel, then merge
  QOV_LABELS=renamed nice -n 10 python controls_analysis.py --sections room --room-only $m $k & done; done; wait
QOV_LABELS=renamed python controls_analysis.py --sections room_merge
QOV_LABELS=renamed python controls_figures.py
```

`build_child_month_means.py`, `development_within_child.py` and `background_response_figures.py` need an environment
with pyarrow (`/data/j7yang/anaconda3/bin/python`); the `vislearnlabpy` env has none.
