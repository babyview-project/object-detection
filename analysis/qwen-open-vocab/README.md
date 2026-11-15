# qwen-open-vocab/: analyses of the Qwen3.8-27B open-vocabulary detections (2025.2)

The full 2025.2 Qwen3.8-27B run (every 5th 1 fps frame; `annotation/qwen_pilot/`) grouped by Qwen's basic-level
category instead of the 129 YOLOE classes, with the manuscript's methods (category means of L2-normalized CLIP / DINOv3
crop vectors, z-scored across the categories compared, cosine RDMs, power-law long tail, domain clustering with a
category bootstrap).

| File | Does |
|---|---|
| `build_open_means.py` | open category set (844 basic-level categories with >= 300 boxes from >= 10 children; no generic labels, person, surfaces) and per-child embedding sums from the Qwen box store -> `out/private/open/` (private) |
| `build_open_nb.py` | writes `open_category_analysis.ipynb`; execute it into `out/private/` |
| `build_subject_nb.py` -> `qwen_subject_analysis.ipynb` | one child (S00320001): coverage, objects by age, long tail, formats, Qwen vs YOLOE, co-occurrence and embedding RDMs |
| `build_children_cache.py`, `build_children_nb.py` -> `children_analysis.ipynb` | the children complete at the time (15): long tail per child and at equal sample size, shared vs child-specific categories, depictions by child / age / domain, co-occurrence agreement |
| `build_cooccurrence_data.py`, `build_cooccurrence_nb.py` -> `cooccurrence_analysis.ipynb` | BabyView vs VEDB object co-occurrence (NPMI over frames, Tier 1 categories in >= 0.1% of frames in both): top pairs and associates, domain structure and scene clusters, split-half BabyView-VEDB agreement by age (all frames, home scenes, real objects only), pairs that differ (bootstrap over children / sessions), per child: closer to other children or to adults, co-occurrence vs CLIP / DINOv3 similarity, frame vs 30 s / 2 min windows -> `results/cooccurrence/`, `figures/cooccurrence/` |
| `make_subject_figures.py` | manuscript-style long-tail bars (all, real, depicted), individual RDMs (alphabetical and clustered within domain) -> `figures/<subject>/` |
| `make_depiction_figures.py` | depictions by category (all categories, top 100, per domain), pooled over children and per child -> `figures/<subject>/`, `figures/children/` |
| `make_depiction_montages.py` | crop montages by format for chosen categories (private: `out/private/depiction_montages/`) |
| `build_controls_data.py`, `controls_analysis.py`, `controls_figures.py` | the comparison-datasets controls (notebooks 10-16) rerun on the open categories, BabyView and VEDB: subject split, co-occurrence control, room / activity held fixed, crop geometry with noise ceilings and BabyView-VEDB equivalence, size floors, exemplar count, rows-only vs Mantel RDM null, category frequency agreement -> `results/controls[_renamed]/`, `figures/controls[_renamed]/`. One shard pass (~10 min) sums box vectors by unit x category x geometry cell / room / activity |
| `scene_within_child_density.py` | within-child age slopes of the scene statistics (`bv_vedb_analysis.ipynb` 1b) weighted by frames per child-month (also unweighted and precision-weighted) -> `results/bv_vedb[_renamed]/scene_statistics_within_child_density.csv` |
| `results/` | aggregate tables: 129-category replication (Qwen vs YOLOE + CLIP), long tail, BV domain clustering, individual RDM agreement, BV vs THINGS (515 shared concepts), category-wise BV-THINGS similarity |
| `figures/` | open set: long tail, BV RDMs (CLIP, DINOv3), BV vs THINGS RDMs; `S00320001/`, `children/` (incl. `depictions_pooled/` and one folder per child) |

Inputs: the Qwen box store and its embeddings (`/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1/`,
`annotation/qwen_pilot/embed_full_run.py`), the hierarchy table (`annotation/qwen_pilot/results/category_hierarchy_*.csv`),
and THINGS embeddings (`/data2/dataset/babyview/related_data/things/embeddings/`, `annotation/qwen_pilot/embed_things.py`).
(`results/replication_129_qwen_vs_yoloe.csv` is a record of the earlier 129-category Qwen vs YOLOE replication; its inputs
were removed on 2026-11-15.)

```bash
cd analysis/qwen-open-vocab
python build_open_means.py --workers 4                    # analysis env (vislearnlabpy), ~10 min
python build_open_nb.py
jupyter nbconvert --to notebook --execute open_category_analysis.ipynb --ExecutePreprocessor.kernel_name=vislearnlabpy \
    --ExecutePreprocessor.timeout=7200 --output-dir out/private --output open_category_analysis_executed.ipynb
```

Controls (after `build_open_means.py`, also with `--dataset vedb`, `build_crossdataset_tables.py` and `build_bv_vedb_data.py`):

```bash
QOV_LABELS=renamed nice -n 19 python build_controls_data.py --workers 16
QOV_LABELS=renamed nice -n 19 python controls_analysis.py --sections split cooc geom size count perm prev
for m in clip dinov3; do for k in room act; do                # room is slow: four slices in parallel, then merge
  QOV_LABELS=renamed nice -n 10 python controls_analysis.py --sections room --room-only $m $k & done; done; wait
QOV_LABELS=renamed python controls_analysis.py --sections room_merge
QOV_LABELS=renamed python controls_figures.py
```
