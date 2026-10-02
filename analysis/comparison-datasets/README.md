# Comparison datasets: ImageNet-1k and LVIS v1

ImageNet-1k and LVIS v1 as comparison datasets for BabyView (BV), alongside THINGS (the manuscript's
comparison set) and VEDB (adult egocentric video, [`../vedb-1612/`](../vedb-1612/)). Each dataset's classes are
mapped onto CDI words by WordNet hyponym pooling, category embeddings are built the way the manuscript
builds THINGS, and BV is compared to each on the categories the datasets share.

**The analysis code is the notebooks in [`notebooks/`](notebooks/).** Each notebook is self-contained: it
defines every function it uses, so you can edit any cell, rerun and save. Notebooks pass results to each other
only through files in `outputs/`; shared helpers (paths, table loading) are deliberately repeated in each
notebook. Open them with the *Python (vislearnlabpy)* kernel, the environment every result was made with.

This folder reads two inputs from other pipelines and writes nothing back into them:

| Input | Written by |
|---|---|
| BV and THINGS raw valid129 category tables: `manuscript-2026/exemplar_set_embeddings/valid129/` | manuscript Stage 0 (notebooks 06–07) |
| VEDB raw valid129 tables: `vedb-1612/exemplar_set_embeddings/valid129/` | `vedb-1612` Stage 0 |

## Layout

```text
comparison-datasets/
├── notebooks/                  00–10: the analysis (documented, parameterized, self-contained)
│   └── mapping_audit_template.html   page template used by notebook 02
├── data_prep/                  get ImageNet/LVIS onto ccn2 and embed them (shell, GPU)
└── outputs/
    ├── mapping/                maps, anchors, coverage summaries, audit tables + page, class/category means, LVIS crop manifest
    ├── montages/               imagenet_val/, lvis_uniform/, lvis_round_robin/   (each split valid129/ not_valid129/)
    ├── exemplar_sets/          imagenet_{hyponym79,exclusive75}/, lvis_{hyponym105,exclusive104}/
    ├── bv_imagenet/            BV vs THINGS vs ImageNet          results/ figures/
    ├── bv_lvis/                BV vs THINGS vs LVIS              results/ figures/
    ├── rdms/<datasets>/        RDMs clustered within CDI domain  results/ figures/
    ├── rdm_matrices/<datasets>/ RDM-to-RDM correlation matrices   results/ figures/
    ├── cluster_strength/<datasets>/ CDI-domain cluster strength  results/ figures/
    └── category_prevalence/    long-tail prevalence: BV, VEDB, ImageNet, LVIS (pooled + native)
```

`outputs/exemplar_sets/`, `outputs/montages/`, the `.npz` means and the LVIS crop manifest are git-ignored
(large and regenerable; see [`.gitignore`](.gitignore)). Results tables and figures are kept.

## Notebooks and run order

Each notebook has a parameters cell at the top (the dataset or datasets), documents its inputs, outputs and
decisions, and ends with a review checklist. Run in this order; 01 and 05 once per comparison dataset.

| # | Notebook | Parameter | What it does | Time |
|---|---|---|---|---|
| — | `data_prep/pull_imagenet_lvis_from_oak.sh`, `unpack_imagenet_lvis.sh` | — | pull ImageNet + LVIS from Oak to `/data2` and unpack | ~1.5 h |
| — | `data_prep/run_embeddings_imagenet_lvis.sh` | — | CLIP + DINOv3 on whole ImageNet images (whole LVIS images too; **not used**) | ~1.5 h GPU |
| 00 | `00_overview_and_setup` | — | paths; checks every input exists | s |
| 01 | `01_cdi_mapping` | `imagenet` / `lvis` | anchors, CDI ↔ class map, coverage; LVIS crop manifest | ~1 min |
| — | `data_prep/run_lvis_crop_embeddings.sh` | — | CLIP + DINOv3 on 928,726 LVIS box crops (needs 01 with `lvis`) | ~1.3 h GPU |
| 03 | `03_class_embeddings` | `imagenet` / `lvis` | per-class means of L2-normalized embeddings (`RECOMPUTE`; default loads saved); LVIS crop sizes | ~50 min to recompute |
| 05 | `05_stage0_category_tables` | `imagenet` / `lvis` | hyponym + exclusive category sets; BV/THINGS re-z-scored on them; reproduces the manuscript's 0.554 / 0.402 | s |
| 02 | `02_mapping_audit` | both | every hand-chosen sense and exclusion, review page | ~30 s |
| 04 | `04_montages` | `imagenet` / `lvis` | 5×5 exemplar montages | ~30 s |
| 06 | `06_bv_vs_comparison_dataset` | `imagenet` / `lvis` | category cosine, RDM correlations (permutation + bootstrap), RDM figures | ~1.5 min |
| 07 | `07_rdms_clustered` | any 2+ of bv vedb things imagenet lvis | pairwise RDM correlations, RDMs clustered within domain | ~1 min |
| 08 | `08_rdm_correlation_matrix` | any 2+ | dataset × dataset and dataset × backbone matrices | ~20 s |
| 09 | `09_cluster_strength` | any 2+ | raw and SD-scaled Δ, shuffle bands, bootstrap pair tests | ~1 min |
| 10 | `10_category_prevalence` | any of bv vedb imagenet lvis | pooled and native prevalence | ~30 s |

Headless execution (e.g. after editing): `jupyter nbconvert --to notebook --execute --inplace
--ExecutePreprocessor.kernel_name=vislearnlabpy notebooks/09_cluster_strength.ipynb`. The `vislearnlabpy` kernel
is registered at `~/.local/share/jupyter/kernels/vislearnlabpy` (remove with `jupyter kernelspec remove vislearnlabpy`).

## Decisions and why

**Mapping (notebook 01).**
- **WordNet hyponym pooling.** A CDI word pools every class at or below its anchor synset(s). Both datasets
  label classes with WordNet synsets, so one rule maps both, with no per-class hand-picking.
- **Senses are chosen per word** (notebook 01, *ImageNet anchors* cell). WordNet's first sense is often wrong for
  child vocabulary (arm → weapon, toy → toy dog, mouse → computer mouse). Words with no referent in a
  dataset get no anchor.
- **Added anchors** where WordNet splits one everyday object (cup + coffee mug; train + locomotives; LVIS
  glove + baseball glove). **Exclusions** where a hyponym isn't the referent (punching bag from ball; LVIS
  kites and frisbees from toy). Every anchor, its origin and every exclusion with its reason:
  `outputs/mapping/mapping_audit_*.csv` and the page `outputs/mapping/mapping_audit.html`.
- **LVIS = ImageNet anchors + LVIS-specific changes** (notebook 01, *LVIS anchors* cell), layered on top so the
  ImageNet mapping and results are unchanged.

**Exemplars.**
- **ImageNet:** all train + val images (1,331,167). **LVIS:** human-labeled ground-truth boxes, cropped;
  no detector, **no CLIP filter** (BV's filter screens detector errors), **no size filter** (BV keeps
  small objects too; a size threshold is to be chosen for all datasets at once). Only the 473 boxes under
  1 px are dropped, because they cannot be cropped. Whole-image LVIS embeddings exist but are not used:
  an LVIS image is a multi-object scene.
- **Category vector = mean over all pooled exemplars of the L2-normalized embedding**, as for BV, VEDB and
  THINGS. Checkpoints match BV: CLIP ViT-B/32 and DINOv3 ViT-B/16 CLS output (same lab HF cache).
  BabyDINOv3 has not been run on ImageNet or LVIS.

**Category sets (`outputs/exemplar_sets/`).**
- `<dataset>_hyponym<n>`: every valid129 word with a non-empty pool (ImageNet 79, LVIS 105). Pools can
  overlap; four ImageNet pairs are identical (bowl/dish, flower/plant, glove/mitten, jeans/pants).
- `<dataset>_exclusive<n>` (**primary**): each class counts only for its closest valid129 word (fewest
  hypernym steps); a tie goes to the word with the larger pool. ImageNet 75 (326 classes, 431,736 images),
  LVIS 104 (285 categories, 603,785 crops). No exemplar is shared between categories.
- **BV, THINGS and VEDB are re-z-scored within each shared set** from their raw valid129 tables, because the
  manuscript z-scores dimensions across categories; a 129-category table is not comparable to a 75-category
  one. Check: our code reproduces the manuscript's 129-category BV–THINGS RDM ρ exactly (0.554 / 0.402).

**Statistics (notebooks 06–09).**
- RDMs: cosine distance between z-scored category vectors; compared by upper-triangle Spearman.
- Is an RDM correlation above chance? One-sided label-shuffle permutation (5,000), as notebook 03.
- CIs and differences between correlations: category bootstrap (2,000), dropping pairs of the same
  original category.
- Cluster strength per CDI domain: Δ = mean between-domain − mean within-domain distance, reported raw and
  **divided by the SD of that RDM's distances** (RDMs differ in spread; see review notes). Structure vs
  chance: shared domain-label shuffle (2,000). **Differences between datasets: within-domain category
  bootstrap**, two-sided, BH-FDR across domains; domains with < 5 categories are not tested.
- RDM figures: CDI-domain blocks; within a domain either alphabetical (manuscript convention) or ordered by
  average-linkage clustering of one reference RDM, shared across panels. Every reference is saved, because
  the reference dataset always looks most block-structured.

## Results and figures

| Folder | Key files |
|---|---|
| `outputs/bv_imagenet/` | `results/summary_bv_things_imagenet.txt` (all numbers), `rdm_comparison_bv_things_imagenet.csv` (ρ, permutation p, bootstrap CI, BV–ImageNet minus BV–THINGS), `category_wise_cosine_bv_things_imagenet.csv`, `clip_vs_dinov3_rdm_anchor_by_dataset.csv`; `figures/rdm_spearman_summary_bv_things_imagenet.png`, `figures/rdms_bv_things_imagenet_<model>_<set>[_clustered_{bv,imagenet}].png` |
| `outputs/bv_lvis/` | same files with `lvis` |
| `outputs/rdms/<datasets>/` | `results/rdm_correlations_<datasets>.csv` (every pair: ρ, permutation p, bootstrap CI; BV–X minus BV–Y), `rdm_summary_<datasets>.txt`; `figures/rdms_<model>_<datasets>_{alpha,clustered_<dataset>}.png`. Sets run: `bv_vedb_imagenet` (75), `bv_vedb_lvis` (104) |
| `outputs/cluster_strength/<datasets>/` | `results/cluster_strength_<model>_<datasets>.csv` (per domain: Δ raw and SD-scaled, each RDM's SD, shuffle bands, every pair's bootstrap CI, p and q); `figures/cluster_strength_bars[_sdnorm]_<model>_<datasets>.png`. Sets run: `bv_vedb_imagenet` (75), `bv_vedb_lvis` (104), `bv_vedb_things_lvis` (104), `bv_vedb_things_imagenet` (75), `bv_vedb_things_imagenet_lvis` (72), `bv_things` (129) |
| `outputs/rdm_matrices/<datasets>/` | `results/rdm_correlation_matrix_<datasets>.csv` (pairs × model, ρ, bootstrap CI), `rdm_correlation_matrix_backbones_<datasets>.csv` (dataset × backbone); matching figures. Sets: `bv_vedb_things_imagenet` (75 categories), `bv_vedb_things_lvis` (104), `bv_vedb_things_imagenet_lvis` (72) |
| `outputs/category_prevalence/` | `results/category_prevalence_<datasets>.csv` (per dataset × category: count, pool size, prevalence, rank), `prevalence_spearman_<datasets>.csv`, `native_category_prevalence_{imagenet,lvis}.csv`; `figures/prevalence_rank_curves_<datasets>.png`, `prevalence_shared_categories_<datasets>.png`, `prevalence_native_imagenet_lvis.png` |
| `outputs/mapping/` | `cdi_{imagenet,lvis}_summary.csv` (per CDI word coverage), `{imagenet,lvis}_cdi_hyponym_map.csv`, `imagenet_class_to_cdi.csv`, `mapping_audit_{pairs,anchors,words}.csv`, `mapping_audit.html`, `lvis_crop_size_by_category_exclusive104.csv` |
| `outputs/montages/` | `imagenet_val/` (round-robin over pooled classes), `lvis_uniform/` (crops sampled uniformly from the pool: the mix each mean averages), `lvis_round_robin/`; each has `montage_manifest.csv` |

**Elsewhere** (extensions of existing VEDB scripts, so they stay with them):
- `vedb-1612/plot_bv_vedb_rdms.py` → `vedb-1612/analyses/{valid129,valid85}/figures/bv_vedb_rdm_comparison_<model>_clustered_{bv,vedb}_*.png`
  (BV vs VEDB on all shared categories, clustered within domain; orders in `results/`).
- `vedb-1612/plot_bv_vedb_category_montage.py` → `…/figures/bv_vedb_category_wise_cosine_montage_animals_*` (all CDI animals).

## Main results (exclusive sets; CLIP / DINOv3)

| Comparison | RDM Spearman ρ |
|---|---|
| BV–ImageNet vs BV–THINGS (75 categories) | 0.62 vs 0.59 / 0.43 vs 0.42; differences' CIs include 0 |
| BV–LVIS vs BV–THINGS (104) | 0.49 vs 0.60 / 0.53 vs 0.41; both differences' CIs exclude 0 (opposite signs) |
| BV–VEDB (75) | 0.83 / 0.77, far above BV–ImageNet (+0.21 / +0.34) |
| Five datasets (72) | BV–VEDB highest; CLIP-vs-DINOv3 stability: BV 0.90, LVIS 0.88, VEDB 0.82, ImageNet 0.57, THINGS 0.51 |
| Domain cluster strength, BV vs VEDB vs ImageNet (75) | no testable domain differs once Δ is scaled by RDM spread |
| Domain cluster strength with LVIS (104) | CLIP: LVIS has the most spread-out RDM (SD 0.271 vs BV 0.214) yet clusters less than BV once scaled: significant in animals, clothing and food (q < .05); furniture and household are borderline (q ≈ .05). DINOv3: same spread as BV; smaller Δ in most domains, none significant |
| Category prevalence | BV and VEDB rank categories alike (ρ = 0.78 on 72 shared); LVIS moderately (0.54 with BV), ImageNet not at all (0.11). Neither comparison dataset has a natural frequency: ImageNet has ~1,300 images per class, so pooled prevalence tracks how many classes a word pools (dog, 118 breeds, tops it); LVIS caps the images labeled per category (≤2,364; person has 1,928), so its frequent categories pile up at the cap |
| Domain cluster strength with THINGS (104) | raw DINOv3 shows BV ≫ THINGS everywhere, but THINGS' RDM is compressed (SD 0.071 vs 0.165); scaled: CLIP THINGS > BV in animals and furniture, DINOv3 BV > THINGS in food only |

## Review notes (2026-10-01)

Checked: every dataset uses per-exemplar L2 normalization and the DINOv3 CLS output; z-scoring is within
each shared set; the Stage 0 and comparison code reproduce the manuscript numbers; a full rerun after the
reorganization reproduced all 87 results tables byte for byte.

Fixed:
1. **Cluster-strength comparisons were confounded by RDM scale.** ImageNet's distances are much less spread
   out (SD 0.148 vs BV 0.208 for CLIP; 0.075 vs 0.161 for DINOv3), which shrinks raw Δ regardless of
   structure. Δ is now also reported divided by each RDM's SD.
2. **The pairwise cluster test used the wrong null.** It compared the observed difference with label
   shuffles, which remove structure from both datasets, so nearly any difference between two structured
   datasets looked significant. Pairwise tests now use a within-domain category bootstrap, and domains with
   fewer than 5 categories (outside, toys, vehicles here) are not tested. Effect: "ImageNet clusters are
   weaker than BV and VEDB" does not hold. With scaling and the bootstrap no domain differs.
3. The BV-vs-comparison analysis silently took the first category-set folder if a stale one existed; it now
   requires exactly one. The RDM correlation matrix failed for two datasets; fixed.
4. The analysis moved from scripts into self-contained notebooks (`notebooks/`), which were checked to reproduce
   all 106 results tables of the scripts byte for byte (one gzip differs only in its header timestamp), for
   every dataset and dataset combination in `outputs/`; the scripts were then retired.

The scaled Δ is one reasonable normalization, not the only one; raw and scaled results are both saved, and they
disagree exactly where the datasets' RDM spreads differ.

Open, outside this folder: notebook 05 (`manuscript-2026/05_within_between_cdi_cluster_correlation.ipynb`,
`scripts/bv_things_cdi_shuffle_inference.py`) and `vedb-1612/plot_bv_vedb_cluster_delta_bars.py` use the same
raw-scale Δ and shuffle-based difference test for BV vs THINGS and VEDB vs BV. The THINGS RDM is also
compressed: on the manuscript's 129 categories the SD of its distances is 0.129 vs BV 0.205 (CLIP) and
0.070 vs 0.160 (DINOv3). Notebook 09 with `DATASETS = ["bv", "things"]` reruns that comparison on the manuscript's 129
categories (`outputs/cluster_strength/bv_things/`):

- Raw Δ (notebook 05's measure), DINOv3: BV clusters more than THINGS in 7 of 8 testable domains.
- Scaled Δ, DINOv3: only two domains differ, in opposite directions: food (BV stronger) and body parts (THINGS stronger).
- Scaled Δ, CLIP: THINGS is stronger than BV in animals and furniture; nothing favors BV.

So a "BV categories cluster more tightly than THINGS" reading mostly reflects THINGS' compressed distances.
Any such claim in the manuscript should be revised or reported with the scaled measure. Notebook 05 itself
has not been changed.
