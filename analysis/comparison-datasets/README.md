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
    ├── context_split/          subject / session split test of shared-scene context  results/ figures/ (cache/ ignored)
    ├── context_controls/       co-occurrence control + LVIS comparison; supplementary tables  results/ (cache/ ignored)
    ├── context_fixed/          room- and activity-fixed cluster strength (BV 2025.2)  results/ figures/ (cache/ ignored)
    ├── background_amount/      LVIS whole image vs crop; ImageNet vs THINGS  results/ figures/ (cache/ ignored)
    ├── crop_geometry/          BV vs VEDB crop geometry, matching, noise ceilings, equivalence  results/ figures/ (cache/ ignored)
    ├── matched_controls/       crop-size / CLIP-filter matching vs LVIS; exemplar-count matching  results/ figures/ (cache/ ignored)
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
| 11 | `11_context_split_cluster_strength` | — (bv, vedb, things) | does BV domain clustering survive when compared categories come from different children (split by subject ID; no household information was available)? Within- vs cross-half Δ, Δ/SD and AUC; BV vs VEDB vs THINGS | ~10 min (+25 min to rebuild the per-video cache) |
| 12 | `12_cooccurrence_and_lvis_controls` | — (bv, vedb, things, lvis) | same-domain coefficient with and without frame co-occurrence / shared-context controls; LVIS in-scene crops vs BV/VEDB/THINGS; writes Supplementary Tables Y and Z | ~25 min |
| 13 | `13_room_activity_fixed_cluster_strength` | — (bv 2025.2, things) | domain clustering with the room / activity of each 10-s clip held fixed (`merged_contexts.csv`): same-room, cross-room, vs pooled and THINGS | ~30 min, single low-priority thread |
| 14 | `14_background_amount` | — (lvis, imagenet, bv, vedb, things) | LVIS crops vs whole images (all / unshared photos); ImageNet whole photos vs THINGS, BV, VEDB with co-occurrence control | ~20 min, single low-priority thread (one pass over LVIS whole-image files) |
| 15 | `15_crop_geometry_bv_vedb` | — (bv, vedb) | crop geometry (size, position, truncation, clutter) by domain; AUC within geometry levels; geometry-matched AUC and BV–VEDB ρ; noise ceilings; BV − VEDB equivalence (±0.04) | ~10 min (+30 min to rebuild the cache) |
| 16 | `16_crop_quality_and_exemplar_count` | — (bv, vedb, things, lvis, imagenet) | short-side floors (≥ 32 / 64 px) on every crop dataset × BV's CLIP filter on LVIS; 12 exemplars per category in every dataset | ~2 min (+30 min to rebuild the cache, mostly reading 604k LVIS crop files) |

Headless execution (e.g. after editing): `jupyter nbconvert --to notebook --execute --inplace
--ExecutePreprocessor.kernel_name=vislearnlabpy notebooks/09_cluster_strength.ipynb`. The `vislearnlabpy` kernel
is registered at `~/.local/share/jupyter/kernels/vislearnlabpy` (remove with `jupyter kernelspec remove vislearnlabpy`).
`notebooks/matplotlibrc` sets `pdf.fonttype: 42`, so figure PDFs written by notebooks run from that folder keep
editable text (TrueType) instead of Type 3 glyph outlines.

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
  small objects too). Notebook 16 applies short-side floors to every crop dataset and BV's CLIP filter to LVIS. Only the 473 boxes under
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
| `outputs/context_split/` | `results/context_split_summary.csv` (per model × comparison × dataset: within- and cross-half Δ, ratio, Δ/SD, AUC with CI), `context_split_dataset_differences.csv` (paired category bootstrap: raw Δ, Δ/SD, AUC), `context_split_per_domain.csv`, `context_split_per_split_*.csv`; `figures/context_split_within_vs_cross_{bv_things,bv_vedb_things}.png`, `context_split_auc_*.png`. Sets: `bv_things` (127 categories), `bv_vedb_things` (116) |
| `outputs/context_controls/` | `results/cooccurrence_control_coefficients.csv` (same-domain coefficient alone / controlled, covariate coefficients, permutation p, R²), `cooccurrence_control_dataset_differences.csv`, `lvis_comparison_{summary,dataset_differences}.csv`, `supp_table_{Y,Z}_*.csv` and `supp_tables_YZ.md` (formatted for the supplement) |
| `outputs/context_fixed/` | `results/same_label_measures.csv` (per room / activity: BV same-label, BV pooled, THINGS; AUC, Δ, Δ/SD), `cross_label_measures.csv`, `bootstrap_tests.csv`; `figures/same_room_auc.png` |
| `outputs/background_amount/` | `results/lvis_whole_vs_crop_{summary,differences}.csv`, `imagenet_comparison_{summary,differences}.csv`; `figures/background_amount_auc.png` |
| `outputs/crop_geometry/` | `results/geometry_per_{domain,category}.csv`, `geometry_size_bins.csv`, `auc_by_geometry_level.csv`, `matching_{summary,differences}.csv`, `noise_ceilings.csv`, `equivalence_bv_vedb_auc.csv`; `figures/geometry_by_domain.png`, `auc_by_geometry_level.png`, `geometry_matching.png` |
| `outputs/matched_controls/` | `results/crop_size_by_dataset.csv`, `size_clip_matching_{summary,differences}.csv`, `exemplar_count_{summary,differences}.csv`; `figures/crop_size_by_dataset.png`, `crop_quality_matching_{auc,differences}.png`, `exemplar_count_auc.png` |
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
| Shared-scene context (notebook 11; 2025.2 BV split by child) | BV keeps 99% (CLIP) / 97% (DINOv3) of its Δ when compared categories never come from the same child; THINGS keeps 100%. So shared scenes do not explain BV clustering. On cross-half RDMs (116): BV and VEDB do not differ in AUC or Δ/SD; BV − THINGS AUC is −0.04 for CLIP (THINGS higher) and +0.05 for DINOv3, and VEDB − THINGS is the same or larger |
| Typical-setting context (notebook 12) | Same-domain structure survives frame co-occurrence and shared-context controls in every dataset (BV keeps 76–80%, p < .001). Shared context adds similarity in BV beyond domain (2–3× THINGS). With identical controls, BV's DINOv3 advantage over THINGS is no longer significant (+0.068, p = .01 → +0.023, p = .46); CLIP THINGS > BV either way; BV = VEDB throughout. LVIS in-scene crops cluster least (CLIP AUC 0.67 vs BV 0.77): context in the crop is not sufficient; most of that gap is LVIS's small crops (notebook 16) |
| Room / activity held fixed (notebook 13; 99.5% of BV crops in labeled clips) | Clustering barely changes when every crop comes from one room (AUC −0.001 to −0.027 vs pooled, part of it from fewer crops) and rises slightly when the two categories come from different rooms (+0.005 to +0.013). DINOv3 BV > THINGS persists with the room fixed (+0.03 to +0.05) and across rooms (+0.05, mostly p < .05); CLIP THINGS > BV throughout. Room-level setting explains neither the structure nor the DINOv3 advantage |
| Amount of background (notebook 14) | LVIS whole images do not cluster more than their own crops (CLIP +0.027, p = .34; DINOv3 −0.032, p = .20; DINOv3 with unshared photos −0.053, p = .04); BV beats LVIS whole images. ImageNet whole curated photos ≈ THINGS (AUC differences −0.02, p > .4). More background does not raise domain clustering |
| BV vs VEDB geometry (notebook 15) | Infants' boxes are ≈ 2× the frame area, more often cut off (39% vs 21%, category averages), lower in the frame and in less cluttered frames (all p < .001). AUC within geometry levels changes ≤ ~0.03; geometry-matched AUC changes ≤ 0.012. BV − VEDB is equivalent within ±0.04 (CLIP +0.006, 90% CI [−0.019, 0.032]; DINOv3 −0.003 [−0.036, 0.023]). BV–VEDB ρ 0.84 → 0.88 (CLIP), 0.78 → 0.81 (DINOv3) after matching, against noise ceilings 0.92–0.95: a reliable 10–16% of structure differs and geometry does not explain it |
| Crop quality and exemplar count (notebook 16) | Only 18% of LVIS crops pass BV's CLIP filter (median short side 24 px vs BV 93, VEDB 191). With a ≥ 64 px floor on every dataset, BV − LVIS AUC falls from +0.097 to +0.025 (CLIP, p = .22) and from +0.062 to +0.046 (DINOv3, p = .014; +0.029, p = .15 at ≥ 32 px); LVIS rises, BV and VEDB move ≤ 0.02. With 12 exemplars per category everywhere: BV − THINGS CLIP −0.060 (p < .001), DINOv3 +0.042 (p = .056); BV − ImageNet −0.016 / +0.079 (p = .69 / .024); BV = VEDB in every set |
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
