# Figure measures: what every data point, bar and interval is

Reference for the revision's figures (CDI domains, DINOv3, renamed labels, 182 core categories). Written from the
plotting and analysis code as of 2026-11-15; file and function names are given so each entry can be checked. Numbers
cited in text should come from `results/numbers/numbers_key.csv`.

## 1. Building blocks shared by all figures

| Term | Definition | Code |
|---|---|---|
| Frame | One frame every 5 s of BabyView (35 children) or VEDB (709 adult sessions) video | `annotation/qwen_pilot/` |
| Box / exemplar | One object Qwen3.8-27B found in a frame, named by its basic-level category and labelled with a format (real, toy, picture, screen) | box index `boxes_index_renamed.csv.gz` |
| Embedding | DINOv3 vector of the box crop, L2-normalized | `embed_full_run.py` |
| Category mean | Mean of the L2-normalized vectors of **every** box of the category, pooled over all children (BabyView) or sessions (VEDB). Children with more boxes weigh more. THINGS: all images of the mapped concepts; ImageNet: whole images of the mapped classes; LVIS: box crops | `build_crossdataset_tables.py`, `build_open_means.py` |
| Real-only mean (`bv_real`, `vedb_real`) | The same, using only boxes labelled `real` | same |
| Core set | 182 categories present in all five datasets (`data/included_categories_qwen_core.txt`); 164 when BabyView and VEDB are restricted to real objects (`..._core_real.txt`) | |
| Domains (CDI) | animals 22, clothing 20, food & drink 13, furniture & rooms 22, household 83, outside 3, toys 9, vehicles 10 (core set). Real-only set: animals 10, toys 8, vehicles 5, others unchanged. Domains with < 3 categories are not analysed; `outside` (3) is computed but left out of the figures | `categories.csv` |
| RDM | Category means z-scored per embedding dimension **across the categories being compared**, then cosine distance between every pair of categories | `rdm()` in every script |
| Δ (raw domain clustering) | Mean distance of between-domain pairs − mean distance of within-domain pairs | `cluster()` |
| Δ/SD | Δ divided by the SD of all pairwise distances in that RDM, so datasets with different distance spreads are comparable. This is the main clustering measure | `cluster()` |
| Per-domain Δ/SD | For one domain: (mean distance from its categories to all categories outside it − mean distance among its own categories) / SD of all pairwise distances | `domain_deltas()` |
| AUC | Probability that a randomly chosen between-domain pair is farther apart than a randomly chosen within-domain pair (0.5 = no clustering). Kept in the control tables and `numbers_key.csv`; no figure draws it | `measures()` in `controls_analysis.py` |
| Frame prevalence | Frames containing ≥ 1 box of the category / all frames, pooled over children (or sessions) | `prevalence()` |
| Age bins | <12 mo (23 children), 12–18 mo (29), 18–24 mo (23), 24+ mo (15); a child can be in several bins | `bv_vedb_analysis.ipynb` |

## 2. Error bars: ±1 SE everywhere (since 2026-11-15)

Every error bar in the main-text figures and the response-letter figures is **±1 standard error**. No figure shows bare
bars: each shows either every unit (child, session, category) as a dot, or ±1 SE, usually both. 95% CIs and p-values stay in the result tables, in `numbers_key.csv` (`ci_low`, `ci_high`) and in in-figure
test text (Fig. 6A bracket, R1d text). Comparisons between datasets are tested on the difference with the same bootstrap
draws, so whether two bars overlap does not decide significance; the stars and difference CIs do.

| Code | SE of | How | Used in |
|---|---|---|---|
| **A** | a statistic computed over categories (Δ/SD, RDM ρ) | SD of the category-bootstrap draws (10,000 where the figure also shows a test: Figs. 6A–C, 7B, E3, R1a, R1d, R1e, R1f, R1j, R3a; 1,000 elsewhere), categories resampled **within each domain** (each domain keeps its size; a category drawn twice is not paired with itself; draw order sorted by domain so reruns are identical) | Figs. 6A–C, 7B (per child), R1a, R1d, R1e, R1f, R1j, R3a; RDM ρ in tables (300 draws) |
| **C** | a mean over children or sessions | SD across units / √n (scene statistics, format and truncation shares: `unit_shares.py`, units with ≥ 20 objects in the cell); for ratios, SD of 1,000 resamples of children and of sessions; validation scorecard: SD of 10,000 resamples of the 17 gold-set children | Figs. 1B, 2E, 3A, 3C, 7B (mean row), 8A, 8C (share depicted) |
| **D′** | a split-half divergence (JS, 1 − RDM ρ) | point = mean over random half-splits (notebook); SE = SD over 200 resamples of the bin's children and the VEDB sessions, re-running the split-half estimate on each (10 splits for JS, 5 for RDM; copies of a resampled unit stay in one half) | Fig. 8C |
| **B′** | a median over categories (MMD excess) | SD of 1,000 category-bootstrap medians | Fig. 8C (MMD panel) |
| **E** | a regression slope | model SE (child fixed effects / mixed model); in tables and titles, not drawn | Fig. 8B, S within-child |

Remaining 95%-CI intervals (unchanged): the bracketed ρ CIs in Fig. 2C (2,000 unstratified category draws). The
subject-split (R1b) and room / activity (R1c) controls were dropped on 2026-11-15.

## 2b. Significance markers: one rule for every figure (since 2026-11-15)

- **Test.** Two-sided bootstrap test on the difference, computed on the same draws as the SEs: p = 2 × the share of draws
  on the far side of 0, floored at 2 / draws. Every drawn test uses 10,000 draws, so p can resolve below .001 (floor .0002).
  Fig. 8B is a mixed-model test (model p).
- **Correction.** Benjamini–Hochberg within each figure's family of tests, the family named in the figure. Stars and
  printed q use the BH q; uncorrected p stays in the tables (`p`, `q` columns of `numbers_key.csv`). A single test (Fig. 6A)
  needs no correction.
- **Stars.** `*` q < .05, `**` q < .01, `***` q < .001, the same in every figure.
- **What is compared.** A bracket joins the two things compared (6A, R1e, R1j text). In many-bar panels (6C, E3, R1a, R1f) a
  star over a bar means "differs from BabyView" (6C, R1f: BabyView real), as the figure's title says.
- **Non-significant.** Shown as "n.s." where the test is the point of the figure (6A, 6B, R1e, R3a); left blank in the
  many-bar panels; open markers in Fig. 7B.

| Figure | Test | Family for BH |
|---|---|---|
| 6A | first-person mean − curated mean, Δ/SD | single test |
| 6B | BabyView − each dataset per domain | domains × datasets |
| S ego vs curated by domain | first-person − curated per domain | across domains |
| 6C / R1f | BabyView real vs each dataset per domain; BabyView all vs real | domains × comparisons |
| E3 | BabyView vs each dataset per domain | domains × datasets |
| 7B | each child vs THINGS on the child's categories | across the 35 children |
| 8B | within-child age slope | across measures |
| R1a | BabyView vs VEDB / LVIS / THINGS per variant; LVIS change vs unmatched | 18 comparisons; 5 changes |
| R1d | the four crop / whole-image comparisons | the four |
| R1e | BabyView vs VEDB before / after matching, and the change in that gap | the six shown |
| R1j | each dataset's change when the background is removed | datasets × versions × subsets |
| R3a | BabyView − THINGS per sharpness cut | the three cuts |

## 2c. Colours (since 2026-11-15)

Datasets and domains never share a colour (OKLab ΔE ≥ 15 between every dataset and domain colour under normal vision, CDI
and THINGS-27). Every figure save also runs `fig_style.text_overlaps`, which prints any pair of overlapping text labels.
- Datasets (`fig_style.DS_COLOR`, same in `talk_style.py`; "slate" scheme chosen 2026-11-15): BabyView blue `#1F5CA8`,
  VEDB rust `#C2511F`; curated datasets in cool slate neutrals, THINGS `#29343F`, ImageNet `#4E5966`, LVIS `#75828F`.
  "Real objects only" keeps the dataset's colour and is marked by hatching or open markers.
- Domains own the light, medium-chroma hues (`DOMAIN_COLOR`); "other" is sand `#D8C7A0`; THINGS-27 "tool" `#A4C639`,
  "vehicle" `#6BA3D5` (as CDI vehicles).
- Formats (Fig. 3A) are told apart by panel, not colour, so they never compete with the domain-coloured labels.
- Known limit: under simulated red-green colour blindness LVIS slate and a few pastel domains (household, home decor) come
  close; domains are always named in text, so identity never rests on colour alone.

## 3. Main-text figures

### Fig. 1 (validation)
- **1A**: example frame, crops and the average; no data.
- **1B** (`a01_scorecard`): one row per measure; dots = the 17 gold-set children (size = frames); black diamond = the rate
  pooled over all frames, ±1 SE from 10,000 resamples of the children (`annotation/gold_set/validation_uncertainty.py`).
  Inter-rater F1 on the 100 shared frames, resampled the same way (16 children).
- **1C, S category PR**: format confusion and per-category precision / recall (dots = categories).

### Fig. 2 (what is seen)
- **2A** (`make_manuscript_figures.long_tail`): each bar is one category's frame prevalence in BabyView (Tier 1 categories in
  the nine CDI domains; people and other left out). Top row: the 60 most frequent; below: the 10 most frequent per domain.
  Colour = domain. **No intervals.** S long tail VEDB: the same for VEDB.
- **2B** (`q03_rank_frequency`): categories present in both datasets; each line is one dataset's prevalences sorted in its
  own order, log–log. **No intervals.**
- **2C** (`q16_frequency_imagenet`): each dot is a category in both BabyView and ImageNet. x = BabyView frame prevalence,
  y = that category's share of mapped ImageNet images. Title ρ = Spearman across categories; bracket = 95% CI, 2,000
  unstratified category draws (`controls_analysis.sec_prev`).
- **2D** (`q04_scatter`): each dot is a category in both datasets, x = VEDB prevalence, y = BabyView prevalence; dashed line
  = equal prevalence. Title ρ = Spearman (its 95% CI is in `prevalence_spearman.csv` but not drawn).
- **2E** (`bv_vedb_analysis` §2, `category_frequency_ratio.png`): each dot is one category's log2(BabyView prevalence /
  VEDB prevalence), among categories with ≥ 50 frames in each dataset; the 15 most child-heavy and 15 most adult-heavy.
  Left: all frames; right: home-scene frames only. Whiskers = ±1 SE, **type C** (1,000 resamples; prevalence recomputed on each;
  `uncertainty_bv_vedb.py`). Drawn by `make_manuscript_figures.fig2e_frequency_ratio`.

### Fig. 3 (formats)
- **3A** (`fig3a_formats_by_domain`): one panel per format (toy, picture, screen); per domain, dots = children with ≥ 20 objects
  in the domain, value = the child's share of that domain's objects in that format; line = mean ± 1 SE across children.
  Categories that are depictions by definition (toy, doll, picture, screen…) left out. Panel title = mean share over all
  objects.
- **3B** (montage, `make_talk_montages.most_depicted`; private draft only): categories that are ≥ 95% depicted in BabyView
  (≥ 1,000 BabyView and ≥ 50 VEDB boxes; people left out), the 8 with the largest BabyView − VEDB gap in % depicted, one
  row each. Row text = % of the category's boxes that are depicted (toy, picture, screen) in each dataset
  (`depiction_gap_bv_vs_vedb.csv`); tiles = 6 BabyView crops of depicted boxes and 3 VEDB crops of any format, border =
  format; random draws spread over children and videos, dark or blank crops skipped, no frame with a person or face.
  Tile list for the privacy review: `manifest_most_depicted.csv`. The former bar chart of the 30 most-depicted categories
  is now **S most-depicted categories** (`q14_depicted_categories`; each bar is one category).
- **3C** (`q05_formats`): left, per child / session, % of its objects that are depicted, toy, picture, screen; right, %
  depicted per domain (intrinsic depictions left out). Dots = units with ≥ 20 objects; line = mean ± 1 SE. Pooled shares
  (every box once) are in `format_prevalence.csv` and `format_depicted_by_domain.csv`; unit means in `numbers_key.csv`
  (`formats.unit_mean.*`).
- **3D, 3E**: montages.

### Fig. 4 (similarity to THINGS)
- **4A** (`q13_similarity`): each dot is a category with a THINGS match and ≥ 30 BabyView boxes: cosine similarity between
  its BabyView mean and its THINGS mean, both z-scored across this shared set. Dots jittered horizontally; black line =
  median per domain; domains sorted by median. **No intervals** (the dots show the spread).
- **4B**: montage.

### Fig. 5 (RDMs)
- **5A** (`fig5a_rdms`): RDMs of the five datasets on the 182 core categories. Order: domain, then average-linkage
  clustering within domain on BabyView's RDM. Colour = percentile of the distance within that RDM. Titles: Spearman ρ
  between that RDM's upper triangle and BabyView's. **No intervals.**
- **5B** (`fig4a_rdm_agreement.png` → `fig5b_rdm_agreement`): Spearman ρ between every pair of RDMs (seven tables
  including real-only BabyView and VEDB; 164 categories when a real-only table is involved, 182 otherwise). Type A SEs and 95% CIs
  (300 draws) are in `pairwise_rdm_correlations.csv`, **not drawn**.

### Fig. 6 (domain clustering)
- **6A** (`q08e_ego_vs_curated`): dot plot; each point is one dataset's Δ/SD on the 182 core categories, whisker = ±1 SE,
  **type A** (10,000 draws); value printed beside it. Short dashed lines = mean of the three curated and of the two
  first-person datasets (no interval). Bracket = (mean of first-person) − (mean of curated) on the same draws, with its
  95% CI and two-sided bootstrap p as stars (`ego_vs_curated.py`). The y-axis is zoomed to the data (dot plot).
- **6B** (`q09b_bv_vs_each_by_domain`): BabyView as the anchor. Per domain, four points: BabyView's Δ/SD minus THINGS,
  ImageNet, LVIS and VEDB (colour = the comparison dataset); whisker = ±1 SE of the difference (the shared 10,000
  within-domain category-bootstrap draws); stars = BH across domains × datasets. Above 0 = BabyView clusters the domain
  more. The earlier first-person − curated by-domain panel is **S ego vs curated by domain** (`q09e_ego_vs_curated_by_domain`).
- **6C** (`fig6c_real_only`): the same on the 164 real-object categories; BabyView and VEDB from `real` boxes only;
  hatched bar = BabyView all formats. Whiskers = ±1 SE, **type A**. Stars: BabyView-real vs each dataset, and BabyView-all vs
  BabyView-real, BH across domains × comparisons.

### Fig. 7 (individual children)
- **7A** (`fig7a_all_children`): one RDM per child on the core categories with ≥ 20 of that child's boxes; BabyView's
  pooled order; percentile colour within each RDM. Title: frames · categories. **No intervals.**
- **7B** (`fig7_children`): one row per child (sorted by frames). For each child, the core categories with ≥ 20 of the
  child's boxes and a THINGS match; navy dot = the child's Δ/SD with ±1 SE (type A, 10,000 draws), dark grey dot = THINGS's
  Δ/SD **on the same categories**; light line joins them (it is **not** an interval). Filled navy = child > THINGS with BH
  q < .05 across the 35 children (two-sided bootstrap test of child − THINGS on shared draws); open = n.s. Top row = mean
  over the 35 children of each series, ±1 SE across children (type C).
- **7C** (`fig7c_children_pairs`): histogram over pairs of children (categories with ≥ 20 boxes in both children, ≥ 20
  shared): Spearman ρ between the two children's RDMs. Vertical line = mean. **No intervals.**

### Fig. 8 (scene statistics and age)
- **8A left** (`q06_size`): per frame, the median box area as a share of the frame; averaged per child (or session), then
  across the 35 children (709 sessions). Dots = each child (left) and each session
  (right); black bar = mean ± 1 SE across units (type C).
- **8A right**: % of objects cut off by the frame edge (box within 1% of an edge), per domain, core categories. Dots =
  children / sessions with ≥ 20 objects in the domain; line = mean ± 1 SE (type C; `unit_shares.py`). Pooled shares are
  in `geom_describe.csv`.
- **8B** (`scene_within_child_density.py`): one panel per measure. Dots = child-month means (child-months with ≥ 200
  frames), area ∝ frames. Thin lines = each child's own frame-weighted linear trend (children with ≥ 3 child-months).
  Thick line = within-child slope from the frame-weighted mixed model (random intercept and age slope per child). Title:
  slope per year, % change per year, BH stars. The **type E** CI is in the table, not drawn.
- **8C** (`make_dinov3_manuscript_figures.py`, five panels; x = age bin):
  - *JS divergence (category / domain)*: per draw, the children of the bin and all VEDB sessions are each split into random
    halves. Category-frame counts are summed per half and all four halves subsampled to the same total. Orange = JS
    (BabyView half 1, VEDB half 1); grey = JS (BabyView half 1, BabyView half 2) baseline; dashed = home-scene frames only.
    Point = mean of 200 draws (notebook); whisker = ±1 SE, **type D′**.
  - *Share of objects depicted*: per frame, share of boxes not `real`; mean per child; mean across the bin's children.
    Dots = each child; whisker = ±1 SE across children (type C). Orange line and band = VEDB mean ± 1 SE across
    sessions.
  - *How things look (MMD²)*: per category, BabyView and VEDB exemplars drawn from random halves of children / sessions
    (40 per half, 30 draws). Excess = MMD²(BabyView, VEDB) − the larger within-dataset MMD². Point = median excess
    over the categories present in every bin. Whisker = ±1 SE, **type B′**. Dashed = real objects only.
  - *1 − RDM ρ*: categories with ≥ 60 exemplars in every bin and in VEDB. Per draw, RDMs from half-samples (categories
    with ≥ 5 exemplars in all four halves). Orange = 1 − ρ(BabyView half, VEDB half); grey = 1 − ρ(BabyView half,
    BabyView half). Point = mean of 100 draws (notebook); whisker = ±1 SE, **type D′**.
- **S within-child development** (`development_within_child.py`): random-slope models; **type E** CIs in the titles.

## 4. Response-letter figures (`revision_private/figures/`)

| Figure | Measure | Intervals |
|---|---|---|
| R1i age trends | as Fig. 8C | ±1 SE |
| R1a crop-size floors / CLIP filter (LVIS) | **Δ/SD**, ≈104 LVIS-exclusive categories, original YOLOE + CLIP crops, CDI domains in both schemes | ±1 SE, type A (10,000 draws, `comparison-datasets` notebook 16); stars: BabyView vs each dataset (BH, 18); grey text: LVIS change vs unmatched (BH q, 5) |
| R1d LVIS crop vs whole image (+ BabyView) | **Δ/SD**, 100 categories | ±1 SE, type A, 10,000 draws; difference CIs and BH q as text |
| R1e geometry matching | **Δ/SD**, before / after matching size × vertical position × truncation | ±1 SE, type A (10,000 draws); brackets: BabyView vs VEDB Δ/SD; text: change in that gap after matching; BH across the six |
| R1f real only by domain | as Fig. 6C | ±1 SE, type A |
| R1j SAM 3 background removal | **Δ/SD**, slope chart (`background_response_figures.plot_sam`) | ±1 SE, type A (10,000 draws); change vs Qwen box with 95% CI and BH q as text |
| R3a blur | **Δ/SD** on all / sharpest half / sharpest third (156 categories) | ±1 SE, type A (10,000 draws); BabyView − THINGS Δ/SD difference with stars (BH across the three cuts) as text |
| R3b prevalence curves | — | not re-checked here |

## 5. Remaining inconsistencies

1. Some intervals exist in the tables but are not drawn (5B RDM ρ matrix, 8B slopes).
2. Figure values for formats and truncation are unweighted means over children / sessions; the pooled shares (every
   box once) differ and are what earlier text quoted. Both are in `numbers_key.csv`.
