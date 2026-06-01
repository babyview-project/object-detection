# Embedding visual-input drift — results and interpretation

**Date:** May 2026  
**Scope:** Pilot on the **top-8 densest** BabyView subjects (~9–29 months), filtered CDI detections (threshold 0.27), **valid129** categories.  
**Independent of** the clutter / `n_objects` analysis (`12_clutter_proxy_objects_per_frame.ipynb`).

This document summarizes crop-level **distribution drift** in DINOv3 (primary) and CLIP embedding space, extensions (null model, bin width, RDM linkage), and how to reproduce everything.

**Context-stratified per-category analyses** (location/activity from `data/all_contexts_public.csv`, merged with CCN-2026 variability) are documented in **[CONTEXT_CCN_INTEGRATION.md](CONTEXT_CCN_INTEGRATION.md)**. Run `per_category_context_analysis.py` for household (cup, plate), animals (dog, cat), and vehicles (car, stroller) with strata including `kitchen_like`, `living_room`, `outside`, `eating`, and `playing`.

---

## 1. Motivation

Development changes what children **see** (egocentric ecology) and how those inputs are **represented**. We treat month-to-month change in the embedding distribution as **visual-input drift**:

- **Centroid displacement** — how much the pooled visual “diet” moves in embedding space (1 − cosine similarity between monthly centroids).
- **Dispersion** — how spread out individual crops are around that centroid (mean 1 − cos(crop, centroid)); captures within-month exemplar / viewpoint variability.
- **Per-category centroids** — same metrics for a single label (e.g. cup), indexing **within-category** viewpoint / exemplar drift.

This complements existing **representational** developmental trajectory work (month-to-month RDM correlations) by measuring drift in **input** space before or alongside RDM change.

---

## 2. Methods (short)

| Choice | Setting |
|--------|---------|
| Embeddings | Per-crop `.npy` from `facebook_dinov3-vitb16-pretrain-lvd1689m` (primary); CLIP for comparison |
| Indexing | `merged_frame_detections_with_metadata_filtered-0.27.csv` |
| Age windows | Integer **month** (default) or **3-month** bins (`age_bin = floor(age/3)*3`) |
| Sampling | Reservoir **256 crops** per (child, month) or (child, month, category) |
| Min crops | 30 global / 15 per-category to keep a window |
| Edges | Consecutive calendar windows only (gap = 1 or 3 months) |
| Subjects | Top-8 densest (same spirit as developmental trajectory top-8 CSVs) |
| Excluded | Subject `00270001`; categories `person`, `picture` |

**Scripts**

```bash
# From repo root
python analysis/manuscript-2026/embedding_visual_drift_pilot.py --top-n 8 --embed-model dinov3
python analysis/manuscript-2026/embedding_drift_extensions.py --all
```

**Outputs:** `results/` (CSVs, JSON, `centroids_full_*.npz`), `figures/` (PNGs/PDFs).

---

## 3. Main results (DINOv3, monthly, top-8)

### 3.1 Global pooled drift

Across all valid129 categories in a month, the global centroid captures the **mix** of visual inputs.

| Statistic | Value |
|-----------|-------|
| Subject-month windows | 108 |
| Consecutive month-pairs | 96 |
| Mean centroid displacement | **0.101** (SD 0.078) |
| Driftₜ vs driftₜ₊₁ (Pearson) | **r ≈ 0.52**, p ≈ 3×10⁻⁷ |

**Interpretation:** Input space shifts modestly each month on average (~0.1 on a 0–2 scale for 1−cos). Drift is **autocorrelated** — a month with large displacement tends to be followed by another relatively large one, which supports treating drift as a predictable developmental time series rather than i.i.d. noise.

**Figure — global trajectories (per child + group mean)**

Top: centroid displacement between consecutive months. Bottom: change in dispersion (Δ dispersion).

![Global drift trajectories (DINOv3, monthly)](figures/global_drift_trajectories_dinov3_valid129_bin1_max256_top8.png)

---

### 3.2 Per-category drift

Pilot categories: cup, book, toy, hand, face, bottle, chair, ball.

| Category | Mean displacement | N edges |
|----------|-------------------|---------|
| bottle | 0.250 | 88 |
| cup | 0.216 | 90 |
| ball | 0.194 | 73 |
| toy | 0.144 | 88 |
| book | 0.134 | 78 |
| chair | 0.127 | 94 |
| hand | 0.055 | 93 |
| face | — | (insufficient consecutive pairs in top-8) |

**Interpretation:** **Within-category** drift is often **much larger** than global drift (cup ~2× global mean). The same word label (e.g. “cup”) corresponds to different regions of embedding space at different ages — consistent with growing viewpoint / context / exemplar diversity. That is the most direct link to a **within-category variability** story for CCN.

Hand shows relatively **low** displacement (body part may be more stereotyped in appearance across months in this pipeline). Bottle and cup are highest (many contexts: kitchen, play, different instances).

**Figure — per-category trajectories**

![Category drift trajectories (DINOv3, monthly)](figures/category_drift_trajectories_dinov3_valid129_bin1_max256_top8.png)

---

## 4. Extensions

### 4.1 Age-shuffled null (temporal structure)

**Procedure:** Within each child, randomly permute which calendar month’s centroid is assigned to which age label; recompute displacement on the **same** consecutive month-pairs (e.g. 10→11). Repeat 500 times.

| | Mean displacement |
|--|-------------------|
| **Real** | 0.101 |
| **Null** | 0.125 |
| **Real − null** | **−0.024** |
| **p** (t-test, real vs null pool) | **0.004** |
| **Cohen’s d** | −0.31 |

**Interpretation:** Real timelines show **less** displacement than age-scrambled labels — month-to-month changes in visual input are **smoother** than expected if month labels carried no temporal order. This is **input-space developmental structure**.

Note the sign convention differs from RDM analyses where higher **correlation** is “more stable.” Here, **lower** displacement means more stability.

![Age-shuffled null (monthly)](figures/null_age_shuffle_dinov3_valid129_bin1_max256_top8.png)

**3-month bins** (fewer edges, n = 33): same pattern — real 0.070 vs null 0.083, p ≈ 0.016.

![Age-shuffled null (3-month bins)](figures/null_age_shuffle_dinov3_valid129_bin3_max256_top8.png)

---

### 4.2 Age bin width (1 vs 3 months)

| Window | Mean displacement | N edges |
|--------|-------------------|---------|
| 1 month | 0.101 | 96 |
| 3 months | 0.070 | 33 |

**Interpretation:** Coarser bins average over more developmental change per step but produce **fewer** edges; per-step displacement is smaller because each step spans more time (or because broader windows smooth variability). Use 1-month bins for alignment with RDM month-to-month CSVs; 3-month bins for smoother descriptive trajectories.

![Global drift (3-month bins)](figures/global_drift_trajectories_dinov3_valid129_bin3_max256_top8.png)

---

### 4.3 Backbone comparison: DINOv3 vs CLIP

Same pipeline, monthly bins, top-8.

| Backbone | Mean displacement | SD |
|----------|-------------------|-----|
| **DINOv3** | 0.101 | 0.078 |
| **CLIP** | 0.007 | 0.008 |

**Interpretation:** CLIP monthly centroids are **nearly static** in global displacement — likely because CLIP’s geometry compresses month-level means into a tight cone, not because development is absent. **DINOv3 is preferred** for this input-drift measure. CLIP may still be useful for category-level or alternative metrics.

![Backbone comparison](figures/backbone_comparison_monthly_top8.png)

CLIP trajectory plots are in `figures/global_drift_trajectories_clip_*` and `category_drift_trajectories_clip_*` for inspection.

---

### 4.4 Link to month-to-month RDM correlations

Merged 96 consecutive month-pairs with `developmental_trajectory_top8_densest/month_to_month_correlations.csv`.

| Comparison | Pearson r | p |
|------------|-----------|---|
| Embedding displacement vs **RDM correlation** | **−0.75** | ≈ 1.7×10⁻¹⁸ |
| Embedding displacement vs **RDM drift** (1 − r) | **+0.75** | ≈ 1.7×10⁻¹⁸ |
| Dispersion Δ vs RDM correlation | +0.26 | ≈ 0.01 |

**Interpretation:** Months when **visual input** shifts more in embedding space are months when **object representations** (RDM over categories) are **less stable**. Input drift and representational drift are strongly coupled in these children. Dispersion change is weakly positively related to RDM stability (smaller effect).

This does **not** prove causality (unmeasured confounds: hours of video, room context, mobility). It does justify studying input and representation drift **together**.

![Embedding drift vs RDM](figures/drift_vs_rdm_dinov3_valid129_bin1_max256_top8.png)

**Data:** `results/drift_rdm_merged_dinov3_valid129_bin1_max256_top8.csv`, `results/drift_rdm_correlations_*.csv`

---

## 5. Relation to other analyses in this repo

| Analysis | What it measures | Connection |
|----------|------------------|------------|
| **Developmental trajectory RDM** | Stability of category-level representation | Linked via r ≈ −0.75; representation vs input |
| **Clutter proxy (`n_objects`)** | Distinct categories per frame | Separate; weak age trend; not used in drift pipeline |
| **Within/between CDI clusters** | Semantic cluster geometry | Complementary representational summary |
| **Exemplar variability / BV vs THINGS** | Category centroids vs norms | Per-category drift here is a longitudinal within-child version |

---

## 6. Limitations and caveats

1. **Top-8 only** — densest recorders; not full cohort (`--top-n 0` not run yet).
2. **Detection-conditioned** — only frames with ≥1 filtered detection; no “empty” scenes.
3. **Reservoir subsample** — up to 256 crops/month; `n_indexed` in CSVs is the full metadata count.
4. **CLIP global metric** — not comparable in magnitude to DINOv3 without calibration.
5. **Category imbalance** — bottle/cup drift may reflect category diversity and detector behavior, not only “development.”
6. **No mobility / hours covariates** in merged models yet (hours available in RDM CSV for follow-on).

---

## 7. Suggested next steps

- Run **all subjects** and/or per-category age-shuffled nulls (especially **cup**).
- Overlay **mobility onset** or motor milestones on global trajectory plots.
- Predictive models: driftₜ → driftₜ₊₁; or joint model with RDM drift and video hours.
- Compare embedding drift to **youngest-vs-later** RDM CSV (not only consecutive months).

---

## 8. Context + CCN merge (per-category, stratified)

Human-coded **Location** and **Activity** labels (10s segments) are joined to detections via `video_id` (~99.6% match). This supports:

- **Household:** cup / plate in `kitchen_like` vs `living_room` vs `all`
- **Animals:** dog / cat in `outside` vs `all`
- **Vehicles:** car / stroller in `outside` vs `all`

Outputs live in `context_merged/` (see [CONTEXT_CCN_INTEGRATION.md](CONTEXT_CCN_INTEGRATION.md)):

| Output | Description |
|--------|-------------|
| `results/category_geometry_by_context_top8.csv` | Monthly dispersion + CCN columns per stratum |
| `results/category_displacement_by_context_top8.csv` | Consecutive-month centroid displacement per stratum |
| `results/dispersion_displacement_link_top8.csv` | Mean dispersion vs mean displacement per category × stratum |
| `figures/household_cup_plate_*`, `animals_dog_cat_*`, `vehicles_car_stroller_*` | Side-by-side panels |
| `figures/domain_context_summary_top8.png` | Three-domain overview |

```bash
python analysis/manuscript-2026/per_category_context_analysis.py --top-n 8
```

---

## 9. File index

### Figures (primary)

| File | Description |
|------|-------------|
| `figures/global_drift_trajectories_dinov3_valid129_bin1_max256_top8.png` | Main global trajectories |
| `figures/category_drift_trajectories_dinov3_valid129_bin1_max256_top8.png` | Per-category displacement |
| `figures/null_age_shuffle_dinov3_valid129_bin1_max256_top8.png` | Real vs null (monthly) |
| `figures/drift_vs_rdm_dinov3_valid129_bin1_max256_top8.png` | Input vs RDM stability |
| `figures/backbone_comparison_monthly_top8.png` | DINOv3 vs CLIP |
| `figures/global_drift_trajectories_dinov3_valid129_bin3_max256_top8.png` | 3-month bins |

### Results tables

| File | Description |
|------|-------------|
| `results/global_trajectory_edges_dinov3_valid129_bin1_max256_top8.csv` | Main edge-level metrics |
| `results/category_trajectory_edges_dinov3_valid129_bin1_max256_top8.csv` | Per-category edges |
| `results/global_windows_*.csv` | Window-level centroid stats |
| `results/centroids_full_*.npz` | Full centroid vectors for modeling |
| `results/null_age_shuffle_summary_*.json` | Null model summaries |
| `results/drift_rdm_merged_*.csv` | Merged embedding + RDM rows |
| `results/backbone_comparison_monthly_top8.csv` | CLIP vs DINOv3 means |
| `results/bin_width_comparison_top8_dinov3.csv` | 1 vs 3 month bins |

### Code

- `../embedding_visual_drift_pilot.py`
- `../embedding_drift_extensions.py`
- `../13_embedding_visual_drift_pilot.ipynb`

---

## 10. One-paragraph takeaway

In the densest eight children, DINOv3 embedding space shows small but structured month-to-month **visual-input drift** (mean displacement ≈ 0.10 globally, ≈ 0.22 for cup), smoother than an age-shuffled null, and **strongly aligned** with month-to-month **RDM instability** (r ≈ −0.75 between displacement and RDM correlation). Per-category trajectories are the most interpretable handle on within-label exemplar change; CLIP is a poor global backbone for this particular metric. Together, this supports a developmental story in which changing egocentric input statistics co-occur with changing representational geometry — with input drift now measurable directly from crops, not only inferred from RDMs.
