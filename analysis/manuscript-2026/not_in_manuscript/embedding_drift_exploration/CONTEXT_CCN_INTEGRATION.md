# Merging context labels, per-category drift, and CCN-2026 variability

This note describes how **`data/all_contexts_public.csv`** connects the **embedding drift pilot** (longitudinal, per child × age) to **CCN-2026 exemplar-variability results** (cross-sectional, pooled over the corpus).

---

## Three layers of analysis

| Layer | What it is | Time | CCN folder / script |
|-------|------------|------|---------------------|
| **A. CCN cross-section** | Within-category spread & kNN structure (pooled exemplars) | Static snapshot | `analysis/ccn-2026/plotC_knn_diversity_outputs/` |
| **B. Drift pilot** | Monthly centroid displacement & dispersion | Longitudinal (top-8) | `embedding_visual_drift_pilot.py` |
| **C. Context merge** | Stratify A/B by **Location** & **Activity** | Both | `per_category_context_analysis.py` + `all_contexts_public.csv` |

**Story arc for CCN + development:**

1. **CCN:** Some categories are high-variability *in general* (cup, bottle rank high on global dispersion / kNN in DINOv3).
2. **Drift:** For a given child, cup embeddings *also move* month-to-month (centroid displacement ~0.22).
3. **Context:** Cup detections are concentrated in **kitchen / dining room** and **eating** segments — we can ask whether drift and dispersion are **kitchen-cup** vs **living-room-cup**, and whether prevalence confounds are reduced.

---

## Joining detections to context

**Context file:** `data/all_contexts_public.csv`  
- One row per **10-second segment** of a recording  
- Columns: `video_id`, `Location`, `Activity`, `superseded_gcp_name_feb25`, `age_mo`, `start_time`, `end_time`

**Detections:** `frame_data/merged_frame_detections_with_metadata_filtered-0.27.csv`  
- `superseded_gcp_name_feb25` matches context  
- Frame index from `original_frame_path` (e.g. `.../00042.jpg`)

**Join rule (validated ~99.6% match on cup detections):**

```text
segment_idx = floor(frame_num / 10)
video_id = f"{superseded_gcp_name_feb25}_processed_{segment_idx:03d}.mp4"
```

Then merge `Location` and `Activity` on `video_id`.

**Caveats:**
- Segment labels are human-coded per 10s clip; `locations_avg` can differ slightly from instantaneous `Location`.
- ~0.4% of detections may not match (missing segment file or path parse edge cases).

---

## Context strata

| Stratum key | Rule | Primary use |
|-------------|------|-------------|
| `all` | Every detection | Comparable to drift pilot |
| `kitchen_like` | Location ∈ {kitchen, dining room} | Household (cup, plate) |
| `living_room` | Location == living room | Household contrast |
| `outside` | Location == outside | Animals / vehicles (when n sufficient) |
| `eating` | Activity == eating | Mealtime behavior |
| `playing` | Activity == playing | Play / locomotion contexts |

**Default categories** (CDI domains, valid129): `cup`, `plate` (household); `dog`, `cat` (animals); `car`, `stroller` (vehicles).

**Outputs:** `category_geometry_by_context_top8.csv`, `category_displacement_by_context_top8.csv`, `dispersion_displacement_link_top8.csv`, plus domain panels under `context_merged/figures/`.

---

## How this addresses the `prop_cup` confound

Raw **proportion of detections that are cup** mixes:

- developmental change in cup experience  
- change in **where families film** (kitchen vs living room)

**With context:**

| Metric | Confound control |
|--------|------------------|
| `n_cup` in stratum `kitchen_like` / month | Cup given **meal-context filming** |
| Centroid / dispersion of cup **only in kitchen_like** | Appearance drift **holding coarse scene constant** |
| Compare cup (kitchen) vs plate (kitchen) same month | Same semantic domain + similar location |

Still **not** gaze or object instances — but much better than global `prop_cup`.

---

## Linking to CCN-2026 results

**CCN metrics** (per category, pooled): e.g.  
`analysis/ccn-2026/plotC_knn_diversity_outputs/new_things_embeddings_20260428/bv_dinov3_local_global_k5_valid129.csv`

| Column | Meaning |
|--------|---------|
| `global_dispersion` | Mean distance to category centroid (pooled) |
| `mean_knn_dist` | Local neighborhood spread (k=5) |
| `local_coherence` | kNN-based compactness |
| `local_over_global` | Ratio local/global |

**Example (cup, DINOv3, valid129):** high `global_dispersion` (~15.3) and high `mean_knn_dist` — CCN “cup is variable.”

**Merge logic:**

- **Cross-sectional:** Each category’s **mean monthly dispersion** (drift pilot) vs CCN `global_dispersion` → do categories that are variable overall also drift more longitudinally?
- **Per child × month:** CCN does not vary by age in current exports; CCN is a **prior** on difficulty, drift is **trajectory**.

Suggested sentence for CCN paper / poster:

> Categories ranked as high within-category variability in our CCN sample (e.g. cup, bottle) also show large month-to-month centroid displacement in dense longitudinal subsets, especially when analyses are restricted to kitchen/dining contexts.

---

## Longitudinal + context outputs

**Script:**

```bash
python analysis/manuscript-2026/per_category_context_analysis.py --top-n 8
# optional: --categories cup plate dog cat car stroller
```

**Writes to** `embedding_drift_exploration/context_merged/`:

| File | Content |
|------|---------|
| `results/category_prevalence_by_context.csv` | `n_indexed` per subject × age × category × stratum |
| `results/category_geometry_by_context_top8.csv` | dispersion (+ centroid stats) per stratum |
| `results/category_cross_sectional_ccn_merge_top8.csv` | category means + CCN columns |
| `figures/household_cup_plate_*_top8.png` | Cup vs plate, kitchen / living room / displacement |
| `figures/animals_dog_cat_*_top8.png` | Dog vs cat (outside vs all when sparse) |
| `figures/vehicles_car_stroller_*_top8.png` | Car vs stroller |
| `figures/domain_context_summary_top8.png` | Three-domain overview |

---

## Suggested figures for CCN + development poster

1. **CCN montage / dispersion bar** (existing Plot A/B/C) — static variability ranking.  
2. **Cup: kitchen vs all dispersion by age** (new) — context-stratified drift.  
3. **Cup vs plate** in `kitchen_like` only: parallel displacement trajectories.  
4. **Scatter:** CCN `global_dispersion` vs mean longitudinal `dispersion` (all strata).  
5. **Optional:** pseudo-instance clusters **within** `(child, month, cup, kitchen_like)` for cup₁/cup₂.

---

## Analysis order (recommended)

```text
1. CCN static variability (done) — which categories are hard?
2. Per-category drift: centroid + dispersion (done, top-8)
3. Context stratification (this merge) — kitchen/eating cup
4. Pseudo-instances within stratum (next)
5. Category pairwise geometry (later)
```

---

## Files reference

| Path | Role |
|------|------|
| `data/all_contexts_public.csv` | Location / Activity |
| `embedding_drift_exploration/RESULTS.md` | Drift pilot summary |
| `embedding_visual_drift_pilot.py` | Longitudinal centroids (all / per cat) |
| `per_category_context_analysis.py` | Context strata + CCN merge |
| `analysis/ccn-2026/plotC_knn_diversity_outputs/` | CCN kNN / dispersion tables |

---

## One-paragraph integration takeaway

CCN establishes that categories like **cup** are **high exemplar variability** in a pooled snapshot; the drift pilot shows that cup embeddings **continue to move** across months within children; attaching **location and activity** labels (~10s resolution) lets us separate **kitchen-cup** from **living-room-cup** and test developmental questions without over-interpreting context-confounded prevalence. Together, static variability (CCN), longitudinal drift, and context-stratified geometry form a single developmental–ecological story rather than three disconnected analyses.
