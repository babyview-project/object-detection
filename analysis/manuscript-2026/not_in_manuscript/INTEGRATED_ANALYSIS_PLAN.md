# Integrated analysis plan: static variability (CCN) → longitudinal drift & context

**Status:** living document (May 2026)  
**Purpose:** Bridge the **CCN 2026 extended abstract** (cross-sectional within-category variability) with **in-progress developmental analyses** (embedding drift, context stratification, RDM linkage). Use this for VSS/preprint planning, collaborator alignment, and figure lists—even where analyses are incomplete.

**CCN reference:** `analysis/ccn-2026/ccn_extended_abstract (2).tex`  
**Drift results (pilot):** `embedding_drift_exploration/RESULTS.md`  
**Context + CCN merge:** `embedding_drift_exploration/CONTEXT_CCN_INTEGRATION.md`

---

## 1. One-sentence pitch

The CCN work asks **which categories are variable in a pooled snapshot** of children’s egocentric input; the extension asks **how that variability changes with age**, **whether it is confounded by where/when families film**, and **whether input drift tracks representational drift**—using the same embedding metrics (global dispersion, local structure) at two timescales.

---

## 2. Conceptual stack (three layers)

```text
Layer A — CCN (DONE, cross-sectional)
  Pooled exemplars per category (85 CDI categories, ~57–100 crops/cat)
  → global dispersion V_c, local kNN coherence
  → dissociation global vs local (ρ ≈ 0 in both CLIP & DINOv3)

Layer B — Developmental drift (PILOT, longitudinal)
  Per child × month × category: centroid + dispersion (reservoir 256 crops)
  → centroid displacement, dispersion_delta between consecutive months
  → age-shuffled null, RDM month-to-month linkage (top-8 densest children)

Layer C — Context / ecology (PILOT, stratified)
  Join Location + Activity (10s segments, ~99.6% match)
  → kitchen_like, living_room, eating, playing, outside
  → household (cup, plate), animals (dog, cat), vehicles (car, stroller)
```

**Key distinction (terminology):**

| Term | CCN (Layer A) | Drift (B/C) |
|------|----------------|-------------|
| **Global dispersion** | Mean distance to **pooled** category centroid across all exemplars in the CCN sample | Mean \(1-\cos(x,\mu)\) within **one child’s month** (and stratum), \(\mu\) = that window’s centroid |
| **Local coherence** | kNN distance (\(k{=}5\)) in pooled sample | *Not yet computed longitudinally* (planned: per-window kNN) |
| **“Geometry”** | Category layout in embedding space (static) | **Within-category** centroid + spread at age \(t\); displacement = change in centroid \(t \to t{+}1\) |

Dispersion and displacement in Layer B/C are **not pooled across ages** for estimation—only **summarized** across subjects in figures/tables.

---

## 3. How this extends the CCN abstract

### 3.1 What CCN already claims

- Everyday object categories differ in **global dispersion** and **local coherence**; the two metrics **dissociate** (especially in CLIP for global vs DINOv3 for both).
- Variability is theoretically linked to **learning difficulty** (frequency alone is insufficient).
- Limitation: **snapshot** of mostly Western, higher-SES homes; viewpoint + exemplar variability **confounded** in single crops.

### 3.2 What the new work adds (even if incomplete)

| Extension | Question | Status |
|-----------|----------|--------|
| **Longitudinal drift** | Does within-category embedding geometry **move** across months within the same child? | Pilot done (top-8, DINOv3) |
| **Static ↔ dynamic link** | Do high–CCN-variability categories also **drift more** month-to-month? | Partial (category-level scatter started; not in paper) |
| **Input ↔ representation** | When input drifts more, is the **category RDM** less stable? | Pilot: \(r \approx -0.75\) displacement vs RDM \(r\) |
| **Context stratification** | Is cup drift **kitchen-cup** vs **living-room-cup**? | Pilot done (cup/plate/dog/cat/car/stroller) |
| **Ecological confounds** | Can we separate prevalence (`prop_cup`) from **geometry given context**? | In progress (stratified counts + geometry) |
| **Pseudo-instances** | Multiple cups per month—one cluster or many? | **Planned** |
| **Category layout over time** | Pairwise category geometry (RDM of centroids) | **Deferred** (after per-category stable) |

### 3.3 Draft “future directions” paragraph (for CCN discussion or follow-on abstract)

> Our cross-sectional metrics characterize variability in a pooled sample of children’s detections, but children’s experiences change over the second year of life. In dense longitudinal subsets of BabyView, monthly within-category centroids in a self-supervised embedding space (DINOv3) show systematic **centroid displacement** (e.g., cup \(\approx 0.22\) vs global pooled input \(\approx 0.10\) on a \(1-\cos\) scale), with smoother trajectories than age-shuffled nulls (\(p \approx 0.004\)). Months with larger input drift co-occur with lower month-to-month RDM stability (\(r \approx -0.75\)). Human-coded location and activity labels suggest that household-object geometry differs by context (e.g., cup dispersion and displacement are elevated during **eating** segments relative to kitchen-only pooling). Together, static variability rankings and longitudinal, context-stratified drift offer complementary constraints on which categories should be easy or hard to learn—and **when** representational change is likely to occur.

---

## 4. Research questions & hypotheses

### RQ1 — Static variability (CCN; **addressed**)

**Q:** Which categories are globally dispersed vs locally coherent in CLIP and DINOv3?  
**H:** Global and local metrics dissociate; DINOv3 shows wider absolute dispersion; CLIP/DINOv3 agree more on local than global rankings.

### RQ2 — Longitudinal within-category drift (**pilot**)

**Q:** Does within-category geometry change month-to-month within children?  
**H:** Yes; displacement exceeds global pooled drift for variable categories (cup, bottle); hand/body-part categories drift less.  
**Evidence (top-8, monthly, DINOv3):** global mean displacement 0.101; cup 0.216; age-shuffled null higher than real (\(p \approx 0.004\)).

### RQ3 — Static ↔ longitudinal coupling (**partial**)

**Q:** Do categories with high CCN `global_dispersion` also show high mean monthly dispersion or displacement?  
**H:** Positive association, stronger in DINOv3 than CLIP; exceptions where ecology dominates (e.g., `hand`).  
**To do:** Formal scatter across all valid129 categories; partial correlation controlling for detection count.

### RQ4 — Input drift ↔ representational drift (**pilot**)

**Q:** When visual input shifts more, does the category-level RDM correlate less with the previous month?  
**H:** Negative coupling (larger input displacement → lower RDM \(r\)).  
**Evidence:** \(r \approx -0.75\) for displacement vs month-to-month RDM correlation (96 edges, top-8).

### RQ5 — Context and confounds (**pilot**)

**Q:** Are cup/plate geometry and drift driven by **where** families film (kitchen vs living room) and **what** they are doing (eating vs playing)?  
**H:** Stratum-specific centroids differ; eating segments show high cup displacement despite tighter dispersion; vehicles often appear indoors (stroller in living room), so `outside` is sparse for car.  
**Evidence (illustrative, `dispersion_displacement_link_top8.csv`):**

| Category | Stratum | Mean dispersion | Mean displacement |
|----------|---------|-----------------|-------------------|
| cup | all | 0.40 | 0.22 |
| cup | eating | 0.34 | **0.32** |
| plate | living_room | 0.37 | **0.37** |
| cat | all | 0.35 | 0.14 |
| car | all | 0.37 | 0.42 (very few edges) |

### RQ6 — Exemplar multiplicity (**planned**)

**Q:** Within `(child, month, category, stratum)`, are crops one tight cluster or several (pseudo-instances)?  
**H:** High-CCN categories show multi-modal within-month structure; cluster count or silhouette increases with age for cup.  
**Method sketch:** k-means or HDBSCAN on L2-normalized crops; compare to THINGS/BV CCN exemplar structure.

### RQ7 — Category pairwise geometry (**deferred**)

**Q:** Does the **layout** among category centroids change with age (not just each category’s internal spread)?  
**H:** RDM of monthly category centroids correlates with existing developmental trajectory RDM analyses.  
**Note:** Explicitly **after** per-category centroid/dispersion and context strata are stable.

---

## 5. Analysis modules (implementation checklist)

| ID | Module | Script / location | Status | Priority |
|----|--------|-------------------|--------|----------|
| A1 | CCN global dispersion + kNN | `analysis/ccn-2026/plotC_knn_diversity_outputs/` | **Done** | — |
| A2 | CCN figures (montage, t-SNE, dissociation plot) | `ccn-2026-draft/figure/` | **Done** | — |
| B1 | Global + per-category monthly drift | `embedding_visual_drift_pilot.py` | **Done** (top-8) | P1: expand cohort |
| B2 | Age-shuffled null, 3-month bins, CLIP compare | `embedding_drift_extensions.py` | **Done** (top-8) | P2 |
| B3 | Drift ↔ RDM merge | `embedding_drift_extensions.py` | **Done** (top-8) | P1: covariates |
| C1 | Context join + strata | `per_category_context_analysis.py` | **Done** (top-8) | P1 |
| C2 | Cup/plate/dog/cat/car/stroller panels | `context_merged/figures/` | **Done** | P2: refine outside stratum |
| D1 | CCN ↔ mean longitudinal dispersion scatter (all categories) | *not scripted* | **Todo** | P1 |
| D2 | Per-window local kNN (parallel CCN local metric) | *not scripted* | **Todo** | P2 |
| E1 | Pseudo-instance clustering within month | *not scripted* | **Todo** | P2 |
| E2 | Link clusters to detection bbox size / co-objects | *not scripted* | **Todo** | P3 |
| F1 | Expand beyond top-8 (`--top-n 0` or quartiles) | pilot flags | **Todo** | P1 |
| F2 | Hours-filmed / mobility covariates in drift–RDM model | RDM CSV has hours | **Todo** | P1 |
| G1 | Monthly category-centroid RDM vs existing trajectory | *deferred* | **Backlog** | P3 |

---

## 6. Unified methods paragraph (for a future Methods section)

We analyzed human-validated object detections from egocentric BabyView recordings (CDI-aligned categories; detection threshold 0.27). **Cross-sectional variability (CCN):** For each of 85 categories, we sampled 57–100 crops and computed (i) **global dispersion**—mean distance to the category centroid in CLIP or DINOv3—and (ii) **local coherence**—mean within-category \(k\)-NN distance (\(k{=}5\)). **Longitudinal input geometry (pilot):** For each child, calendar month, and category (and optionally location/activity stratum), we reservoir-sampled up to 256 crop embeddings, computed a unit-norm **centroid** \(\mu_{c,t}\) and **dispersion** \(\mathbb{E}[1-\cos(x,\mu_{c,t})]\). **Drift** was \(1-\cos(\mu_{c,t},\mu_{c,t+1})\) for consecutive months with sufficient crops (global minimum 30 crops; per-category 15). **Context:** Location and activity labels on 10s segments were joined via `video_id` derived from frame index (\(\approx\)99.6% match rate for cup). **Representational stability:** Month-to-month RDM correlations from developmental trajectory analyses were merged on (child, age) pairs. Analyses initially focused on the eight densest recorders (9–29 months) for parity with existing RDM pipelines; cohort expansion is planned.

---

## 7. Figure & table plan (integrated storyline)

### Track 1 — CCN poster / proceedings (static)

1. Low vs high variability montages (Fig 1A).  
2. t-SNE exemplar clouds (Fig 1B).  
3. Global dispersion vs local coherence dissociation (Fig 1C).  
4. Optional inset: category ranking table (top/bottom 5 each metric).

### Track 2 — Development + context (new; partial)

5. **Global drift trajectories** — per child + mean (`global_drift_trajectories_dinov3_*`).  
6. **Per-category drift** — cup vs hand vs bottle (`category_drift_trajectories_*`).  
7. **Null model** — real vs age-shuffled (`null_age_shuffle_*`).  
8. **Input vs RDM** — displacement vs month-to-month RDM \(r\) (`drift_vs_rdm_*`).  
9. **Context panels** — cup/plate kitchen & living room; dog/cat & car/stroller (`context_merged/figures/`).  
10. **Domain summary** — three-row displacement + dispersion (`domain_context_summary_top8.png`).

### Track 3 — Planned (not yet figures)

11. Scatter: CCN `global_dispersion` vs mean child-month **dispersion** or **displacement** (all valid129).  
12. Pseudo-instance montage: cup at 12 vs 18 mo, kitchen_like, 2–3 clusters.  
13. Partial regression: drift ~ CCN variability + log(n_detections) + hours filmed.

---

## 8. Draft Results outline (follow-on paper / VSS)

### § Static snapshot (replicate CCN briefly)

- 85 categories; dissociation global vs local; exemplar categories (hand vs dish/swing/stick).  
- One sentence pointing to developmental extension.

### § Longitudinal within-category drift (pilot)

- Global displacement small but autocorrelated; cup/bottle > global.  
- Age-shuffled null: real trajectories smoother than chance.  
- DINOv3 preferred over CLIP for month-level input drift.

### § Coupling to representational change

- Strong negative association with RDM stability; interpret as coordinated input/representation reorganization, not causal claim.

### § Context-stratified geometry

- Household: eating vs kitchen vs living room effects on cup/plate.  
- Animals/vehicles: sparse outside; stroller dominated by indoor contexts.  
- Prevalence tables per stratum to show confound reduction vs raw `prop_*`.

### § Planned / incomplete (label clearly in draft)

- Pseudo-instances; full-cohort replication; local kNN longitudinal; category-layout RDM.

---

## 9. Limitations (carry across CCN + extension)

1. **Detection-biased** — only frames with objects; not full visual diet.  
2. **Western, higher-SES** sample (CCN + dense longitudinal subset).  
3. **Viewpoint + exemplar confound** in single crops (CCN); partially addressed by context labels, not gaze/instance IDs.  
4. **Top-8 longitudinal** — may overrepresent families who film often.  
5. **CLIP vs DINOv3** — different constructs; use DINOv3 for month-level drift, CLIP for semantic/global CCN story.  
6. **Category base rates** — high-displacement categories may reflect diverse contexts, not purely “learning cup.”

---

## 10. Suggested writing targets

| Venue | Emphasis | Use from this plan |
|-------|----------|-------------------|
| **CCN 2026 (current)** | Static variability only | §3.1; optional one-sentence “future work” from §3.3 |
| **CCN poster add-on** | One panel: cup drift + kitchen stratum | Figures 5–9 from §7 |
| **VSS / preprint** | Full stack A→B→C + RDM | §6–8 as backbone |
| **Grant / internal** | Modules D–G timeline | §5 checklist |

---

## 11. Immediate next steps (ordered)

1. **Cohort expansion:** Run `embedding_visual_drift_pilot.py` and `per_category_context_analysis.py` with `--top-n 0` or predefined quartiles; document sensitivity.  
2. **CCN ↔ drift scatter:** All valid129 categories; report Spearman \(\rho\) for CCN dispersion vs mean displacement and vs mean monthly dispersion.  
3. **Covariate model:** Drift ~ displacement + hours + `n_indexed` (+ stratum fixed effects).  
4. **Pseudo-instance pilot:** One category (cup), one stratum (kitchen_like), 2–3 children × 3 ages.  
5. **Draft VSS abstract** using §3.3 + §6 once (1)–(2) are stable.

---

## 12. File index

| Path | Role |
|------|------|
| `analysis/ccn-2026/ccn_extended_abstract (2).tex` | CCN submission text |
| `analysis/ccn-2026/plotC_knn_diversity_outputs/` | Pooled dispersion / kNN tables |
| `analysis/manuscript-2026/not_in_manuscript/embedding_visual_drift_pilot.py` | Longitudinal centroids |
| `analysis/manuscript-2026/not_in_manuscript/embedding_drift_extensions.py` | Null, RDM, CLIP |
| `analysis/manuscript-2026/not_in_manuscript/per_category_context_analysis.py` | Context strata |
| `embedding_drift_exploration/RESULTS.md` | Pilot numbers + figures |
| `embedding_drift_exploration/CONTEXT_CCN_INTEGRATION.md` | Join rules + merge logic |
| **This file** | Integrated plan + draft language |

---

*Last updated: May 2026. Update §5 status column as modules complete.*
