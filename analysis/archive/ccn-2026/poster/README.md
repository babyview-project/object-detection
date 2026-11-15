# CCN 2026 poster draft

Print-ready draft styled after the **2025 Affinity Designer poster**
(`ccn_poster_vertical_20250808_jane.jpg`): navy header + lab logos, methods left /
results right, takeaway bar, bottom summary + QR.

Content and numbers match the camera-ready extended abstract
(`447_Variability_in_Young_Child_valid7018.tex`).

## Specs (CCN 2026)

- Board: 60″ W × 70″ H (1″ border)
- Recommended print: **58″ W × 68″ H**, portrait
- Distant readability (~6 ft): large type, light figure backgrounds

Source: https://2026.ccneuro.org/poster-presenter-instructions/

## Files

| File | Role |
|------|------|
| `ccn2026_poster.tex` | Main Overleaf / pdfLaTeX poster |
| `preview.html` | Browser layout preview (scaled down) |
| `ccn_poster_vertical_20250808_jane.jpg` | 2025 visual reference |
| `figures/ccn_2026_combined_06092026.pdf` | Camera-ready combined figure (source for A–C crops) |
| `figures/fig1A.{pdf,png}` | **A** Montage strip — cropped from combined |
| `figures/fig1B.{pdf,png}` | **B** DINOv3 t-SNE — cropped from combined |
| `figures/fig1C.{pdf,png}` | **C** Frequency vs global dispersion — cropped from combined |
| `figures/fig1D_cross_model.{pdf,png}` | **D** CLIP ↔ DINOv3 agreement (still from abstract_figures) |
| `figures/fig1E_cdi_groups.{pdf,png}` | Optional: CDI semantic-group summary |
| `figures/placeholders/aoa_vs_vc.{png,pdf}` | **E** In-progress CDI AoA vs \(V_c\) (fake cloud; replace) |
| `figures/placeholders/` | Logos, pipeline, example frame, QR, AoA |

Panels **A–C** are crops from `ccn_2026_combined_06092026.pdf` (also kept as
`ccn_2026_combined_06092026 (1).pdf`). Re-crop if you re-export the combined figure.

## Figure plan

| Panel | Why include | Status |
|-------|-------------|--------|
| Pipeline schematic | Matches 2025 methods storytelling | **Placeholder** — replace with final diagram |
| Example detections | Grounds the cohort in egocentric frames | **Placeholder** — use a real annotated BabyView frame |
| **A** Montages | Qualitative low→high variability | Cropped from combined PDF |
| **B** t-SNE | Shows within-category spread in embedding space | Cropped from combined PDF |
| **C** Frequency vs \(V_c\) | Main takeaway: frequency ≠ variability | Cropped from combined PDF |
| **D** Cross-model | Supports moderate CLIP–DINOv3 agreement | Ready (abstract_figures; not in combined) |
| **E** CDI groups | Optional extra if board space allows | Ready, not in main layout |
| Lab logos | Header branding like 2025 | **Placeholder** |
| QR | Link to OpenReview / OSF / abstract PDF | **Placeholder** |
| **E** AoA vs \(V_c\) | Signals next step (reviews: link to learning) | **Placeholder** (fake points) |

Also considered (not in layout): `fig_explore_frequency_vs_global_robustness_2x2`
(sensitivity check — better as a handout / supplement than on the board).

## Compile

**Overleaf (recommended):** create a project, upload `ccn2026_poster.tex` and the
`figures/` directory (include `placeholders/`), compile with pdfLaTeX.

**Local:**

```bash
cd analysis/ccn-2026/poster
pdflatex ccn2026_poster.tex
```

**Browser preview:** open `preview.html` (needs local file paths for figures).

## Content source of truth

Numbers and claims match `447_Variability_in_Young_Child_valid7018.tex`
(camera-ready): cross-model ρ = .55 / .63; frequency vs global dispersion
CLIP ρ = .18, DINOv3 ρ = .26; N = 7,018 crops / 85 CDI categories.

## Suggested edit passes before print

1. Drop in final **lab logos** and a real **annotated BabyView frame**.
2. Replace **QR** with OpenReview / OSF / PDF link when available; add board number.
3. Check montage axis labels remain readable at 58″ (re-export fig1A larger if not).
4. Optional: swap fig1B for `fig1B_valid7018_tsne_dinov3_semantic_diverse` if you want
   more CDI-group diversity in the t-SNE panel.
5. Optional Affinity pass: export PDF from Overleaf, then polish spacing in Affinity
   Designer to match the 2025 finish quality.
