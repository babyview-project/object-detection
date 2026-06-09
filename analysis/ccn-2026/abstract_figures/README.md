# CCN 2026 abstract figure panels

Individual panels for manual layout in the submission PDF.

| File | Content |
|------|---------|
| `fig1A_valid7018_montages_low_to_high_global.*` | Montages: clock → oven → chair → paper → book ($V_c$ from DINOv3) |
| `fig1B_valid7018_tsne_dinov3.*` | DINOv3 t-SNE — same 5 montage categories (low→high global dispersion) |
| `fig1B_valid7018_tsne_dinov3_semantic_diverse.*` | Alternative t-SNE — pillow, sweater, doll, window, slide (one per CDI group) |
| `fig1C_valid7018_frequency_vs_global_dispersion.*` | Full infant-view frame prevalence vs DINOv3 global dispersion |
| `fig_explore_frequency_vs_global_robustness_2x2.*` | Frame prevalence (full infant-view) vs valid7018 dispersion (CLIP + DINOv3) |
| `valid7018_figure_category_selection.csv` | Category picks (see `valid7018_paper_stats.json`) |

Regenerate: `generate_valid7018_paper_figures.py --from-zip`
