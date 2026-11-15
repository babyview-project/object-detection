**Supplementary Table Y.** Subject split (BabyView split by child, VEDB by session, THINGS by image): dataset differences in CDI-domain cluster strength on the split-averaged cross-half RDM (115 categories). Difference [95% category-bootstrap CI], two-sided p.

| Model | Contrast | AUC | Δ / SD | Raw Δ (cosine) |
|---|---|---|---|---|
| CLIP | BV − THINGS | -0.039 [-0.084, +0.007], p = .104 | -0.243 [-0.405, -0.083], p = .002 | +0.054 [+0.022, +0.086], p = .002 |
| CLIP | BV − VEDB | +0.000 [-0.027, +0.028], p = .930 | -0.014 [-0.117, +0.090], p = .800 | +0.025 [+0.006, +0.046], p = .011 |
| CLIP | VEDB − THINGS | -0.039 [-0.080, +0.004], p = .068 | -0.229 [-0.398, -0.076], p < .001 | +0.029 [+0.001, +0.056], p = .042 |
| DINOv3 | BV − THINGS | +0.053 [+0.006, +0.102], p = .028 | +0.134 [-0.033, +0.311], p = .125 | +0.100 [+0.072, +0.128], p < .001 |
| DINOv3 | BV − VEDB | -0.016 [-0.048, +0.017], p = .356 | -0.067 [-0.176, +0.042], p = .238 | +0.013 [-0.004, +0.033], p = .136 |
| DINOv3 | VEDB − THINGS | +0.069 [+0.020, +0.120], p = .002 | +0.201 [+0.035, +0.373], p = .021 | +0.087 [+0.063, +0.110], p < .001 |

**Supplementary Table Z.** Same-domain coefficient (rank regression of category-pair distance on same-domain membership; larger = tighter clusters) without and with controls for frame co-occurrence and shared-context similarity (BabyView co-occurrence), and AUC against LVIS. Difference [95% CI], p.

| Model | Contrast | Categories | Domain coefficient, no control | Domain coefficient, co-occurrence controlled | AUC |
|---|---|---|---|---|---|
| CLIP | BV − THINGS | 129 | -0.059 [-0.105, -0.005], p = .034 | -0.082 [-0.135, -0.026], p = .002 | — |
| CLIP | BV − VEDB | 124 | +0.001 [-0.033, +0.036], p = .936 | -0.010 [-0.045, +0.024], p = .536 | — |
| CLIP | VEDB − THINGS | 124 | -0.055 [-0.101, -0.005], p = .026 | -0.068 [-0.115, -0.015], p = .012 | — |
| CLIP | BV − LVIS | 104 | +0.130 [+0.064, +0.193], p < .001 | +0.098 [+0.035, +0.157], p < .001 | +0.097 [+0.049, +0.147], p < .001 |
| CLIP | THINGS − LVIS | 104 | +0.207 [+0.128, +0.271], p < .001 | +0.203 [+0.127, +0.274], p < .001 | +0.155 [+0.098, +0.207], p < .001 |
| CLIP | VEDB − LVIS | 104 | +0.127 [+0.054, +0.195], p < .001 | +0.114 [+0.042, +0.182], p < .001 | +0.095 [+0.041, +0.149], p < .001 |
| DINOv3 | BV − THINGS | 129 | +0.068 [+0.015, +0.124], p = .010 | +0.023 [-0.031, +0.080], p = .456 | — |
| DINOv3 | BV − VEDB | 124 | -0.005 [-0.043, +0.034], p = .806 | -0.023 [-0.062, +0.017], p = .248 | — |
| DINOv3 | VEDB − THINGS | 124 | +0.079 [+0.021, +0.136], p = .008 | +0.050 [-0.013, +0.112], p = .120 | — |
| DINOv3 | BV − LVIS | 104 | +0.082 [+0.022, +0.142], p = .006 | +0.038 [-0.023, +0.103], p = .250 | +0.062 [+0.017, +0.109], p = .006 |
| DINOv3 | THINGS − LVIS | 104 | -0.014 [-0.081, +0.053], p = .628 | -0.000 [-0.073, +0.074], p = .960 | -0.010 [-0.062, +0.040], p = .628 |
| DINOv3 | VEDB − LVIS | 104 | +0.083 [+0.020, +0.150], p = .010 | +0.062 [-0.005, +0.136], p = .074 | +0.062 [+0.015, +0.114], p = .010 |
