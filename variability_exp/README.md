# variability_exp/

Perceptual free-sort experiment (Brady-lab style) on rater-validated BabyView crops. Not
tracked in git except this README.

| File | What |
|---|---|
| `bv_validated_perceptual_sorting.html`, `bradylab_perceptual_sorting_template.html` | the task page and the template it was built from |
| `bv_validated_crops/` | crop lists per stream and practice set, build manifest (git-ignored: file names carry subject ids and dates) |

Built by `preprocessing/build_bv_validated_perceptual_sorting_assets.py`. The crop images
(`bv_validated_crops/media/`) were links into `/data2/.../yoloe_cdi_all_cropped_by_class/`;
`/data2` is gone and the links were removed on 2026-10-22, so the assets must be rebuilt
from another crop source before the task can run.
