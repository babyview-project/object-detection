# preprocessing/

Between detection and analysis: cropping, filtering, sampling, and small builders.

| Script | Does |
|---|---|
| `merge_bbox_pred_csv.py` | merges per-video YOLOE `bounding_box_predictions.csv` files |
| `crop_bbox.py`, `crop_bbox_from_csv.py`, `crop_object_mask.py` | crops from boxes / masks |
| `filter_blurry.py`, `filter_cropped_images_by_size.py` | blur (Laplacian variance) and size filters |
| `sample_object_crops_variability.py` | samples exemplars per category for crowdsourcing (writes to `annotation/precision_868h/`) |
| `sample_non_object_attention_tiles.py` | non-object attention-check tiles |
| `build_bv_validated_perceptual_sorting_assets.py` | assets for the perceptual free-sort experiment (`variability_exp/`) |
| `group_embeddings_by_agemo.py` | per category x child x age-in-months embedding averages |
| `make_category_rsa.py`, `make_category_rsa_from_doc.py` | early category RSA |
| `make_montage.py`, `visualize_mask.py` | QC images |
| `compare_skipped_and_expected_counts.py`, `update_csv.py` | bookkeeping |
| `merge_metadata_objects.Rmd` | joins detections with video metadata |

Most of these were written for the 868-hour release; the 2025.2 crops and embeddings come
from `image-embedding/run_cdi_crop_pipeline.py` instead.
