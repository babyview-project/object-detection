# depth/

Monocular depth for frames and detected objects (868-hour era, partly work in progress).

| Script | Does |
|---|---|
| `extract_depth.py` | depth maps from frames with a pre-trained model (saved as `.npy`) |
| `visualize_extracted_depth.py` | visual check of the saved maps |
| `create_depth_for_objects_masks_csv.py` | per detected object: 100 random points in its mask -> depth |
| `create_depth_for_objects_csv.py` | the same from boxes (marked work in progress) |
