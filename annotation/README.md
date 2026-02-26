# Crop annotation tool

GUI for annotating cropped object images (correct category and CDI semantic category). See `annotate_crops.py` for usage.

## Environment setup

### Option A: Conda (recommended)

From the **object-detection** project root:

```bash
conda env create -f annotation/environment.yml
conda activate crop-annotation
cd annotation && python annotate_crops.py
```

### Option B: pip only

```bash
pip install -r annotation/requirements.txt
python annotation/annotate_crops.py
```

Or from the `annotation` folder:

```bash
pip install -r requirements.txt
python annotate_crops.py
```

## Files

- **annotate_crops.py** – annotation GUI
- **requirements.txt** – pip dependencies (Pillow)
- **environment.yml** – conda env definition for `crop-annotation`
