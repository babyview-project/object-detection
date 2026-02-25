#!/usr/bin/env python3
"""
Sample 300 images per word from selected CDI semantic categories (animals, household,
furniture_rooms). Only includes images that appear in the CLIP-filtered embedding list
(removes false alarm detections). Saves sampled images to annotation/sampled_object_crops
and a CSV of source paths.
"""
from pathlib import Path
from typing import Optional
import shutil
import pandas as pd
import random

# Paths (relative to project root when run from repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CDI_VOCAB_CSV = PROJECT_ROOT / "data" / "cdi_words.csv"
CROPPED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class")
# CLIP-filtered list: one embedding path per line ( .../category/filename.npy ); only these stems are sampled
FILTER_LIST = PROJECT_ROOT / "data" / "clip_image_embeddings_doc_normalized_filtered-by-clip-0.26.txt"
OUTPUT_DIR = PROJECT_ROOT / "annotation" / "sampled_object_crops"
OUTPUT_CSV = PROJECT_ROOT / "annotation" / "sampled_object_crops.csv"
SEED_FILE = PROJECT_ROOT / "annotation" / "sampled_object_crops_seed.txt"

# Reproducibility
RANDOM_SEED = 42
N_PER_CATEGORY = 300

# CDI semantic categories to sample (word folders: english_gloss from these categories)
SEMANTIC_CATEGORIES = ("animals", "household", "furniture_rooms")

# Image extensions to consider
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_word_categories_from_cdi(cdi_csv: Path, semantic_categories: tuple[str, ...]) -> list[str]:
    """Return sorted list of english_gloss for the given CDI semantic categories."""
    df = pd.read_csv(cdi_csv)
    mask = df["category"].isin(semantic_categories)
    words = df.loc[mask, "english_gloss"].astype(str).str.strip()
    return sorted(words.unique().tolist())


def get_category_crop_dir(cropped_dir: Path, cat_name: str) -> Optional[Path]:
    """Return path to category subfolder (case-insensitive), or None if missing."""
    cat_lower = cat_name.strip().lower()
    direct = cropped_dir / cat_name
    if direct.exists() and direct.is_dir():
        return direct
    for p in cropped_dir.iterdir():
        if p.is_dir() and p.name.lower() == cat_lower:
            return p
    return None


def load_filtered_stems(filter_list_path: Path) -> set[tuple[str, str]]:
    """
    Stream the CLIP-filtered embedding list and return set of (category_lower, stem).
    Each line is a path like .../category/filename.npy; category and stem (no .npy) are extracted.
    """
    allowed = set()
    with open(filter_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.suffix.lower() == ".npy":
                stem = p.stem
            else:
                stem = p.name
            category = p.parent.name.strip().lower()
            allowed.add((category, stem))
    return allowed


def main():
    random.seed(RANDOM_SEED)

    # Load CLIP-filtered list: only sample images whose (category, stem) appear here
    if not FILTER_LIST.exists():
        raise FileNotFoundError(f"Filter list not found: {FILTER_LIST}")
    print(f"Loading filtered stems from {FILTER_LIST}...")
    allowed_stems = load_filtered_stems(FILTER_LIST)
    print(f"  Loaded {len(allowed_stems)} (category, stem) pairs from filter list")

    # Save seed for reproducibility
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SEED_FILE, "w") as f:
        f.write(str(RANDOM_SEED))

    word_categories = get_word_categories_from_cdi(CDI_VOCAB_CSV, SEMANTIC_CATEGORIES)
    print(f"Found {len(word_categories)} word categories in CDI ({SEMANTIC_CATEGORIES}): {word_categories}")

    rows = []
    for cat in word_categories:
        cat_dir = get_category_crop_dir(CROPPED_DIR, cat)
        if cat_dir is None:
            print(f"  Skip {cat}: no folder in cropped dir")
            continue

        all_paths = []
        for ext in IMAGE_EXTENSIONS:
            all_paths.extend(cat_dir.glob(f"*{ext}"))
        # Keep only images that appear in the CLIP-filtered list
        cat_lower = cat.strip().lower()
        paths = [p for p in all_paths if (cat_lower, p.stem) in allowed_stems]
        paths = sorted(paths, key=lambda p: p.name)

        if len(paths) == 0:
            print(f"  Skip {cat}: no filtered images (total in folder: {len(all_paths)})")
            continue

        n_sample = min(N_PER_CATEGORY, len(paths))
        sampled = random.sample(paths, n_sample)

        # One subfolder per word (e.g. annotation/sampled_object_crops/cat/, .../bottle/, .../chair/)
        out_cat_dir = OUTPUT_DIR / cat
        out_cat_dir.mkdir(parents=True, exist_ok=True)

        for src in sampled:
            dst = out_cat_dir / src.name
            shutil.copy2(src, dst)
            rows.append({"category": cat, "path": str(src.resolve())})

        print(f"  {cat}: sampled {n_sample} (filtered {len(paths)}, total in folder {len(all_paths)})")

    df = pd.DataFrame(rows, columns=["category", "path"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")
    print(f"Seed saved to {SEED_FILE}")


if __name__ == "__main__":
    main()
