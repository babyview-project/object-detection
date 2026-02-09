"""
Load THINGS embeddings for within-category variability.
Both DinoV3 and CLIP use directory-of-.npy layout: {dir}/{category}/*.npy (one .npy per exemplar).

Paths (per-exemplar):
- DinoV3: .../image_embeddings/.../facebook_dinov3-vitb16-pretrain-lvd1689m
- CLIP:   .../clip_image_embeddings_npy_by_category  (by_category = folder per category, .npy per exemplar)
"""
from pathlib import Path
from collections import defaultdict
import numpy as np

# Default paths (per-image exemplars, not averaged)
THINGS_DINOV3_DIR = Path("/ccn2/dataset/babyview/outputs_20250312/image_embeddings/things_bv_overlapping_categories_corrected/facebook_dinov3-vitb16-pretrain-lvd1689m")
THINGS_CLIP_DOCS = Path("/ccn2/dataset/babyview/outputs_20250312/things_bv_overlapping_categories_corrected/embeddings/image_embeddings/clip_image_embeddings_doc_normalized_filtered-by-clip-26.docs")
THINGS_CLIP_NPY_DIR = Path("/ccn2/dataset/babyview/outputs_20250312/things_bv_overlapping_categories_corrected/embeddings/image_embeddings/clip_image_embeddings_npy_by_category")


def count_npy_per_category(dir_path, max_categories=10):
    """
    Diagnostic: count .npy files per category folder (recursive **/*.npy).
    Returns list of (category_name, count). Use to verify per-exemplar layout.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    out = []
    for cat_folder in sorted(dir_path.iterdir()):
        if not cat_folder.is_dir():
            continue
        n = len(list(cat_folder.glob("**/*.npy")))
        out.append((cat_folder.name, n))
        if len(out) >= max_categories:
            break
    return out


def load_things_dinov3_from_dir(
    dir_path,
    allowed_categories=None,
    min_exemplars=2,
):
    """
    Load THINGS DinoV3 from directory: {dir_path}/{category}/*.npy (one .npy per image).
    Returns category_embeddings, category_exemplar_ids (same format as BV loader).
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"THINGS DinoV3 dir not found: {dir_path}")
    category_embeddings = {}
    category_exemplar_ids = {}
    for cat_folder in sorted(dir_path.iterdir()):
        if not cat_folder.is_dir():
            continue
        cat_name = cat_folder.name
        if allowed_categories is not None and cat_name not in allowed_categories:
            continue
        embs = []
        ids = []
        # Recursive: find all .npy under this category (direct or in subdirs)
        for f in sorted(cat_folder.glob("**/*.npy")):
            try:
                e = np.load(f)
                e = np.asarray(e, dtype=np.float64).flatten()
                embs.append(e)
                ids.append((f.stem, None))
            except Exception:
                continue
        if len(embs) >= min_exemplars:
            category_embeddings[cat_name] = np.array(embs)
            category_exemplar_ids[cat_name] = ids
    return category_embeddings, category_exemplar_ids


def load_things_clip_from_docs(
    docs_path,
    allowed_categories=None,
    min_exemplars=2,
):
    """
    Load THINGS CLIP from a .docs file using vislearnlabpy EmbeddingStore.
    Requires: conda activate vislearnlabpy
    Returns category_embeddings, category_exemplar_ids.
    """
    from vislearnlabpy.embeddings.embedding_store import EmbeddingStore

    docs_path = Path(docs_path)
    if not docs_path.exists():
        raise FileNotFoundError(f"THINGS CLIP .docs not found: {docs_path}")
    store = EmbeddingStore.from_doc(str(docs_path))
    # EmbeddingList has .normed_embedding and .text (category)
    embeddings = np.array(store.EmbeddingList.normed_embedding)
    categories = np.array(store.EmbeddingList.text)
    # group by category
    cat_to_rows = defaultdict(list)
    for i, cat in enumerate(categories):
        c = str(cat).strip().lower() if cat is not None else ""
        if allowed_categories is not None and c not in allowed_categories:
            continue
        cat_to_rows[c].append(i)
    category_embeddings = {}
    category_exemplar_ids = {}
    for cat, indices in cat_to_rows.items():
        if len(indices) < min_exemplars:
            continue
        X = embeddings[indices].astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        category_embeddings[cat] = X
        category_exemplar_ids[cat] = [(f"{cat}_{i}", None) for i in range(len(indices))]
    return category_embeddings, category_exemplar_ids


def load_things_clip_from_dir(
    dir_path,
    allowed_categories=None,
    min_exemplars=2,
):
    """
    Load THINGS CLIP from directory: {dir_path}/{category}/*.npy (one .npy per exemplar).
    Same layout as DinoV3; same return format.
    """
    return load_things_dinov3_from_dir(dir_path, allowed_categories, min_exemplars)
