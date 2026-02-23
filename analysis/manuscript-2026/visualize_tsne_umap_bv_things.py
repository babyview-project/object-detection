#!/usr/bin/env python3
"""
Produce 2D coordinate CSVs and line/arrow plots for BabyView vs THINGs embeddings (UMAP and t-SNE).
Uses CSV embedding files from data/embeddings/. Outputs are written to
bv_things_results_{clip,dinov3}/ with:
  - {umap,tsne}_bv_things_coordinates.csv
  - {umap,tsne}_bv_things_lines.png/.pdf (arrow BV → THINGs, colored by CDI)
  - {umap,tsne}_bv_things_lines_labeled.png/.pdf (labels at line midpoints)
  - {umap,tsne}_bv_things_lines_dots_labeled.png/.pdf (selected labels at dots, not on lines)
  - {umap,tsne}_bv_things_lines_exemplars.png/.pdf (optional: with BV/THINGs exemplar thumbnails)

Each CSV has columns: category (word), cdi_category, bv_*_x, bv_*_y, things_*_x, things_*_y.

Run from manuscript-2026/ or set DATA_DIR and OUT_BASE as needed.
Exemplar plots require --cropped-dir and --things-images-dir.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mimg
from PIL import Image

# Paths: script lives in object-detection/analysis/manuscript-2026/
SCRIPT_DIR = Path(__file__).resolve().parent
OBJECT_DETECTION = SCRIPT_DIR.parent.parent  # analysis -> object-detection
DATA_DIR = OBJECT_DETECTION / "data"
EMBED_DIR = DATA_DIR / "embeddings"
CDI_PATH = DATA_DIR / "cdi_words.csv"
OUT_BASE = SCRIPT_DIR  # bv_things_results_* will be created here


def _blend_with_white(color, white_frac=0.5):
    """Blend a color with white. Returns rgba tuple."""
    rgba = np.array(mcolors.to_rgba(color))
    w = min(1.0, max(0.0, white_frac))
    blended = (1 - w) * rgba + w * np.array([1.0, 1.0, 1.0, rgba[3]])
    return tuple(blended)


def _harmonious_cdi_palette(n_categories):
    """Return a list of n_categories soft, harmonious colors for CDI categories."""
    if n_categories <= 0:
        return []
    base_hex = [
        "#2d7d7d", "#6b9080", "#c97b84", "#5c6b73", "#c9a227",
        "#b8736b", "#5b8a9e", "#7d8c4a", "#8b6b8b", "#4a7c6b",
        "#a65d4a", "#3d6b7c", "#9a8c5a", "#7c5a7c", "#5a7c6a",
    ]
    if n_categories <= len(base_hex):
        return [mcolors.to_rgba(c) for c in base_hex[:n_categories]]
    out = [mcolors.to_rgba(c) for c in base_hex]
    for i in range(n_categories - len(base_hex)):
        out.append(mcolors.to_rgba(base_hex[i % len(base_hex)]))
    return out[:n_categories]


def _indices_to_label(n: int, cdi_cats: list, max_labels: int = 12):
    """Return indices to label: one per CDI category (first occurrence), or spread if no CDI."""
    if cdi_cats and set(cdi_cats) != {"unknown"}:
        unique_cdi = sorted(set(cdi_cats))
        indices = []
        for cat in unique_cdi:
            for i in range(n):
                if cdi_cats[i] == cat:
                    indices.append(i)
                    break
        return indices
    step = max(1, n // max_labels)
    return list(range(0, n, step))[:max_labels]


def _get_category_crop_dir(cropped_dir: Path, cat_name: str) -> Path:
    """Return path to category subfolder (match by name, case-insensitive)."""
    cat_lower = cat_name.strip().lower()
    direct = cropped_dir / cat_name
    if direct.exists() and direct.is_dir():
        return direct
    for p in cropped_dir.iterdir():
        if p.is_dir() and p.name.lower() == cat_lower:
            return p
    return cropped_dir / cat_name


def _resize_to_fixed(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize image array to (size, size) so all exemplars display at the same size."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    pil = pil.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(pil)


def _get_one_bv_crop_path(cropped_dir: Path, cat_name: str) -> Path | None:
    """Return path to one BV cropped image in category folder, or None."""
    cat_dir = _get_category_crop_dir(cropped_dir, cat_name)
    if not cat_dir.exists():
        return None
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in cat_dir.glob(ext):
            return p
    return None


_things_image_list_cache: dict = {}


def _get_things_image_by_index(things_images_dir: Path, cat_name: str, index: int) -> Path | None:
    """Return path to the index-th image in category (sorted by name), or None."""
    key = (Path(things_images_dir), cat_name.strip().lower())
    if key not in _things_image_list_cache:
        cat_dir = _get_category_crop_dir(things_images_dir, cat_name)
        if not cat_dir.exists():
            _things_image_list_cache[key] = []
        else:
            paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                paths.extend(cat_dir.glob(ext))
            _things_image_list_cache[key] = sorted(paths, key=lambda p: p.name)
    paths = _things_image_list_cache[key]
    if index < 0 or index >= len(paths):
        return None
    return paths[index]


def load_embeddings(bv_path: Path, things_path: Path, bv_word_col: str, things_word_col: str):
    """Load BV and THINGs embedding CSVs and align by shared words. Returns (words, cdi_cats, bv_mat, things_mat)."""
    bv = pd.read_csv(bv_path)
    th = pd.read_csv(things_path)
    bv_word_col = bv_word_col if bv_word_col in bv.columns else bv.columns[0]
    bv_words = bv[bv_word_col].astype(str).str.strip().str.lower()
    bv = bv.set_index(bv_words)
    th_words = th[things_word_col].astype(str).str.strip().str.lower()
    th = th.set_index(th_words)
    # Numeric columns only
    bv_mat = bv.select_dtypes(include=[np.number])
    th_mat = th.select_dtypes(include=[np.number])
    # Align by shared words
    common = bv_mat.index.intersection(th_mat.index).unique()
    bv_mat = bv_mat.loc[common].values
    th_mat = th_mat.loc[common].values
    words = list(common)
    # CDI category: try uni_lemma then english_gloss to match project CDI usage
    cdi = pd.read_csv(CDI_PATH)
    word_to_cat = {}
    if "uni_lemma" in cdi.columns and "category" in cdi.columns:
        for _, row in cdi.iterrows():
            w = str(row["uni_lemma"]).strip().lower()
            word_to_cat[w] = row["category"]
    if "english_gloss" in cdi.columns and "category" in cdi.columns:
        for _, row in cdi.iterrows():
            w = str(row["english_gloss"]).strip().lower()
            word_to_cat.setdefault(w, row["category"])
    cdi_cats = [word_to_cat.get(w, "unknown") for w in words]
    return words, cdi_cats, bv_mat.astype(np.float64), th_mat.astype(np.float64)


def fit_and_save_coordinates(
    words: list,
    cdi_cats: list,
    bv_mat: np.ndarray,
    things_mat: np.ndarray,
    out_dir: Path,
    method: str,
    prefix: str,
    random_state: int = 42,
    cropped_dir: Path | None = None,
    things_images_dir: Path | None = None,
):
    """Stack BV and THINGs rows, fit 2D reduction, write CSV with bv_*_x/y and things_*_x/y."""
    n = len(words)
    # Stack so row i = BV for word i, row n+i = THINGs for word i  -> (2*n, dim)
    stacked = np.vstack([bv_mat, things_mat])
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")
        n_neighbors = min(15, max(2, 2 * n - 1))
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=0.1)
        coords_2d = reducer.fit_transform(stacked)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(1, 2 * n - 1))  # 2*n points in stacked
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        coords_2d = reducer.fit_transform(stacked)
    else:
        raise ValueError("method must be 'umap' or 'tsne'")
    bv_x = coords_2d[:n, 0]
    bv_y = coords_2d[:n, 1]
    things_x = coords_2d[n:, 0]
    things_y = coords_2d[n:, 1]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "category": words,
        "cdi_category": cdi_cats,
        f"bv_{prefix}_x": bv_x,
        f"bv_{prefix}_y": bv_y,
        f"things_{prefix}_x": things_x,
        f"things_{prefix}_y": things_y,
    })
    out_path = out_dir / f"{prefix}_bv_things_coordinates.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    # Line/arrow plots (BV → THINGs), colored by CDI when available
    _plot_bv_things_lines(
        words, cdi_cats, bv_x, bv_y, things_x, things_y,
        prefix, method_label=method.upper(), out_dir=out_dir,
    )
    _plot_bv_things_lines_labeled(
        words, cdi_cats, bv_x, bv_y, things_x, things_y,
        prefix, method_label=method.upper(), out_dir=out_dir,
    )
    _plot_bv_things_lines_dots_labeled(
        words, cdi_cats, bv_x, bv_y, things_x, things_y,
        prefix, method_label=method.upper(), out_dir=out_dir,
    )
    if cropped_dir is not None and things_images_dir is not None and cropped_dir.exists() and things_images_dir.exists():
        _plot_bv_things_lines_with_exemplars(
            words, cdi_cats, bv_x, bv_y, things_x, things_y,
            prefix, method_label=method.upper(), out_dir=out_dir,
            cropped_dir=cropped_dir, things_images_dir=things_images_dir,
        )
    return out_path


def _plot_bv_things_lines(
    words: list,
    cdi_cats: list,
    bv_x: np.ndarray,
    bv_y: np.ndarray,
    things_x: np.ndarray,
    things_y: np.ndarray,
    prefix: str,
    method_label: str,
    out_dir: Path,
):
    """Draw arrows BV → THINGs, scatter points, optional CDI coloring. Save PNG/PDF."""
    n = len(words)
    unique_cdi = sorted(set(cdi_cats)) if cdi_cats else []
    if unique_cdi and unique_cdi != ["unknown"]:
        palette = _harmonious_cdi_palette(len(unique_cdi))
        cdi_to_color = {c: palette[i] for i, c in enumerate(unique_cdi)}
        line_colors = [_blend_with_white(cdi_to_color.get(c, "#888888"), 0.5) for c in cdi_cats]
        dot_colors = [cdi_to_color.get(c, "#888888") for c in cdi_cats]
    else:
        cdi_to_color = {}
        line_colors = ["#6b6b6b"] * n
        dot_colors = None

    fig, ax = plt.subplots(figsize=(16, 14))
    for i in range(n):
        ax.annotate(
            "",
            xy=(things_x[i], things_y[i]),
            xytext=(bv_x[i], bv_y[i]),
            arrowprops=dict(arrowstyle="->", color=line_colors[i], lw=1.0, alpha=0.6),
        )
    if dot_colors is not None:
        ax.scatter(bv_x, bv_y, c=dot_colors, s=90, marker="o", label="BabyView (BV)", edgecolors="none", alpha=0.9, zorder=3)
        ax.scatter(things_x, things_y, c=dot_colors, s=90, marker="s", label="THINGs", edgecolors="none", alpha=0.9, zorder=3)
    else:
        ax.scatter(bv_x, bv_y, c="#c45c3e", s=90, marker="o", label="BabyView (BV)", edgecolors="none", alpha=0.9, zorder=3)
        ax.scatter(things_x, things_y, c="#3d6cb9", s=90, marker="s", label="THINGs", edgecolors="none", alpha=0.9, zorder=3)

    if unique_cdi and unique_cdi != ["unknown"]:
        legend_elements = [
            Line2D([0], [0], color="#555555", marker="o", linestyle="None", markersize=10, markeredgecolor="none", label="BabyView (BV)"),
            Line2D([0], [0], color="#555555", marker="s", linestyle="None", markersize=10, markeredgecolor="none", label="THINGs"),
        ]
        for cat in unique_cdi:
            legend_elements.append(Line2D([0], [0], color=cdi_to_color[cat], lw=3, label=cat))
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title(
        f"{method_label}: BabyView vs THINGs category embeddings\n(Line connects BV → THINGs for each category)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel(f"{method_label} 1", fontsize=12)
    ax.set_ylabel(f"{method_label} 2", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = out_dir / f"{prefix}_bv_things_lines.{ext}"
        plt.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("Saved:", p)
    plt.close()


def _plot_bv_things_lines_labeled(
    words: list,
    cdi_cats: list,
    bv_x: np.ndarray,
    bv_y: np.ndarray,
    things_x: np.ndarray,
    things_y: np.ndarray,
    prefix: str,
    method_label: str,
    out_dir: Path,
):
    """Same as lines plot but with category labels at line midpoints. Save PNG/PDF."""
    n = len(words)
    unique_cdi = sorted(set(cdi_cats)) if cdi_cats else []
    if unique_cdi and unique_cdi != ["unknown"]:
        palette = _harmonious_cdi_palette(len(unique_cdi))
        cdi_to_color = {c: palette[i] for i, c in enumerate(unique_cdi)}
        line_colors = [_blend_with_white(cdi_to_color.get(c, "#888888"), 0.5) for c in cdi_cats]
        dot_colors = [cdi_to_color.get(c, "#888888") for c in cdi_cats]
    else:
        cdi_to_color = {}
        line_colors = ["#6b6b6b"] * n
        dot_colors = None

    fig, ax = plt.subplots(figsize=(18, 14))
    for i in range(n):
        ax.annotate(
            "",
            xy=(things_x[i], things_y[i]),
            xytext=(bv_x[i], bv_y[i]),
            arrowprops=dict(arrowstyle="->", color=line_colors[i], lw=1.0, alpha=0.6),
        )
        mid_x = (bv_x[i] + things_x[i]) / 2
        mid_y = (bv_y[i] + things_y[i]) / 2
        ax.annotate(
            words[i],
            (mid_x, mid_y),
            fontsize=6, ha="center", va="center", alpha=0.9,
        )
    if dot_colors is not None:
        ax.scatter(bv_x, bv_y, c=dot_colors, s=60, marker="o", label="BV", edgecolors="none", alpha=0.9, zorder=3)
        ax.scatter(things_x, things_y, c=dot_colors, s=60, marker="s", label="THINGs", edgecolors="none", alpha=0.9, zorder=3)
    else:
        ax.scatter(bv_x, bv_y, c="#c45c3e", s=60, marker="o", label="BV", edgecolors="none", alpha=0.9, zorder=3)
        ax.scatter(things_x, things_y, c="#3d6cb9", s=60, marker="s", label="THINGs", edgecolors="none", alpha=0.9, zorder=3)
    if unique_cdi and unique_cdi != ["unknown"]:
        legend_elements = [
            Line2D([0], [0], color="#555555", marker="o", linestyle="None", markersize=8, markeredgecolor="none", label="BabyView (BV)"),
            Line2D([0], [0], color="#555555", marker="s", linestyle="None", markersize=8, markeredgecolor="none", label="THINGs"),
        ]
        for cat in unique_cdi:
            legend_elements.append(Line2D([0], [0], color=cdi_to_color[cat], lw=3, label=cat))
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title(f"{method_label}: BV vs THINGs (labels at line midpoints)", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method_label} 1", fontsize=12)
    ax.set_ylabel(f"{method_label} 2", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = out_dir / f"{prefix}_bv_things_lines_labeled.{ext}"
        plt.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("Saved:", p)
    plt.close()


def _plot_bv_things_lines_dots_labeled(
    words: list,
    cdi_cats: list,
    bv_x: np.ndarray,
    bv_y: np.ndarray,
    things_x: np.ndarray,
    things_y: np.ndarray,
    prefix: str,
    method_label: str,
    out_dir: Path,
    max_labels: int = 14,
):
    """Draw arrows and dots; add selected category labels offset from the dots (not on the lines)."""
    n = len(words)
    unique_cdi = sorted(set(cdi_cats)) if cdi_cats else []
    if unique_cdi and unique_cdi != ["unknown"]:
        palette = _harmonious_cdi_palette(len(unique_cdi))
        cdi_to_color = {c: palette[i] for i, c in enumerate(unique_cdi)}
        line_colors = [_blend_with_white(cdi_to_color.get(c, "#888888"), 0.5) for c in cdi_cats]
        dot_colors = [cdi_to_color.get(c, "#888888") for c in cdi_cats]
    else:
        cdi_to_color = {}
        line_colors = ["#6b6b6b"] * n
        dot_colors = None

    label_indices = _indices_to_label(n, cdi_cats, max_labels=max_labels)

    fig, ax = plt.subplots(figsize=(16, 14))
    for i in range(n):
        ax.annotate(
            "",
            xy=(things_x[i], things_y[i]),
            xytext=(bv_x[i], bv_y[i]),
            arrowprops=dict(arrowstyle="->", color=line_colors[i], lw=1.0, alpha=0.6),
        )
    if dot_colors is not None:
        ax.scatter(bv_x, bv_y, c=dot_colors, s=90, marker="o", label="BabyView (BV)", edgecolors="none", alpha=0.9, zorder=3)
        ax.scatter(things_x, things_y, c=dot_colors, s=90, marker="s", label="THINGs", edgecolors="none", alpha=0.9, zorder=3)
    else:
        ax.scatter(bv_x, bv_y, c="#c45c3e", s=90, marker="o", label="BabyView (BV)", edgecolors="none", alpha=0.9, zorder=3)
        ax.scatter(things_x, things_y, c="#3d6cb9", s=90, marker="s", label="THINGs", edgecolors="none", alpha=0.9, zorder=3)

    # Labels offset from dots (BV dot for selected categories), not on the lines
    for i in label_indices:
        ax.annotate(
            words[i],
            xy=(bv_x[i], bv_y[i]),
            xytext=(12, 6),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="bottom",
            alpha=0.95,
            zorder=5,
        )
        ax.annotate(
            words[i],
            xy=(things_x[i], things_y[i]),
            xytext=(12, 6),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="bottom",
            alpha=0.95,
            zorder=5,
        )

    if unique_cdi and unique_cdi != ["unknown"]:
        legend_elements = [
            Line2D([0], [0], color="#555555", marker="o", linestyle="None", markersize=10, markeredgecolor="none", label="BabyView (BV)"),
            Line2D([0], [0], color="#555555", marker="s", linestyle="None", markersize=10, markeredgecolor="none", label="THINGs"),
        ]
        for cat in unique_cdi:
            legend_elements.append(Line2D([0], [0], color=cdi_to_color[cat], lw=3, label=cat))
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title(
        f"{method_label}: BabyView vs THINGs (selected labels at dots)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel(f"{method_label} 1", fontsize=12)
    ax.set_ylabel(f"{method_label} 2", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        p = out_dir / f"{prefix}_bv_things_lines_dots_labeled.{ext}"
        plt.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("Saved:", p)
    plt.close()


def _plot_bv_things_lines_with_exemplars(
    words: list,
    cdi_cats: list,
    bv_x: np.ndarray,
    bv_y: np.ndarray,
    things_x: np.ndarray,
    things_y: np.ndarray,
    prefix: str,
    method_label: str,
    out_dir: Path,
    cropped_dir: Path,
    things_images_dir: Path,
    max_exemplar_cats: int = 20,
    exemplar_zoom: float = 0.2,
    exemplar_cell_size: int = 64,
):
    """One exemplar per CDI category (BV + THINGs, same object category). Crops aligned along
    left/right edges, same size, with connector lines to corresponding dots."""
    n = len(words)
    unique_cdi = sorted(set(cdi_cats)) if cdi_cats else []
    if unique_cdi and unique_cdi != ["unknown"]:
        palette = _harmonious_cdi_palette(len(unique_cdi))
        cdi_to_color = {c: palette[i] for i, c in enumerate(unique_cdi)}
        line_colors = [_blend_with_white(cdi_to_color.get(c, "#888888"), 0.5) for c in cdi_cats]
        dot_colors = [cdi_to_color.get(c, "#888888") for c in cdi_cats]
    else:
        line_colors = ["#6b6b6b"] * n
        dot_colors = None

    # One index per CDI category (BV and THINGs from same object category = same word)
    exemplar_indices = _indices_to_label(n, cdi_cats, max_labels=max(20, len(unique_cdi) if unique_cdi else 20))
    K = len(exemplar_indices)
    if K == 0:
        plt.close()
        return

    # Keep figsize small to avoid MemoryError (renderer buffer = width*height*dpi in pixels)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n):
        ax.annotate(
            "",
            xy=(things_x[i], things_y[i]),
            xytext=(bv_x[i], bv_y[i]),
            arrowprops=dict(arrowstyle="->", color=line_colors[i], lw=1.0, alpha=0.6),
        )
    if dot_colors is not None:
        ax.scatter(bv_x, bv_y, c=dot_colors, s=70, marker="o", label="BabyView (BV)", edgecolors="none", alpha=0.9, zorder=3)
        ax.scatter(things_x, things_y, c=dot_colors, s=70, marker="s", label="THINGs", edgecolors="none", alpha=0.9, zorder=3)
    else:
        ax.scatter(bv_x, bv_y, c="#c45c3e", s=70, marker="o", label="BabyView (BV)", edgecolors="none", alpha=0.9, zorder=3)
        ax.scatter(things_x, things_y, c="#3d6cb9", s=70, marker="s", label="THINGs", edgecolors="none", alpha=0.9, zorder=3)

    if unique_cdi and unique_cdi != ["unknown"]:
        legend_elements = [
            Line2D([0], [0], color="#555555", marker="o", linestyle="None", markersize=8, markeredgecolor="none", label="BabyView (BV)"),
            Line2D([0], [0], color="#555555", marker="s", linestyle="None", markersize=8, markeredgecolor="none", label="THINGs"),
        ]
        for cat in unique_cdi:
            legend_elements.append(Line2D([0], [0], color=cdi_to_color[cat], lw=3, label=cat))
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title(
        f"{method_label}: BabyView vs THINGs (with exemplar thumbnails)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel(f"{method_label} 1", fontsize=12)
    ax.set_ylabel(f"{method_label} 2", fontsize=12)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.subplots_adjust(left=0.16, right=0.84)

    fig = ax.get_figure()
    data_to_fig = fig.transFigure.inverted() + ax.transData
    left_strip_x = 0.06
    right_strip_x = 0.94
    # Align exemplars along each edge: evenly spaced y in figure fraction
    y_positions = np.linspace(0.08, 0.92, K)

    def _load_and_resize(path: Path | None) -> np.ndarray | None:
        if path is None or not path.exists():
            return None
        try:
            arr = mimg.imread(path)
            return _resize_to_fixed(arr, exemplar_cell_size)
        except Exception:
            return None

    # Connector lines from each crop position to its dot (figure coords)
    for k in range(K):
        i = exemplar_indices[k]
        fy = y_positions[k]
        bv_fig = data_to_fig.transform((bv_x[i], bv_y[i]))
        th_fig = data_to_fig.transform((things_x[i], things_y[i]))
        line_bv = Line2D(
            [left_strip_x, bv_fig[0]], [fy, bv_fig[1]],
            transform=fig.transFigure, color="gray", lw=0.8, alpha=0.7, zorder=1,
        )
        line_th = Line2D(
            [right_strip_x, th_fig[0]], [fy, th_fig[1]],
            transform=fig.transFigure, color="gray", lw=0.8, alpha=0.7, zorder=1,
        )
        fig.add_artist(line_bv)
        fig.add_artist(line_th)

    # Place crops along edges (same zoom + same pixel size = same display size)
    for k in range(K):
        i = exemplar_indices[k]
        cat = words[i]
        fy = y_positions[k]
        bv_path = _get_one_bv_crop_path(cropped_dir, cat)
        th_path = _get_things_image_by_index(things_images_dir, cat, 0)
        for path, fx in [(bv_path, left_strip_x), (th_path, right_strip_x)]:
            arr = _load_and_resize(path)
            if arr is None:
                continue
            imbox = OffsetImage(arr, zoom=exemplar_zoom)
            ab = AnnotationBbox(
                imbox, (fx, fy),
                xycoords="figure fraction",
                frameon=True,
                pad=0.15,
                bboxprops=dict(edgecolor="gray", linewidth=0.8),
            )
            fig.add_artist(ab)

    # Save PDF first (vector, low memory); then PNG with low DPI and fallback on MemoryError
    for ext in ["pdf", "png"]:
        p = out_dir / f"{prefix}_bv_things_lines_exemplars.{ext}"
        dpi = 72 if ext == "png" else None
        try:
            plt.savefig(p, dpi=dpi, bbox_inches="tight")
            print("Saved:", p)
        except MemoryError:
            if ext == "png":
                print("Skipped", p, "(MemoryError; use the PDF or run on a machine with more RAM)")
            else:
                raise
    plt.close()


def run_embedding_pair(
    emb_name: str,
    bv_file: str,
    things_file: str,
    bv_word_col: str = "Unnamed: 0",
    things_word_col: str = "text",
    out_base: Path = None,
    random_state: int = 42,
    methods: list = None,
    cropped_dir: Path | None = None,
    things_images_dir: Path | None = None,
):
    """Load one BV/THINGs pair, run UMAP and/or t-SNE, save to bv_things_results_{emb_name}/."""
    out_base = out_base or OUT_BASE
    methods = methods or [("umap", "umap"), ("tsne", "tsne")]
    bv_path = EMBED_DIR / bv_file
    things_path = EMBED_DIR / things_file
    if not bv_path.exists() or not things_path.exists():
        print("Skip", emb_name, "(missing files:", bv_path, things_path, ")")
        return
    words, cdi_cats, bv_mat, th_mat = load_embeddings(bv_path, things_path, bv_word_col, things_word_col)
    print(f"{emb_name}: {len(words)} shared words")
    out_dir = out_base / f"bv_things_results_{emb_name}"
    for method, prefix in methods:
        fit_and_save_coordinates(
            words, cdi_cats, bv_mat, th_mat, out_dir, method, prefix,
            random_state=random_state,
            cropped_dir=cropped_dir,
            things_images_dir=things_images_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute UMAP and t-SNE coordinates and line/arrow plots for BV vs THINGs from CSV embeddings."
    )
    parser.add_argument(
        "--embedding_type",
        choices=["clip", "dinov3", "both"],
        default="both",
        help="Which embedding pair to run: clip, dinov3, or both",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=f"Base output directory (default: {OUT_BASE})",
    )
    parser.add_argument(
        "--method",
        choices=["both", "umap", "tsne"],
        default="both",
        help="2D reduction: both (default), umap, or tsne only",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for UMAP/t-SNE",
    )
    parser.add_argument(
        "--cropped-dir",
        type=Path,
        default=None,
        help="Path to BV cropped-by-class images (category subdirs). If set with --things-images-dir, exemplar plots are generated.",
    )
    parser.add_argument(
        "--things-images-dir",
        type=Path,
        default=None,
        help="Path to THINGs images (category subdirs). If set with --cropped-dir, exemplar plots are generated.",
    )
    args = parser.parse_args()
    out_base = args.output_dir if args.output_dir is not None else OUT_BASE
    cropped_dir = getattr(args, "cropped_dir", None)
    things_images_dir = getattr(args, "things_images_dir", None)
    if args.method == "both":
        methods = [("umap", "umap"), ("tsne", "tsne")]
    elif args.method == "umap":
        methods = [("umap", "umap")]
    else:
        methods = [("tsne", "tsne")]

    if args.embedding_type in ("clip", "both"):
        run_embedding_pair(
            "clip",
            "babyview_clip_filtered26_category_average_embeddings.csv",
            "things_clip_embeddings.csv",
            bv_word_col="Unnamed: 0",
            things_word_col="text",
            out_base=out_base,
            random_state=args.random_seed,
            methods=methods,
            cropped_dir=cropped_dir,
            things_images_dir=things_images_dir,
        )
    if args.embedding_type in ("dinov3", "both"):
        run_embedding_pair(
            "dinov3",
            "babyview_dinov3_filtered26_category_average_embeddings.csv",
            "THINGS_dino_embeddings.csv",
            bv_word_col="Unnamed: 0",
            things_word_col="label",
            out_base=out_base,
            random_state=args.random_seed,
            methods=methods,
            cropped_dir=cropped_dir,
            things_images_dir=things_images_dir,
        )
    print("Done. Outputs in", out_base)


if __name__ == "__main__":
    main()
