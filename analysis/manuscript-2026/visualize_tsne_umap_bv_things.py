#!/usr/bin/env python3
"""
Produce 2D coordinate CSVs and line/arrow plots for BabyView vs THINGs embeddings (UMAP and t-SNE).
Uses CSV embedding files from data/embeddings/. Outputs are written to
bv_things_results_{clip,dinov3}/ with:
  - {umap,tsne}_bv_things_coordinates.csv
  - {umap,tsne}_bv_things_lines.png/.pdf (arrow BV → THINGs, colored by CDI)
  - {umap,tsne}_bv_things_lines_labeled.png/.pdf (labels at line midpoints)

Each CSV has columns: category (word), cdi_category, bv_*_x, bv_*_y, things_*_x, things_*_y.

Run from manuscript-2026/ or set DATA_DIR and OUT_BASE as needed.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

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


def run_embedding_pair(
    emb_name: str,
    bv_file: str,
    things_file: str,
    bv_word_col: str = "Unnamed: 0",
    things_word_col: str = "text",
    out_base: Path = None,
    random_state: int = 42,
    methods: list = None,
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
        fit_and_save_coordinates(words, cdi_cats, bv_mat, th_mat, out_dir, method, prefix, random_state=random_state)


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
    args = parser.parse_args()
    out_base = args.output_dir if args.output_dir is not None else OUT_BASE
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
        )
    print("Done. Outputs in", out_base)


if __name__ == "__main__":
    main()
