#!/usr/bin/env python3
"""Side-by-side BV vs VEDB example-crop montages for (1) the top/bottom
cosine-similarity categories and (2) every CDI "animals" category, matching
analysis/manuscript-2026/02_category-wise_cosine_sim.ipynb's
category_wise_cosine_montage_topbottom_clip_dinov3 figure -- but crops are
pulled fresh from each dataset's crops_index.csv.gz + frames (bbox crop at
read time) rather than from a pre-saved cropped_by_class directory, since no
such directory exists for either the current BV ("2025.2") pipeline or VEDB
(run_cdi_crop_pipeline.py embeds crops in memory; it never writes individual
crop JPEGs to disk).

Usage:
    python plot_bv_vedb_category_montage.py [valid129|valid85]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-vedb")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

VEDB_DIR = Path(__file__).resolve().parent
REPO = VEDB_DIR.parents[1]
ANALYSES = VEDB_DIR / "analyses"

sys.path.insert(0, str(REPO / "analysis" / "manuscript-2026"))
from manuscript_config import load_cdi_semantic_map  # noqa: E402

BV_FRAMES = Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps")
BV_CROPS_INDEX = Path("/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_full/crops_index.csv.gz")
VEDB_FRAMES = Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps")
VEDB_CROPS_INDEX = Path(
    "/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/image_embeddings/vedb_full/crops_index.csv.gz"
)

N_TOP = 5
N_BOTTOM = 5
N_CROPS_PER_SIDE = 10
CELL_SIZE = (80, 80)
N_COLS = 5

# Same exclusions as the manuscript notebook: glasses + body parts (too easy/degenerate for a montage).
MONTAGE_EXCLUDE_CATEGORIES = frozenset(
    {
        "glasses", "ankle", "arm", "ear", "eye", "face", "finger",
        "foot", "hair", "hand", "mouth", "nose", "toe", "tooth",
    }
)

USECOLS = ["video_id", "frame_number", "class_name", "xmin", "ymin", "xmax", "ymax", "clip_sim", "passed"]
DTYPES = {
    "video_id": "string", "frame_number": "int32", "class_name": "string",
    "xmin": "float32", "ymin": "float32", "xmax": "float32", "ymax": "float32",
    "clip_sim": "float32", "passed": "bool",
}


def top_crops_for_categories(crops_index: Path, categories: set[str], n_per_cat: int) -> dict[str, pd.DataFrame]:
    """Stream the (large) crops_index.csv.gz in chunks, and for each category return
    up to n_per_cat rows, EACH FROM A DIFFERENT video_id (session) -- the single
    best (highest clip_sim) passed crop per session, then the top n_per_cat
    sessions by that score. Avoids montages that repeatedly sample one session."""
    best_per_video: dict[tuple[str, str], pd.Series] = {}
    for chunk in pd.read_csv(crops_index, usecols=USECOLS, dtype=DTYPES, chunksize=1_000_000):
        chunk = chunk[chunk["passed"] & chunk["class_name"].isin(categories)]
        if chunk.empty:
            continue
        idx = chunk.groupby(["class_name", "video_id"], observed=True)["clip_sim"].idxmax()
        for _, row in chunk.loc[idx].iterrows():
            key = (row["class_name"], row["video_id"])
            prev = best_per_video.get(key)
            if prev is None or row["clip_sim"] > prev["clip_sim"]:
                best_per_video[key] = row

    by_cat: dict[str, list] = {c: [] for c in categories}
    for (cat, _video_id), row in best_per_video.items():
        by_cat[cat].append(row)

    best: dict[str, pd.DataFrame] = {}
    for cat, rows in by_cat.items():
        if not rows:
            best[cat] = pd.DataFrame(columns=USECOLS)
            continue
        best[cat] = pd.DataFrame(rows).nlargest(n_per_cat, "clip_sim")
    return best


def crop_thumbnail(frames_root: Path, row) -> Image.Image | None:
    # Frame filename zero-padding differs by dataset: BV extracted at 5 digits,
    # VEDB at 8 digits (both via ffmpeg -frame_pts with a sequential counter
    # here, not a PTS value -- frame_number lines up with the plain index).
    video_dir = frames_root / row.video_id
    path = None
    for width in (5, 8):
        candidate = video_dir / f"{int(row.frame_number):0{width}d}.jpg"
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        return None
    try:
        img = Image.open(path).convert("RGB")
        box = (row.xmin, row.ymin, row.xmax, row.ymax)
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            return None
        return img.crop(box)
    except Exception:
        return None


def make_montage(frames_root: Path, rows: pd.DataFrame) -> Image.Image | None:
    thumbs = []
    for row in rows.itertuples(index=False):
        t = crop_thumbnail(frames_root, row)
        if t is not None:
            thumbs.append(t.resize(CELL_SIZE, Image.Resampling.LANCZOS))
    if not thumbs:
        return None
    n_rows = int(np.ceil(len(thumbs) / N_COLS))
    out = Image.new("RGB", (N_COLS * CELL_SIZE[0], n_rows * CELL_SIZE[1]), (240, 240, 240))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, N_COLS)
        out.paste(t, (c * CELL_SIZE[0], r * CELL_SIZE[1]))
    return out


def plot_paired_montage(
    plot_categories: list[str],
    clip_df: pd.DataFrame,
    dino_df: pd.DataFrame,
    bv_best: dict[str, pd.DataFrame],
    vedb_best: dict[str, pd.DataFrame],
    title: str,
    stem: Path,
    hist_title: str | None = None,
) -> None:
    """One row per category: BV crops (left) vs VEDB crops (right). With hist_title,
    adds a histogram of all category-wise cosines underneath."""
    with_histogram = hist_title is not None
    dino_lookup = dict(zip(dino_df["category"], dino_df["cosine_similarity"]))
    n_rows = len(plot_categories)
    extra = [0.45] if with_histogram else []
    fig = plt.figure(figsize=(10, 2.2 * n_rows + (1.0 if with_histogram else 0.4)))
    gs = GridSpec(
        n_rows + len(extra), 2, figure=fig, height_ratios=[1.0] * n_rows + extra,
        hspace=0.28, wspace=0.05, top=0.93, bottom=0.06 if with_histogram else 0.02,
    )

    for i, cat in enumerate(plot_categories):
        clip_sim = float(clip_df.loc[clip_df["category"] == cat, "cosine_similarity"].iloc[0])
        dino_sim = float(dino_lookup.get(cat, float("nan")))

        ax_bv = fig.add_subplot(gs[i, 0])
        ax_vedb = fig.add_subplot(gs[i, 1])

        mont_bv = make_montage(BV_FRAMES, bv_best.get(cat, pd.DataFrame()))
        if mont_bv is not None:
            ax_bv.imshow(np.array(mont_bv))
        else:
            print(f"Warning: no BV crops for {cat!r}", flush=True)
        ax_bv.axis("off")
        ax_bv.set_title(f"{cat}, cos_CLIP={clip_sim:.2f}, cos_DINOv3={dino_sim:.2f}", fontsize=9)

        mont_vedb = make_montage(VEDB_FRAMES, vedb_best.get(cat, pd.DataFrame()))
        if mont_vedb is not None:
            ax_vedb.imshow(np.array(mont_vedb))
        else:
            print(f"Warning: no VEDB crops for {cat!r}", flush=True)
        ax_vedb.axis("off")

    fig.text(0.25, 0.94, "BV", fontsize=10, ha="center")
    fig.text(0.75, 0.94, "VEDB", fontsize=10, ha="center")

    if with_histogram:
        ax_hist = fig.add_subplot(gs[n_rows, :])
        clip_vals = clip_df["cosine_similarity"].to_numpy(dtype=float)
        dino_vals = dino_df["cosine_similarity"].to_numpy(dtype=float)
        all_vals = np.concatenate([clip_vals, dino_vals])
        bins = np.linspace(all_vals.min(), all_vals.max(), 21)
        ax_hist.hist(clip_vals, bins=bins, color="C0", alpha=0.6, edgecolor="white", label=f"CLIP (n={len(clip_vals)})")
        ax_hist.hist(dino_vals, bins=bins, color="C1", alpha=0.6, edgecolor="white", label=f"DINOv3 (n={len(dino_vals)})")
        ax_hist.legend(loc="upper right", fontsize=8)
        ax_hist.set_xlabel("Cosine similarity")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(hist_title)

    fig.suptitle(title, fontsize=12, y=0.965)
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}.png", flush=True)


def main(category_set: str) -> None:
    results = ANALYSES / category_set / "results"
    figures = ANALYSES / category_set / "figures"

    clip_df = pd.read_csv(results / f"bv_vedb_category_wise_cosine_similarity_clip_filtered-0.27_{category_set}.csv")
    dino_df = pd.read_csv(results / f"bv_vedb_category_wise_cosine_similarity_dinov3_filtered-0.27_{category_set}.csv")

    ranked = (
        clip_df[~clip_df["category"].isin(MONTAGE_EXCLUDE_CATEGORIES)]
        .sort_values("cosine_similarity", ascending=False)
        .reset_index(drop=True)
    )
    top_cats = ranked.head(N_TOP)["category"].tolist()
    bottom_cats = ranked.tail(N_BOTTOM)["category"].tolist()
    print(f"[{category_set}] montage categories: top={top_cats} bottom={bottom_cats}", flush=True)

    # Every CDI "animals" category in this set, sorted by CLIP cosine (high -> low).
    semantic = load_cdi_semantic_map()
    by_clip = clip_df.sort_values("cosine_similarity", ascending=False)
    animal_cats = [c for c in by_clip["category"] if semantic.get(c) == "animals"]
    print(f"[{category_set}] animal categories: {animal_cats}", flush=True)

    # Scan each (large) crops index once for both figures.
    wanted = set(top_cats + bottom_cats + animal_cats)
    print("scanning VEDB crops_index ...", flush=True)
    vedb_best = top_crops_for_categories(VEDB_CROPS_INDEX, wanted, N_CROPS_PER_SIDE)
    print("scanning BV crops_index ...", flush=True)
    bv_best = top_crops_for_categories(BV_CROPS_INDEX, wanted, N_CROPS_PER_SIDE)

    plot_paired_montage(
        top_cats + bottom_cats, clip_df, dino_df, bv_best, vedb_best,
        title=f"BV-VEDB cosine similarity (top/bottom by {category_set}, CLIP + DINOv3)",
        stem=figures / f"bv_vedb_category_wise_cosine_montage_topbottom_clip_dinov3_filtered-0.27_{category_set}",
        hist_title=f"BV-VEDB cosine similarity distribution ({category_set})",
    )
    plot_paired_montage(
        animal_cats, clip_df, dino_df, bv_best, vedb_best,
        title=f"BV-VEDB cosine similarity (animal categories, {category_set}, sorted by CLIP)",
        stem=figures / f"bv_vedb_category_wise_cosine_montage_animals_clip_dinov3_filtered-0.27_{category_set}",
    )


if __name__ == "__main__":
    category_set = sys.argv[1] if len(sys.argv) > 1 else "valid129"
    main(category_set)
