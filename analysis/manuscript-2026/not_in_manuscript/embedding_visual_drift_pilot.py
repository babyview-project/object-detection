#!/usr/bin/env python3
"""Pilot: embedding-space visual-input drift across age (crop-level).

Independent of clutter / n_objects analyses. Uses filtered detection metadata to
index crop embeddings, then per child × age window (default: integer month):

  - **Global pooled**: centroid + dispersion over all categories in the window
  - **Per category**: centroid for pilot categories (cup, book, toy, hand, face, …)

Trajectory metrics between consecutive age windows with data:
  - centroid_displacement = 1 - cosine(centroid_t, centroid_{t+1})
  - dispersion_delta = dispersion_{t+1} - dispersion_t

Outputs under analysis/manuscript-2026/not_in_manuscript/embedding_drift_exploration/

Examples:
  python analysis/manuscript-2026/not_in_manuscript/embedding_visual_drift_pilot.py --top-n 8
  BV_EMBED_MODEL=dinov3 python analysis/manuscript-2026/not_in_manuscript/embedding_visual_drift_pilot.py
  python ... --plots-only
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from _paths import MANUSCRIPT_DIR, NOT_IN_MANUSCRIPT_DIR, PROJECT_ROOT

SCRIPT_DIR = NOT_IN_MANUSCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
FRAME_DATA_CSV = PROJECT_ROOT / "frame_data" / "merged_frame_detections_with_metadata_filtered-0.27.csv"
TRAJECTORY_CSV = (
    PROJECT_ROOT
    / "analysis"
    / "individual_analyses"
    / "developmental_trajectory_rdms_clip"
    / "trajectory_correlations.csv"
)

YOLOE_ROOT = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings")
EMBED_DIRS = {
    "clip": YOLOE_ROOT / "clip_embeddings_new",
    "dinov3": YOLOE_ROOT / "facebook_dinov3-vitb16-pretrain-lvd1689m",
}

CATEGORY_SET_FILES = {
    "valid129": DATA_DIR / "included_categories_valid129.txt",
    "valid85": DATA_DIR / "included_categories_valid85.txt",
}

PILOT_CATEGORIES = ("cup", "book", "toy", "hand", "face", "bottle", "chair", "ball")
ALWAYS_EXCLUDE = {"person", "picture"}
EXCLUDED_SUBJECT = "00270001"

OUTPUT_ROOT = SCRIPT_DIR / "embedding_drift_exploration"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"

NPY_MMAP_MODE = os.environ.get("EMBED_DRIFT_NPY_MMAP", "r").strip() or "r"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop embedding drift pilot (per child × age).")
    p.add_argument("--embed-model", choices=("clip", "dinov3"), default=os.environ.get("BV_EMBED_MODEL", "dinov3"))
    p.add_argument("--category-set", choices=tuple(CATEGORY_SET_FILES), default="valid129")
    p.add_argument("--top-n", type=int, default=0, help="Restrict to N densest subjects (0 = all).")
    p.add_argument("--age-bin-months", type=int, default=1, help="Age window width in months (1 = exact month).")
    p.add_argument("--max-crops-per-window", type=int, default=256, help="Reservoir cap per (subject, age, [cat]).")
    p.add_argument("--min-crops-global", type=int, default=30, help="Min crops for global window stats.")
    p.add_argument("--min-crops-category", type=int, default=15, help="Min crops for per-category window stats.")
    p.add_argument("--pilot-categories", nargs="*", default=None, help="Override default pilot category list.")
    p.add_argument(
        "--top-categories-by-count",
        type=int,
        default=0,
        help="Use the N most frequent categories (top-N subjects) instead of PILOT_CATEGORIES.",
    )
    p.add_argument("--plots-only", action="store_true", help="Regenerate figures from cached CSVs.")
    return p.parse_args()


def normalize_subject_id(sid: str) -> str:
    s = str(sid).strip().lstrip("S")
    return s.zfill(8) if s.isdigit() else s


def load_allowed_categories(path: Path) -> set[str]:
    cats = {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return cats - ALWAYS_EXCLUDE


def age_to_bin(age_mo: float, bin_months: int) -> int:
    a = int(round(age_mo))
    if bin_months <= 1:
        return a
    return (a // bin_months) * bin_months


def l2_normalize(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(l2_normalize(a), l2_normalize(b)), -1.0, 1.0))


def window_key(subject_id: str, age_bin: int, category: str | None) -> tuple:
    return (subject_id, age_bin, category)


def cache_tag(args: argparse.Namespace) -> str:
    top = f"_top{args.top_n}" if args.top_n > 0 else ""
    if args.top_categories_by_count > 0:
        cat = f"_cat{args.top_categories_by_count}"
    elif args.pilot_categories and set(args.pilot_categories) != set(PILOT_CATEGORIES):
        cat = f"_pilot{len(args.pilot_categories)}"
    else:
        cat = ""
    return (
        f"{args.embed_model}_{args.category_set}_bin{args.age_bin_months}"
        f"_max{args.max_crops_per_window}{top}{cat}"
    )


def top_categories_by_detection_count(
    metadata_csv: Path,
    allowed_categories: set[str],
    subject_filter: set[str] | None,
    n_top: int,
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    usecols = ["class_name", "subject_id"]
    for chunk in pd.read_csv(
        metadata_csv,
        usecols=usecols,
        chunksize=400_000,
        dtype={"subject_id": str, "class_name": str},
    ):
        chunk = chunk.dropna(subset=usecols)
        chunk["category"] = chunk["class_name"].astype(str).str.strip().str.lower()
        chunk = chunk[chunk["category"].isin(allowed_categories)]
        chunk["subject_id"] = chunk["subject_id"].map(normalize_subject_id)
        chunk = chunk[chunk["subject_id"] != EXCLUDED_SUBJECT]
        if subject_filter is not None:
            chunk = chunk[chunk["subject_id"].isin(subject_filter)]
        for cat, n in chunk.groupby("category").size().items():
            counts[cat] += int(n)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [c for c, _ in ranked[:n_top]]


def get_top_subjects(n_top: int) -> list[str]:
    traj = pd.read_csv(TRAJECTORY_CSV)
    traj["subject_id"] = traj["subject_id"].astype(str).map(normalize_subject_id)
    ranked = traj.sort_values("rdm_correlation", ascending=False)
    return ranked.head(n_top)["subject_id"].tolist()


class StemReservoir:
    """Reservoir sample of embedding stems (strings) per window."""

    __slots__ = ("max_n", "items", "n_seen", "_rng")

    def __init__(self, max_n: int, seed: int):
        self.max_n = max_n
        self.items: list[str] = []
        self.n_seen = 0
        self._rng = np.random.default_rng(seed)

    def add(self, stem: str) -> None:
        self.n_seen += 1
        if self.n_seen <= self.max_n:
            self.items.append(stem)
            return
        j = int(self._rng.integers(0, self.n_seen))
        if j < self.max_n:
            self.items[j] = stem


@dataclass
class WindowAccumulator:
    stems: StemReservoir


def reservoir_index_pass(
    metadata_csv: Path,
    allowed_categories: set[str],
    pilot_categories: set[str],
    subject_filter: set[str] | None,
    age_bin_months: int,
    max_crops: int,
    seed: int,
) -> tuple[dict[tuple, WindowAccumulator], dict[tuple, WindowAccumulator], dict[str, tuple[str, int, str]]]:
    global_acc: dict[tuple, WindowAccumulator] = {}
    cat_acc: dict[tuple, WindowAccumulator] = {}
    stem_meta: dict[str, tuple[str, int, str]] = {}

    usecols = ["class_name", "age_mo", "subject_id", "original_embedding_name"]
    for chunk in tqdm(
        pd.read_csv(
            metadata_csv,
            usecols=usecols,
            chunksize=400_000,
            dtype={"subject_id": str, "class_name": str},
        ),
        desc="Reservoir stems per window",
    ):
        chunk = chunk.dropna(subset=usecols)
        chunk["category"] = chunk["class_name"].astype(str).str.strip().str.lower()
        chunk = chunk[chunk["category"].isin(allowed_categories)]
        chunk["subject_id"] = chunk["subject_id"].map(normalize_subject_id)
        chunk = chunk[chunk["subject_id"] != EXCLUDED_SUBJECT]
        if subject_filter is not None:
            chunk = chunk[chunk["subject_id"].isin(subject_filter)]
        chunk["age_bin"] = chunk["age_mo"].map(lambda a: age_to_bin(float(a), age_bin_months))
        chunk["stem"] = (
            chunk["original_embedding_name"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.npy$", "", regex=True)
            .str.lower()
        )

        for row in chunk.itertuples(index=False):
            stem = row.stem
            sid, age_bin, cat = row.subject_id, int(row.age_bin), row.category
            if stem not in stem_meta:
                stem_meta[stem] = (sid, age_bin, cat)

            gk = window_key(sid, age_bin, None)
            if gk not in global_acc:
                global_acc[gk] = WindowAccumulator(StemReservoir(max_crops, seed + hash(gk) % 10_000))
            global_acc[gk].stems.add(stem)

            if cat in pilot_categories:
                ck = window_key(sid, age_bin, cat)
                if ck not in cat_acc:
                    cat_acc[ck] = WindowAccumulator(StemReservoir(max_crops, seed + 17 + hash(ck) % 10_000))
                cat_acc[ck].stems.add(stem)

    return global_acc, cat_acc, stem_meta


def window_stats_from_stems(
    stem_list: list[str],
    stem_to_path: dict[str, Path],
    n_indexed: int,
) -> tuple[np.ndarray | None, dict]:
    vecs = []
    for stem in stem_list:
        p = stem_to_path.get(stem)
        if p is None:
            continue
        try:
            vecs.append(l2_normalize(np.load(p, mmap_mode=NPY_MMAP_MODE)))
        except Exception:
            continue
    if not vecs:
        return None, {}
    X = np.stack(vecs, axis=0)
    centroid = X.mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn < 1e-12:
        return None, {}
    cu = centroid / cn
    dispersion = float(np.mean(1.0 - X @ cu))
    return cu, {
        "n_crops": len(X),
        "n_indexed": n_indexed,
        "dispersion": dispersion,
        "resultant_length": float(cn),
    }


def _consecutive_age_gap(age_bin_months: int) -> int:
    return max(1, int(age_bin_months))


def compute_trajectories(
    global_acc: dict[tuple, WindowAccumulator],
    cat_acc: dict[tuple, WindowAccumulator],
    stem_to_path: dict[str, Path],
    min_crops_global: int,
    min_crops_cat: int,
    age_bin_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], int | None]:
    age_gap = _consecutive_age_gap(age_bin_months)
    global_rows = []
    centroids: dict[str, np.ndarray] = {}
    embed_dim = None

    for gk, acc in tqdm(global_acc.items(), desc="Global windows"):
        sid, age_bin, _ = gk
        cu, stats = window_stats_from_stems(acc.stems.items, stem_to_path, acc.stems.n_seen)
        if cu is None or stats["n_crops"] < min_crops_global:
            continue
        embed_dim = len(cu)
        centroids[f"g|{sid}|{age_bin}"] = cu.astype(np.float32)
        global_rows.append({"subject_id": sid, "age_bin": age_bin, **stats})

    cat_rows = []
    for ck, acc in tqdm(cat_acc.items(), desc="Category windows"):
        sid, age_bin, cat = ck
        cu, stats = window_stats_from_stems(acc.stems.items, stem_to_path, acc.stems.n_seen)
        if cu is None or stats["n_crops"] < min_crops_cat:
            continue
        centroids[f"c|{sid}|{age_bin}|{cat}"] = cu.astype(np.float32)
        cat_rows.append({"subject_id": sid, "age_bin": age_bin, "category": cat, **stats})

    gdf = pd.DataFrame(global_rows)
    cdf = pd.DataFrame(cat_rows)

    edge_g = []
    for sid, g in gdf.groupby("subject_id"):
        g = g.sort_values("age_bin")
        ages = g["age_bin"].to_numpy()
        for i in range(len(g) - 1):
            if ages[i + 1] - ages[i] != age_gap:
                continue
            k0 = f"g|{sid}|{int(g.iloc[i]['age_bin'])}"
            k1 = f"g|{sid}|{int(g.iloc[i + 1]['age_bin'])}"
            if k0 not in centroids or k1 not in centroids:
                continue
            edge_g.append(
                {
                    "subject_id": sid,
                    "age_from": int(g.iloc[i]["age_bin"]),
                    "age_to": int(g.iloc[i + 1]["age_bin"]),
                    "age_midpoint": (ages[i] + ages[i + 1]) / 2.0,
                    "centroid_displacement": cosine_distance(centroids[k0], centroids[k1]),
                    "dispersion_from": g.iloc[i]["dispersion"],
                    "dispersion_to": g.iloc[i + 1]["dispersion"],
                    "dispersion_delta": g.iloc[i + 1]["dispersion"] - g.iloc[i]["dispersion"],
                    "n_crops_from": int(g.iloc[i]["n_crops"]),
                    "n_crops_to": int(g.iloc[i + 1]["n_crops"]),
                    "n_indexed_from": int(g.iloc[i]["n_indexed"]),
                    "n_indexed_to": int(g.iloc[i + 1]["n_indexed"]),
                }
            )

    edge_c = []
    for (sid, cat), g in cdf.groupby(["subject_id", "category"]):
        g = g.sort_values("age_bin")
        ages = g["age_bin"].to_numpy()
        for i in range(len(g) - 1):
            if ages[i + 1] - ages[i] != age_gap:
                continue
            k0 = f"c|{sid}|{int(g.iloc[i]['age_bin'])}|{cat}"
            k1 = f"c|{sid}|{int(g.iloc[i + 1]['age_bin'])}|{cat}"
            if k0 not in centroids or k1 not in centroids:
                continue
            edge_c.append(
                {
                    "subject_id": sid,
                    "category": cat,
                    "age_from": int(g.iloc[i]["age_bin"]),
                    "age_to": int(g.iloc[i + 1]["age_bin"]),
                    "age_midpoint": (ages[i] + ages[i + 1]) / 2.0,
                    "centroid_displacement": cosine_distance(centroids[k0], centroids[k1]),
                    "dispersion_from": g.iloc[i]["dispersion"],
                    "dispersion_to": g.iloc[i + 1]["dispersion"],
                    "dispersion_delta": g.iloc[i + 1]["dispersion"] - g.iloc[i]["dispersion"],
                    "n_crops_from": int(g.iloc[i]["n_crops"]),
                    "n_crops_to": int(g.iloc[i + 1]["n_crops"]),
                }
            )

    return gdf, cdf, pd.DataFrame(edge_g), pd.DataFrame(edge_c), centroids, embed_dim


def plot_global_trajectories(edges: pd.DataFrame, out_dir: Path, tag: str) -> None:
    if edges.empty:
        print("No global edges to plot.")
        return
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for sid in sorted(edges["subject_id"].unique()):
        g = edges[edges["subject_id"] == sid]
        axes[0].plot(g["age_midpoint"], g["centroid_displacement"], marker="o", alpha=0.55, label=sid)
        axes[1].plot(g["age_midpoint"], g["dispersion_delta"], marker="o", alpha=0.55)
    pooled = edges.groupby("age_midpoint", as_index=False).agg(
        centroid_displacement=("centroid_displacement", "mean"),
        dispersion_delta=("dispersion_delta", "mean"),
    )
    axes[0].plot(
        pooled["age_midpoint"],
        pooled["centroid_displacement"],
        color="black",
        lw=3,
        marker="s",
        label="mean across kids",
    )
    axes[1].plot(pooled["age_midpoint"], pooled["dispersion_delta"], color="black", lw=3, marker="s")
    axes[0].set_ylabel("Centroid displacement\n(1 − cos, consecutive windows)")
    axes[1].set_ylabel("Δ dispersion\n(consecutive windows)")
    axes[0].set_title("Global visual-input drift (all categories pooled)")
    axes[1].set_xlabel("Age (months, midpoint)")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"global_drift_trajectories_{tag}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"global_drift_trajectories_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_category_trajectories(edges: pd.DataFrame, categories: list[str], out_dir: Path, tag: str) -> None:
    if edges.empty:
        print("No category edges to plot.")
        return
    cats = [c for c in categories if c in set(edges["category"])]
    if not cats:
        return
    ncols = 3
    nrows = (len(cats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cat in zip(axes, cats):
        sub = edges[edges["category"] == cat]
        for sid in sub["subject_id"].unique():
            g = sub[sub["subject_id"] == sid]
            ax.plot(g["age_midpoint"], g["centroid_displacement"], alpha=0.45, marker="o", ms=4)
        pooled = sub.groupby("age_midpoint")["centroid_displacement"].mean()
        ax.plot(pooled.index, pooled.values, color="black", lw=2.5)
        ax.set_title(cat)
        ax.set_xlabel("Age (mo)")
    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)
    fig.supylabel("Within-category centroid displacement")
    fig.suptitle("Per-category embedding drift (consecutive months)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"category_drift_trajectories_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = cache_tag(args)

    allowed = load_allowed_categories(CATEGORY_SET_FILES[args.category_set])
    subject_filter = None
    if args.top_n > 0:
        subject_filter = set(get_top_subjects(args.top_n))
        print(f"Top-{args.top_n} subjects: {sorted(subject_filter)}")

    if args.top_categories_by_count > 0:
        pilot_list = top_categories_by_detection_count(
            FRAME_DATA_CSV,
            allowed,
            subject_filter,
            args.top_categories_by_count,
        )
        pilot_cats = set(pilot_list) & allowed
        print(f"Top-{args.top_categories_by_count} categories by detection count: {pilot_list}")
    else:
        pilot_cats = set(args.pilot_categories or PILOT_CATEGORIES) & allowed

    if args.plots_only:
        plot_global_trajectories(pd.read_csv(RESULTS_DIR / f"global_trajectory_edges_{tag}.csv"), FIGURES_DIR, tag)
        plot_category_trajectories(
            pd.read_csv(RESULTS_DIR / f"category_trajectory_edges_{tag}.csv"),
            sorted(pilot_cats),
            FIGURES_DIR,
            tag,
        )
        print(f"Figures written to {FIGURES_DIR}")
        return

    embeddings_dir = EMBED_DIRS[args.embed_model]
    if not embeddings_dir.is_dir():
        raise FileNotFoundError(f"Embeddings dir missing: {embeddings_dir}")

    global_acc, cat_acc, stem_meta = reservoir_index_pass(
        FRAME_DATA_CSV,
        load_allowed_categories(CATEGORY_SET_FILES[args.category_set]),
        pilot_cats,
        subject_filter,
        args.age_bin_months,
        args.max_crops_per_window,
        seed=42,
    )
    print(f"Stems indexed: {len(stem_meta):,} | global windows: {len(global_acc)} | category windows: {len(cat_acc)}")

    stems_needed: set[str] = set()
    for acc in list(global_acc.values()) + list(cat_acc.values()):
        stems_needed.update(acc.stems.items)

    stem_to_path: dict[str, Path] = {}
    stems_by_cat: dict[str, set[str]] = defaultdict(set)
    for stem in stems_needed:
        if stem in stem_meta:
            stems_by_cat[stem_meta[stem][2]].add(stem)
    for cat, stems in tqdm(stems_by_cat.items(), desc="Resolve npy paths"):
        cat_dir = embeddings_dir / cat
        if not cat_dir.is_dir():
            continue
        for stem in stems:
            p = cat_dir / f"{stem}.npy"
            if p.is_file():
                stem_to_path[stem] = p
    print(f"NPY paths found: {len(stem_to_path):,} / {len(stems_needed):,}")

    gdf, cdf, edges_g, edges_c, centroids, embed_dim = compute_trajectories(
        global_acc,
        cat_acc,
        stem_to_path,
        args.min_crops_global,
        args.min_crops_category,
        age_bin_months=args.age_bin_months,
    )

    gdf.to_csv(RESULTS_DIR / f"global_windows_{tag}.csv", index=False)
    cdf.to_csv(RESULTS_DIR / f"category_windows_{tag}.csv", index=False)
    edges_g.to_csv(RESULTS_DIR / f"global_trajectory_edges_{tag}.csv", index=False)
    edges_c.to_csv(RESULTS_DIR / f"category_trajectory_edges_{tag}.csv", index=False)
    if centroids:
        np.savez_compressed(RESULTS_DIR / f"centroids_full_{tag}.npz", **centroids)

    meta = {
        "embed_model": args.embed_model,
        "category_set": args.category_set,
        "embed_dim": embed_dim,
        "n_global_windows": len(gdf),
        "n_category_windows": len(cdf),
        "n_global_edges": len(edges_g),
        "n_category_edges": len(edges_c),
        "pilot_categories": sorted(pilot_cats),
        "top_n": args.top_n,
    }
    (RESULTS_DIR / f"run_meta_{tag}.json").write_text(json.dumps(meta, indent=2))

    print(f"\nGlobal windows: {len(gdf)} | consecutive edges: {len(edges_g)}")
    print(f"Category windows: {len(cdf)} | category edges: {len(edges_c)}")
    if len(edges_g):
        print(
            f"  Mean centroid displacement: {edges_g['centroid_displacement'].mean():.4f} "
            f"(sd {edges_g['centroid_displacement'].std():.4f})"
        )

    plot_global_trajectories(edges_g, FIGURES_DIR, tag)
    plot_category_trajectories(edges_c, sorted(pilot_cats), FIGURES_DIR, tag)
    print(f"Done.\n  Results: {RESULTS_DIR}\n  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
