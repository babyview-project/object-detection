#!/usr/bin/env python3
"""Per-category centroid/dispersion with location/activity context + CCN + displacement.

Strata: all, kitchen_like, living_room, outside, eating, playing.

Default categories span household (cup, plate), animals (dog, cat), vehicles (car, stroller).

Examples:
  python analysis/manuscript-2026/per_category_context_analysis.py --top-n 8
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from _paths import NOT_IN_MANUSCRIPT_DIR, PROJECT_ROOT

SCRIPT_DIR = NOT_IN_MANUSCRIPT_DIR
DATA_DIR = PROJECT_ROOT / "data"
FRAME_CSV = PROJECT_ROOT / "frame_data" / "merged_frame_detections_with_metadata_filtered-0.27.csv"
CONTEXT_CSV = DATA_DIR / "all_contexts_public.csv"
CDI_CSV = DATA_DIR / "long_tailed_dist_prop_included_categories_valid129.csv"
CCN_KNN_CSV = (
    PROJECT_ROOT
    / "analysis"
    / "ccn-2026"
    / "plotC_knn_diversity_outputs"
    / "new_things_embeddings_20260428"
    / "bv_dinov3_local_global_k5_valid129.csv"
)
TRAJECTORY_CSV = (
    PROJECT_ROOT
    / "analysis"
    / "individual_analyses"
    / "developmental_trajectory_rdms_clip"
    / "trajectory_correlations.csv"
)
EMBED_DIR = Path("/data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/facebook_dinov3-vitb16-pretrain-lvd1689m")

OUTPUT_ROOT = SCRIPT_DIR / "embedding_drift_exploration" / "context_merged"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"

# Default analysis set: household + animals + vehicles (valid129, well-attested)
DEFAULT_CATEGORIES = (
    "cup",
    "plate",
    "dog",
    "cat",
    "car",
    "stroller",
)
DOMAIN_BY_CATEGORY = {
    "cup": "household",
    "plate": "household",
    "bowl": "household",
    "bottle": "household",
    "dog": "animals",
    "cat": "animals",
    "bird": "animals",
    "car": "vehicles",
    "stroller": "vehicles",
    "motorcycle": "vehicles",
}
# Primary context stratum for domain comparison plots (vs `all`)
PRIMARY_STRATUM_BY_DOMAIN = {
    "household": "kitchen_like",
    "animals": "outside",
    "vehicles": "outside",
}

KITCHEN_LOCATIONS = {"kitchen", "dining room"}
EXCLUDED_SUBJECT = "00270001"
MAX_CROPS = 256
MIN_CROPS = 12
ALL_STRATA = ("all", "kitchen_like", "living_room", "outside", "eating", "playing")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--max-crops", type=int, default=MAX_CROPS)
    p.add_argument("--min-crops", type=int, default=MIN_CROPS)
    p.add_argument("--skip-embed-load", action="store_true")
    return p.parse_args()


def normalize_sid(s: str) -> str:
    s = str(s).strip().lstrip("S")
    return s.zfill(8) if s.isdigit() else s


def get_top_subjects(n: int) -> set[str]:
    traj = pd.read_csv(TRAJECTORY_CSV)
    traj["subject_id"] = traj["subject_id"].astype(str).map(normalize_sid)
    return set(traj.sort_values("rdm_correlation", ascending=False).head(n)["subject_id"])


def load_cdi_domains() -> dict[str, str]:
    df = pd.read_csv(CDI_CSV, usecols=["category", "cdi_semantic"])
    return {str(r.category).strip().lower(): str(r.cdi_semantic).strip().lower() for r in df.itertuples()}


def load_context_lookup() -> pd.DataFrame:
    ctx = pd.read_csv(CONTEXT_CSV, usecols=["video_id", "Location", "Activity"])
    ctx["Location"] = ctx["Location"].astype(str).str.strip().str.lower()
    ctx["Activity"] = ctx["Activity"].astype(str).str.strip().str.lower()
    return ctx.drop_duplicates("video_id")


def frame_to_video_id(superseded: str, frame_path: str) -> str | None:
    import re

    m = re.search(r"/(\d+)\.jpg$", str(frame_path))
    if not m:
        return None
    seg = int(m.group(1)) // 10
    return f"{superseded}_processed_{seg:03d}.mp4"


def strata_for_context(loc: str | None, act: str | None) -> list[str]:
    out = ["all"]
    if loc in KITCHEN_LOCATIONS:
        out.append("kitchen_like")
    if loc == "living room":
        out.append("living_room")
    if loc == "outside":
        out.append("outside")
    if act == "eating":
        out.append("eating")
    if act == "playing":
        out.append("playing")
    return out


def l2_normalize(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(l2_normalize(a), l2_normalize(b)), -1.0, 1.0))


class StemReservoir:
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
class Bucket:
    reservoir: StemReservoir
    n_indexed: int = 0


def bucket_key(sid: str, age_mo: int, category: str, stratum: str) -> tuple:
    return (sid, age_mo, category, stratum)


def finalize_vecs(stems: list[str], stem_to_path: dict[str, Path]) -> dict | None:
    vecs = []
    for stem in stems:
        p = stem_to_path.get(stem)
        if not p:
            continue
        try:
            vecs.append(l2_normalize(np.load(p, mmap_mode="r")))
        except Exception:
            pass
    if len(vecs) < 2:
        return None
    X = np.stack(vecs, axis=0)
    centroid = X.mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn < 1e-12:
        return None
    cu = centroid / cn
    dispersion = float(np.mean(1.0 - X @ cu))
    return {
        "n_crops": len(X),
        "dispersion": dispersion,
        "resultant_length": float(cn),
        "centroid": cu.astype(np.float32),
    }


def index_detections(
    categories: set[str],
    subjects: set[str],
    ctx_map: dict,
    max_crops: int,
) -> tuple[dict, dict[str, tuple[str, int, str]]]:
    buckets: dict[tuple, Bucket] = {}
    stem_meta: dict[str, tuple[str, int, str]] = {}
    usecols = [
        "class_name",
        "age_mo",
        "subject_id",
        "superseded_gcp_name_feb25",
        "original_frame_path",
        "original_embedding_name",
    ]

    for chunk in tqdm(pd.read_csv(FRAME_CSV, usecols=usecols, chunksize=400_000), desc="Index with context"):
        chunk = chunk.copy()
        chunk["category"] = chunk["class_name"].astype(str).str.strip().str.lower()
        chunk = chunk[chunk["category"].isin(categories)]
        chunk["subject_id"] = chunk["subject_id"].map(normalize_sid)
        chunk = chunk[chunk["subject_id"].isin(subjects) & (chunk["subject_id"] != EXCLUDED_SUBJECT)]
        chunk["age_mo"] = chunk["age_mo"].astype(int)
        chunk["stem"] = chunk["original_embedding_name"].astype(str).str.replace(r"\.npy$", "", regex=True).str.lower()
        chunk["video_id"] = [
            frame_to_video_id(s, p) for s, p in zip(chunk["superseded_gcp_name_feb25"], chunk["original_frame_path"])
        ]
        chunk = chunk[chunk["video_id"].notna()]

        for row in chunk.itertuples(index=False):
            stem = row.stem
            if stem not in stem_meta:
                stem_meta[stem] = (row.subject_id, int(row.age_mo), row.category)
            loc = act = None
            if row.video_id in ctx_map:
                loc = ctx_map[row.video_id]["Location"]
                act = ctx_map[row.video_id]["Activity"]
            for st in strata_for_context(loc, act):
                bk = bucket_key(row.subject_id, int(row.age_mo), row.category, st)
                if bk not in buckets:
                    buckets[bk] = Bucket(StemReservoir(max_crops, 42 + hash(bk) % 10_000))
                buckets[bk].reservoir.add(stem)
                buckets[bk].n_indexed += 1

    return buckets, stem_meta


def compute_displacement_edges(geom_df: pd.DataFrame, centroids: dict[str, np.ndarray], min_crops: int) -> pd.DataFrame:
    rows = []
    for (sid, cat, st), g in geom_df.groupby(["subject_id", "category", "stratum"]):
        g = g.sort_values("age_mo")
        ages = g["age_mo"].to_numpy()
        for i in range(len(g) - 1):
            if ages[i + 1] - ages[i] != 1:
                continue
            r0, r1 = g.iloc[i], g.iloc[i + 1]
            if r0["n_crops"] < min_crops or r1["n_crops"] < min_crops:
                continue
            k0 = f"{sid}|{int(r0['age_mo'])}|{cat}|{st}"
            k1 = f"{sid}|{int(r1['age_mo'])}|{cat}|{st}"
            if k0 not in centroids or k1 not in centroids:
                continue
            rows.append(
                {
                    "subject_id": sid,
                    "category": cat,
                    "stratum": st,
                    "cdi_domain": DOMAIN_BY_CATEGORY.get(cat, "other"),
                    "age_from": int(r0["age_mo"]),
                    "age_to": int(r1["age_mo"]),
                    "age_midpoint": (ages[i] + ages[i + 1]) / 2.0,
                    "centroid_displacement": cosine_distance(centroids[k0], centroids[k1]),
                    "dispersion_from": float(r0["dispersion"]),
                    "dispersion_to": float(r1["dispersion"]),
                    "dispersion_delta": float(r1["dispersion"]) - float(r0["dispersion"]),
                    "n_crops_from": int(r0["n_crops"]),
                    "n_crops_to": int(r1["n_crops"]),
                }
            )
    return pd.DataFrame(rows)


def load_ccn_metrics() -> pd.DataFrame:
    df = pd.read_csv(CCN_KNN_CSV)
    df["category"] = df["category"].astype(str).str.lower()
    return df.rename(
        columns={
            "global_dispersion": "ccn_global_dispersion",
            "mean_knn_dist": "ccn_mean_knn_dist",
            "local_coherence": "ccn_local_coherence",
            "local_over_global": "ccn_local_over_global",
        }
    )


def plot_dispersion_context_compare(geom_df: pd.DataFrame, categories: list[str], strata: tuple[str, str], out: Path, title: str) -> None:
    """Side-by-side: mean dispersion by age for two strata, multiple categories."""
    sns.set_theme(style="whitegrid", context="talk")
    n = len(categories)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    colors = {"all": "#0d47a1", "kitchen_like": "#e65100", "living_room": "#6a1b9a", "outside": "#2e7d32", "eating": "#c62828", "playing": "#f9a825"}
    for ax, cat in zip(axes, categories):
        sub = geom_df[geom_df["category"] == cat]
        for st in strata:
            s = sub[sub["stratum"] == st].groupby("age_mo")["dispersion"].mean()
            if len(s) == 0:
                continue
            ax.plot(s.index, s.values, marker="o", ms=5, label=st, color=colors.get(st, "gray"), alpha=0.85)
        ax.set_title(cat)
        ax.set_xlabel("Age (mo)")
    axes[0].set_ylabel("Within-month dispersion")
    fig.suptitle(title, y=1.02)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_displacement_context_compare(edges: pd.DataFrame, categories: list[str], strata: tuple[str, str], out: Path, title: str) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    n = len(categories)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    colors = {"all": "#0d47a1", "kitchen_like": "#e65100", "living_room": "#6a1b9a", "outside": "#2e7d32", "eating": "#c62828", "playing": "#f9a825"}
    for ax, cat in zip(axes, categories):
        sub = edges[edges["category"] == cat]
        for st in strata:
            e = sub[sub["stratum"] == st]
            if e.empty:
                continue
            pooled = e.groupby("age_midpoint")["centroid_displacement"].mean()
            ax.plot(pooled.index, pooled.values, marker="o", ms=5, label=st, color=colors.get(st, "gray"))
        ax.set_title(cat)
        ax.set_xlabel("Age (mo)")
    axes[0].set_ylabel("Centroid displacement (1 − cos)")
    fig.suptitle(title, y=1.02)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_domain_summary(edges: pd.DataFrame, geom_df: pd.DataFrame, out: Path) -> None:
    """3-row panel: household / animals / vehicles — displacement (primary stratum vs all)."""
    domains = ["household", "animals", "vehicles"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for i, dom in enumerate(domains):
        cats = [c for c, d in DOMAIN_BY_CATEGORY.items() if d == dom and c in geom_df["category"].unique()]
        if not cats:
            continue
        prim = PRIMARY_STRATUM_BY_DOMAIN[dom]
        for j, (metric, df, col) in enumerate(
            [("Displacement", edges, "centroid_displacement"), ("Dispersion", geom_df, "dispersion")]
        ):
            ax = axes[i, j]
            for cat in cats:
                for st, ls in [(prim, "-"), ("all", "--")]:
                    sub = df[(df["category"] == cat) & (df["stratum"] == st)]
                    if sub.empty:
                        continue
                    key = "age_midpoint" if "age_midpoint" in sub.columns else "age_mo"
                    pooled = sub.groupby(key)[col].mean()
                    ax.plot(pooled.index, pooled.values, ls=ls, marker="o", ms=4, label=f"{cat} ({st})")
            ax.set_title(f"{dom} — {metric}")
            ax.set_xlabel("Age (mo)")
            if j == 0:
                ax.set_ylabel(col)
    axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.suptitle("Context-stratified geometry (solid=context, dashed=all)", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    categories = set(args.categories or DEFAULT_CATEGORIES)
    subjects = get_top_subjects(args.top_n)
    tag = f"top{args.top_n}"

    print(f"Subjects (top-{args.top_n}): {sorted(subjects)}")
    print(f"Categories: {sorted(categories)}")

    ctx_lut = load_context_lookup()
    ctx_map = ctx_lut.set_index("video_id")[["Location", "Activity"]].to_dict("index")
    buckets, stem_meta = index_detections(categories, subjects, ctx_map, args.max_crops)

    prev_df = pd.DataFrame(
        [
            {"subject_id": k[0], "age_mo": k[1], "category": k[2], "stratum": k[3], "n_indexed": b.n_indexed}
            for k, b in buckets.items()
        ]
    )
    prev_df["cdi_domain"] = prev_df["category"].map(lambda c: DOMAIN_BY_CATEGORY.get(c, "other"))
    prev_df.to_csv(RESULTS_DIR / f"category_prevalence_by_context_{tag}.csv", index=False)

    if args.skip_embed_load:
        print(f"Prevalence only → {RESULTS_DIR}")
        return

    stems_needed = {s for b in buckets.values() for s in b.reservoir.items}
    stem_to_path: dict[str, Path] = {}
    stems_by_cat: dict[str, set[str]] = defaultdict(set)
    for stem in stems_needed:
        if stem in stem_meta:
            stems_by_cat[stem_meta[stem][2]].add(stem)
    for cat, stems in tqdm(stems_by_cat.items(), desc="Resolve npy"):
        cat_dir = EMBED_DIR / cat
        if not cat_dir.is_dir():
            continue
        for stem in stems:
            p = cat_dir / f"{stem}.npy"
            if p.is_file():
                stem_to_path[stem] = p

    centroids: dict[str, np.ndarray] = {}
    geom_rows = []
    for k, b in buckets.items():
        stats = finalize_vecs(b.reservoir.items, stem_to_path)
        if stats is None or stats["n_crops"] < args.min_crops:
            continue
        sid, age_mo, cat, st = k
        centroids[f"{sid}|{age_mo}|{cat}|{st}"] = stats.pop("centroid")
        geom_rows.append(
            {
                "subject_id": sid,
                "age_mo": age_mo,
                "category": cat,
                "stratum": st,
                "cdi_domain": DOMAIN_BY_CATEGORY.get(cat, "other"),
                "n_indexed": b.n_indexed,
                **stats,
            }
        )

    geom_df = pd.DataFrame(geom_rows).merge(load_ccn_metrics(), on="category", how="left")
    edges_df = compute_displacement_edges(geom_df, centroids, args.min_crops)

    np.savez_compressed(RESULTS_DIR / f"centroids_by_context_{tag}.npz", **centroids)
    geom_df.to_csv(RESULTS_DIR / f"category_geometry_by_context_{tag}.csv", index=False)
    edges_df.to_csv(RESULTS_DIR / f"category_displacement_by_context_{tag}.csv", index=False)

    cross = (
        geom_df[geom_df["stratum"] == "all"]
        .groupby("category")
        .agg(mean_dispersion=("dispersion", "mean"), n_windows=("dispersion", "count"))
        .reset_index()
        .merge(load_ccn_metrics(), on="category")
    )
    cross.to_csv(RESULTS_DIR / f"category_cross_sectional_ccn_merge_{tag}.csv", index=False)

    # Link dispersion level vs displacement (same stratum, per category)
    link_rows = []
    for cat in categories:
        for st in ALL_STRATA:
            g = geom_df[(geom_df["category"] == cat) & (geom_df["stratum"] == st)]
            e = edges_df[(edges_df["category"] == cat) & (edges_df["stratum"] == st)]
            if len(g) < 3 or len(e) < 3:
                continue
            link_rows.append(
                {
                    "category": cat,
                    "stratum": st,
                    "cdi_domain": DOMAIN_BY_CATEGORY.get(cat, "other"),
                    "mean_dispersion": g["dispersion"].mean(),
                    "mean_displacement": e["centroid_displacement"].mean(),
                    "n_windows": len(g),
                    "n_edges": len(e),
                }
            )
    link_df = pd.DataFrame(link_rows)
    link_df.to_csv(RESULTS_DIR / f"dispersion_displacement_link_{tag}.csv", index=False)

    # --- Figures ---
    plot_dispersion_context_compare(
        geom_df,
        ["cup", "plate"],
        ("all", "kitchen_like"),
        FIGURES_DIR / f"household_cup_plate_dispersion_kitchen_vs_all_{tag}.png",
        "Household: dispersion (kitchen vs all)",
    )
    plot_dispersion_context_compare(
        geom_df,
        ["cup", "plate"],
        ("all", "living_room"),
        FIGURES_DIR / f"household_cup_plate_dispersion_livingroom_vs_all_{tag}.png",
        "Household: dispersion (living room vs all)",
    )
    plot_dispersion_context_compare(
        geom_df,
        ["dog", "cat"],
        ("all", "outside"),
        FIGURES_DIR / f"animals_dog_cat_dispersion_outside_vs_all_{tag}.png",
        "Animals: dispersion (outside vs all)",
    )
    plot_dispersion_context_compare(
        geom_df,
        ["car", "stroller"],
        ("all", "outside"),
        FIGURES_DIR / f"vehicles_car_stroller_dispersion_outside_vs_all_{tag}.png",
        "Vehicles: dispersion (outside vs all)",
    )

    plot_displacement_context_compare(
        edges_df,
        ["cup", "plate"],
        ("all", "kitchen_like"),
        FIGURES_DIR / f"household_cup_plate_displacement_kitchen_vs_all_{tag}.png",
        "Household: centroid displacement",
    )
    plot_displacement_context_compare(
        edges_df,
        ["dog", "cat"],
        ("all", "outside"),
        FIGURES_DIR / f"animals_dog_cat_displacement_outside_vs_all_{tag}.png",
        "Animals: centroid displacement",
    )
    plot_displacement_context_compare(
        edges_df,
        ["car", "stroller"],
        ("all", "outside"),
        FIGURES_DIR / f"vehicles_car_stroller_displacement_outside_vs_all_{tag}.png",
        "Vehicles: centroid displacement",
    )
    plot_domain_summary(edges_df, geom_df, FIGURES_DIR / f"domain_context_summary_{tag}.png")

    meta = {
        "categories": sorted(categories),
        "strata": list(ALL_STRATA),
        "n_geom_rows": len(geom_df),
        "n_edges": len(edges_df),
        "domains": {d: [c for c in categories if DOMAIN_BY_CATEGORY.get(c) == d] for d in ("household", "animals", "vehicles")},
    }
    (RESULTS_DIR / f"run_meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"Done → {OUTPUT_ROOT}")
    print(f"  Edges: {len(edges_df)} | Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
