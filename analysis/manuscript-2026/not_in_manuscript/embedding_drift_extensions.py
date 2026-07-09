#!/usr/bin/env python3
"""Extensions for embedding visual drift pilot.

  1. Age-shuffled null (from cached centroids)
  2. 3-month age bins (orchestrates pilot run)
  3. CLIP vs DINOv3 comparison
  4. Correlate embedding drift with month-to-month RDM correlations

Usage:
  python analysis/manuscript-2026/not_in_manuscript/embedding_drift_extensions.py --all
  python analysis/manuscript-2026/not_in_manuscript/embedding_drift_extensions.py --null-only --tag dinov3_valid129_bin1_max256_top8
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _paths import NOT_IN_MANUSCRIPT_DIR, PROJECT_ROOT  # noqa: E402

SCRIPT_DIR = NOT_IN_MANUSCRIPT_DIR
RESULTS_DIR = SCRIPT_DIR / "embedding_drift_exploration" / "results"
FIGURES_DIR = SCRIPT_DIR / "embedding_drift_exploration" / "figures"
PILOT_SCRIPT = SCRIPT_DIR / "embedding_visual_drift_pilot.py"

RDM_CSV = (
    PROJECT_ROOT
    / "analysis"
    / "individual_analyses"
    / "developmental_trajectory_top8_densest"
    / "month_to_month_correlations.csv"
)

TOP_N = 8
NULL_SHUFFLES = 500
NULL_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embedding drift extensions (null, bins, CLIP, RDM).")
    p.add_argument("--all", action="store_true", help="Run missing pilots + all extension analyses.")
    p.add_argument("--run-pilots", action="store_true", help="Run clip / 3-mo pilots if outputs missing.")
    p.add_argument("--null-only", action="store_true")
    p.add_argument("--rdm-only", action="store_true")
    p.add_argument("--compare-only", action="store_true")
    p.add_argument("--tag", default=None, help="Base tag for null-only (default: dinov3 monthly top8).")
    p.add_argument("--null-shuffles", type=int, default=NULL_SHUFFLES)
    p.add_argument("--top-n", type=int, default=TOP_N)
    return p.parse_args()


def tag(embed_model: str, age_bin_months: int, top_n: int) -> str:
    top = f"_top{top_n}" if top_n > 0 else ""
    return f"{embed_model}_valid129_bin{age_bin_months}_max256{top}"


def paths_for(t: str) -> dict[str, Path]:
    return {
        "edges_g": RESULTS_DIR / f"global_trajectory_edges_{t}.csv",
        "edges_c": RESULTS_DIR / f"category_trajectory_edges_{t}.csv",
        "windows_g": RESULTS_DIR / f"global_windows_{t}.csv",
        "centroids": RESULTS_DIR / f"centroids_full_{t}.npz",
    }


def run_pilot(embed_model: str, age_bin_months: int, top_n: int) -> None:
    t = tag(embed_model, age_bin_months, top_n)
    if paths_for(t)["edges_g"].exists():
        print(f"  skip pilot (exists): {t}")
        return
    cmd = [
        sys.executable,
        str(PILOT_SCRIPT),
        "--embed-model",
        embed_model,
        "--age-bin-months",
        str(age_bin_months),
        "--top-n",
        str(top_n),
    ]
    print(f"  running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_centroids_by_subject(npz_path: Path) -> dict[str, dict[int, np.ndarray]]:
    z = np.load(npz_path)
    out: dict[str, dict[int, np.ndarray]] = {}
    for key, vec in z.items():
        if not str(key).startswith("g|"):
            continue
        _, sid, age_s = str(key).split("|", 2)
        out.setdefault(sid, {})[int(age_s)] = np.asarray(vec, dtype=np.float64)
    return out


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


def age_shuffled_null(
    centroids_path: Path,
    windows_g_path: Path,
    edges_g_path: Path,
    age_bin_months: int,
    n_shuffles: int,
    seed: int,
) -> pd.DataFrame:
    """Shuffle age labels of centroids within subject; recompute consecutive displacements."""
    def _norm_sid(sid) -> str:
        s = str(sid).strip().lstrip("S")
        return s.zfill(8) if s.isdigit() else s

    by_subj = load_centroids_by_subject(centroids_path)
    edges_real = pd.read_csv(edges_g_path)
    edges_real["subject_id"] = edges_real["subject_id"].map(_norm_sid)
    rng = np.random.default_rng(seed)

    # Real displacements aligned to edges
    real_vals = edges_real["centroid_displacement"].to_numpy()

    null_pool: list[float] = []
    per_edge_null_means: list[float] = []

    # Build calendar edges template per subject from real edges
    templates: dict[str, list[tuple[int, int]]] = {}
    for row in edges_real.itertuples(index=False):
        templates.setdefault(_norm_sid(row.subject_id), []).append((int(row.age_from), int(row.age_to)))

    for rep in range(n_shuffles):
        rep_vals = []
        for sid, age_map in by_subj.items():
            ages = sorted(age_map.keys())
            if len(ages) < 2:
                continue
            perm = ages.copy()
            rng.shuffle(perm)
            shuffled = {ages[i]: age_map[perm[i]] for i in range(len(ages))}
            edge_list = templates.get(sid, [])
            for age_from, age_to in edge_list:
                if age_from not in shuffled or age_to not in shuffled:
                    continue
                rep_vals.append(cosine_distance(shuffled[age_from], shuffled[age_to]))
        null_pool.extend(rep_vals)

    # Per-edge mean null: shuffle many times, match edge count structure
    n_edges = len(edges_real)
    for _ in range(n_shuffles):
        ev = []
        for sid, age_map in by_subj.items():
            ages = sorted(age_map.keys())
            if len(ages) < 2:
                continue
            perm = ages.copy()
            rng.shuffle(perm)
            shuffled = {ages[i]: age_map[perm[i]] for i in range(len(ages))}
            for age_from, age_to in templates.get(sid, []):
                if age_from in shuffled and age_to in shuffled:
                    ev.append(cosine_distance(shuffled[age_from], shuffled[age_to]))
        if ev:
            per_edge_null_means.append(float(np.mean(ev)))

    summary = {
        "n_shuffles": n_shuffles,
        "n_real_edges": int(n_edges),
        "real_mean": float(np.mean(real_vals)),
        "real_std": float(np.std(real_vals)),
        "null_mean": float(np.mean(null_pool)),
        "null_std": float(np.std(null_pool)),
        "difference_real_minus_null": float(np.mean(real_vals) - np.mean(null_pool)),
        "ttest_stat": float(stats.ttest_ind(real_vals, null_pool, equal_var=False).statistic),
        "ttest_p": float(stats.ttest_ind(real_vals, null_pool, equal_var=False).pvalue),
        "perm_mean_of_means": float(np.mean(per_edge_null_means)) if per_edge_null_means else np.nan,
    }
    cohen_d = summary["difference_real_minus_null"] / (
        np.sqrt((np.var(real_vals) + np.var(null_pool)) / 2) + 1e-12
    )
    summary["cohens_d"] = float(cohen_d)

    out_tag = centroids_path.stem.replace("centroids_full_", "")
    summary_path = RESULTS_DIR / f"null_age_shuffle_summary_{out_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Null summary: real={summary['real_mean']:.4f} null={summary['null_mean']:.4f} "
          f"Δ={summary['difference_real_minus_null']:.4f} p={summary['ttest_p']:.2e} d={summary['cohens_d']:.3f}")

    plot_null_comparison(real_vals, null_pool, summary, out_tag)
    return pd.DataFrame([summary])


def plot_null_comparison(real: np.ndarray, null_pool: np.ndarray, summary: dict, out_tag: str) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(real, bins=20, alpha=0.7, label=f"Real (mean={summary['real_mean']:.3f})", color="#4ECDC4")
    ax.hist(null_pool, bins=20, alpha=0.6, label=f"Null (mean={summary['null_mean']:.3f})", color="#FF6B6B")
    ax.set_xlabel("Centroid displacement")
    ax.set_ylabel("Count")
    ax.set_title("Age-shuffled null: global drift")
    ax.legend()

    ax = axes[1]
    bp = ax.boxplot(
        [real, null_pool[: len(real) * 20]],
        tick_labels=["Real", "Null draws"],
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], ["#4ECDC4", "#FF6B6B"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Centroid displacement")
    ax.set_title(f"Δ={summary['difference_real_minus_null']:.3f}, p={summary['ttest_p']:.2e}")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"null_age_shuffle_{out_tag}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"null_age_shuffle_{out_tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def correlate_with_rdm(edges_g_path: Path, rdm_path: Path, out_tag: str) -> pd.DataFrame:
    edges = pd.read_csv(edges_g_path)
    edges["subject_id"] = edges["subject_id"].astype(str).str.zfill(8)
    rdm = pd.read_csv(rdm_path)
    rdm["subject_id"] = rdm["subject_id"].astype(str).str.zfill(8)

    merged = edges.merge(
        rdm[["subject_id", "age_from", "age_to", "correlation", "hours_bin", "n_common_categories"]],
        on=["subject_id", "age_from", "age_to"],
        how="inner",
    )
    merged["rdm_drift"] = 1.0 - merged["correlation"]
    merged["embed_drift"] = merged["centroid_displacement"]

    rows = []
    for x, y, label in [
        ("embed_drift", "correlation", "displacement_vs_rdm_corr"),
        ("embed_drift", "rdm_drift", "displacement_vs_rdm_drift"),
        ("dispersion_delta", "correlation", "dispersion_delta_vs_rdm_corr"),
    ]:
        sub = merged[[x, y]].dropna()
        if len(sub) < 5:
            continue
        pr = stats.pearsonr(sub[x], sub[y])
        sr = stats.spearmanr(sub[x], sub[y])
        rows.append(
            {
                "comparison": label,
                "n": len(sub),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_rho": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
            }
        )

    corr_df = pd.DataFrame(rows)
    merged.to_csv(RESULTS_DIR / f"drift_rdm_merged_{out_tag}.csv", index=False)
    corr_df.to_csv(RESULTS_DIR / f"drift_rdm_correlations_{out_tag}.csv", index=False)
    print(f"  RDM merge n={len(merged)} | correlations:\n{corr_df.to_string(index=False)}")

    plot_rdm_scatter(merged, out_tag)
    return merged


def plot_rdm_scatter(merged: pd.DataFrame, out_tag: str) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    sns.scatterplot(data=merged, x="embed_drift", y="correlation", hue="subject_id", ax=ax, alpha=0.75)
    pr = stats.pearsonr(merged["embed_drift"], merged["correlation"])
    ax.set_xlabel("Embedding centroid displacement")
    ax.set_ylabel("Month-to-month RDM correlation")
    ax.set_title(f"Input drift vs representational stability\nr={pr.statistic:.3f}, p={pr.pvalue:.3g}")

    ax = axes[1]
    sns.scatterplot(data=merged, x="embed_drift", y="rdm_drift", hue="subject_id", ax=ax, alpha=0.75, legend=False)
    pr2 = stats.pearsonr(merged["embed_drift"], merged["rdm_drift"])
    ax.set_xlabel("Embedding centroid displacement")
    ax.set_ylabel("RDM drift (1 − correlation)")
    ax.set_title(f"Input vs representation drift\nr={pr2.statistic:.3f}, p={pr2.pvalue:.3g}")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"drift_vs_rdm_{out_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_backbones(tags: list[str], labels: list[str]) -> None:
    rows = []
    for t, lab in zip(tags, labels):
        p = paths_for(t)["edges_g"]
        if not p.exists():
            continue
        e = pd.read_csv(p)["centroid_displacement"]
        rows.append({"backbone": lab, "tag": t, "mean": e.mean(), "std": e.std(), "n": len(e)})

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(RESULTS_DIR / "backbone_comparison_monthly_top8.csv", index=False)

    if len(cmp_df) < 2:
        return

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {"DINOv3": "#0d47a1", "CLIP": "#e65100"}
    for lab in cmp_df["backbone"]:
        t = cmp_df.loc[cmp_df["backbone"] == lab, "tag"].iloc[0]
        e = pd.read_csv(paths_for(t)["edges_g"])
        ax.hist(e["centroid_displacement"], bins=18, alpha=0.55, label=lab, color=palette.get(lab))
    ax.set_xlabel("Centroid displacement")
    ax.set_ylabel("Count")
    ax.set_title("Backbone comparison (top-8, monthly bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "backbone_comparison_monthly_top8.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Backbone comparison:\n{cmp_df.to_string(index=False)}")


def compare_bin_widths(tag_m1: str, tag_m3: str) -> None:
    rows = []
    for t, w in [(tag_m1, "1 mo"), (tag_m3, "3 mo")]:
        e = pd.read_csv(paths_for(t)["edges_g"])
        rows.append({"window": w, "mean_disp": e["centroid_displacement"].mean(), "n_edges": len(e)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "bin_width_comparison_top8_dinov3.csv", index=False)
    print(f"  Bin width comparison:\n{df.to_string(index=False)}")


def process_tag(t: str, age_bin_months: int, n_null: int, do_null: bool, do_rdm: bool) -> None:
    p = paths_for(t)
    if not p["centroids"].exists():
        print(f"  missing centroids for {t}; run pilot first")
        return
    if do_null:
        age_shuffled_null(p["centroids"], p["windows_g"], p["edges_g"], age_bin_months, n_null, NULL_SEED)
    if do_rdm and age_bin_months == 1:
        correlate_with_rdm(p["edges_g"], RDM_CSV, t)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tag_dino_m1 = tag("dinov3", 1, args.top_n)
    tag_dino_m3 = tag("dinov3", 3, args.top_n)
    tag_clip_m1 = tag("clip", 1, args.top_n)

    if args.all or args.run_pilots:
        print("Running pilots if needed…")
        run_pilot("dinov3", 1, args.top_n)
        run_pilot("dinov3", 3, args.top_n)
        run_pilot("clip", 1, args.top_n)

    do_null = args.all or args.null_only
    do_rdm = args.all or args.rdm_only
    do_cmp = args.all or args.compare_only

    if args.null_only and args.tag:
        process_tag(args.tag, 1, args.null_shuffles, True, False)
        return

    if do_null or do_rdm:
        print("Extensions: null + RDM (DINOv3 monthly)…")
        process_tag(tag_dino_m1, 1, args.null_shuffles, do_null, do_rdm)
        if paths_for(tag_dino_m3)["centroids"].exists() and do_null:
            print("Extensions: null (DINOv3 3-mo)…")
            process_tag(tag_dino_m3, 3, args.null_shuffles, True, False)

    if do_cmp:
        print("Extensions: comparisons…")
        compare_backbones([tag_dino_m1, tag_clip_m1], ["DINOv3", "CLIP"])
        if paths_for(tag_dino_m3)["edges_g"].exists():
            compare_bin_widths(tag_dino_m1, tag_dino_m3)

    print(f"Done. Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
