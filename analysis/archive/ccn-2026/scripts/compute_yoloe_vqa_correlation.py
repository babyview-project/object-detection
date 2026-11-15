#!/usr/bin/env python3
"""YOLOE vs unconstrained-VQA frequency correlation (frame prevalence).

Compares log frame prevalence from:
  - YOLOE: full infant-view valid129 pool (0.27-filtered detections)
  - VQA: unconstrained object mentions (frames with lemma / total VQA frames)

Uses the fixed 99-category overlap from ``all_objects.csv`` (source=vqa_yoloe).

Run from repo root::

  python analysis/ccn-2026/scripts/compute_yoloe_vqa_correlation.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "data"
SHARED_VQA = DATA / "shared_data_manuscript_2026" / "vqa_detections"
CCN_DIR = REPO_ROOT / "analysis" / "ccn-2026"

VQA_COUNTS = SHARED_VQA / "unconstrained_objects" / "vqa_object_counts.csv"
ALL_OBJECTS = SHARED_VQA / "unconstrained_objects" / "all_objects.csv"
YOLOE_FREQ = DATA / "long_tailed_dist_prop_included_categories_filtered-0.27_valid129.csv"
VALID85_LIST = DATA / "included_categories_valid85.txt"

OUT_JSON = CCN_DIR / "valid7018" / "yoloe_vqa_correlation.json"
OUT_CSV = CCN_DIR / "valid7018" / "yoloe_vqa_correlation_by_category.csv"
OUT_FIG_PNG = SHARED_VQA / "figures" / "vqa_comparison_frame_prevalence.png"
OUT_FIG_PDF = SHARED_VQA / "figures" / "vqa_comparison_frame_prevalence.pdf"
OUT_FIG_LEGACY_PNG = DATA / "figures" / "vqa_comparison.png"
OUT_FIG_LEGACY_PDF = DATA / "figures" / "vqa_comparison.pdf"
OUT_FIG_SI_PNG = (
    REPO_ROOT / "analysis" / "manuscript-2026" / "figures" / "si" / "vqa_comparison.png"
)
OUT_FIG_SI_PDF = (
    REPO_ROOT / "analysis" / "manuscript-2026" / "figures" / "si" / "vqa_comparison.pdf"
)


def load_overlap_table() -> pd.DataFrame:
    overlap = pd.read_csv(ALL_OBJECTS)
    overlap["object"] = overlap["object"].astype(str).str.strip().str.lower()
    overlap = overlap[overlap["source"] == "vqa_yoloe"].copy()

    vqa = pd.read_csv(VQA_COUNTS)
    vqa["cleaned_lemma"] = vqa["cleaned_lemma"].astype(str).str.strip().str.lower()
    vqa = vqa.rename(columns={"cleaned_lemma": "object", "proportion": "vqa_frame_prevalence"})

    yoloe = pd.read_csv(YOLOE_FREQ)
    yoloe["category"] = yoloe["category"].astype(str).str.strip().str.lower()
    yoloe = yoloe.rename(
        columns={
            "category": "object",
            "proportion": "yoloe_frame_prevalence",
            "count_instances": "yoloe_count_instances",
        }
    )

    valid85 = {
        line.strip().lower()
        for line in VALID85_LIST.read_text().splitlines()
        if line.strip()
    }

    merged = (
        overlap[["object"]]
        .merge(vqa[["object", "vqa_frame_prevalence", "frame_count"]], on="object", how="left")
        .merge(
            yoloe[["object", "yoloe_frame_prevalence", "yoloe_count_instances", "count_frames"]],
            on="object",
            how="left",
        )
    )
    merged["in_valid85"] = merged["object"].isin(valid85)
    merged["yoloe_detection_proportion"] = (
        merged["yoloe_count_instances"] / merged["yoloe_count_instances"].sum()
    )
    merged = merged.dropna(subset=["vqa_frame_prevalence", "yoloe_frame_prevalence"])
    merged = merged[(merged["vqa_frame_prevalence"] > 0) & (merged["yoloe_frame_prevalence"] > 0)]
    if len(merged) != len(overlap):
        missing = set(overlap["object"]) - set(merged["object"])
        raise RuntimeError(
            f"Expected {len(overlap)} overlapping categories, matched {len(merged)}; missing: {sorted(missing)}"
        )
    return merged.reset_index(drop=True)


def correlation_block(df: pd.DataFrame, xcol: str, ycol: str) -> dict:
    x = np.log(df[xcol].astype(float))
    y = np.log(df[ycol].astype(float))
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(df[xcol], df[ycol])
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    resid = np.abs(y - (intercept + slope * x))
    return {
        "n_categories": int(len(df)),
        "log_log_pearson_r": float(pr),
        "log_log_pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "log_log_slope": float(slope),
        "log_log_intercept": float(intercept),
        "outlier_threshold_residual": float(np.quantile(resid, 0.85)),
    }


def _annotate_nonoverlapping(
    ax,
    outliers: pd.DataFrame,
    color_map: dict[str, str],
    fontsize: float = 10,
    max_labels: int = 12,
) -> None:
    """Place category labels inside axes via axes-fraction repulsion (always on-plot)."""
    to_label = outliers.sort_values("resid", ascending=False).head(max_labels).reset_index(drop=True)
    if to_label.empty:
        return

    # Convert data -> axes fraction for layout.
    coords = np.array(
        [ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
         for x, y in zip(to_label["log_prop"], to_label["log_yoloe"])],
        dtype=float,
    )
    # Initial nudge so labels aren't centered on markers.
    pos = coords.copy()
    pos[:, 1] += 0.035

    # Approximate label half-size in axes fraction.
    half_w, half_h = 0.062, 0.028
    margin = 0.025
    point_r = 0.02

    rng = np.random.default_rng(0)
    for _ in range(500):
        # Clamp inside axes.
        pos[:, 0] = np.clip(pos[:, 0], margin + half_w, 1.0 - margin - half_w)
        pos[:, 1] = np.clip(pos[:, 1], margin + half_h, 1.0 - margin - half_h)

        forces = np.zeros_like(pos)
        # Repel from other labels.
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[i] - pos[j]
                # Separation needed for non-overlap of rectangles.
                sx = abs(d[0]) - 2 * half_w
                sy = abs(d[1]) - 2 * half_h
                if sx < 0 and sy < 0:
                    # Push along the weaker overlapping axis.
                    if sx > sy:
                        push = np.array([np.sign(d[0]) or 1.0, 0.0]) * (-sx + 0.006)
                    else:
                        push = np.array([0.0, np.sign(d[1]) or 1.0]) * (-sy + 0.006)
                    forces[i] += push
                    forces[j] -= push
            # Soft spring toward own point (keep leader line short).
            to_pt = coords[i] - pos[i]
            dist = np.linalg.norm(to_pt)
            if dist > 0.14:
                forces[i] += 0.12 * to_pt
            elif dist < point_r + 0.012:
                n = to_pt / (dist + 1e-9)
                forces[i] -= 0.1 * n
            forces[i] += rng.normal(0, 0.0005, size=2)

        pos += 0.4 * forces

    # Final clamp.
    pos[:, 0] = np.clip(pos[:, 0], margin + half_w, 1.0 - margin - half_w)
    pos[:, 1] = np.clip(pos[:, 1], margin + half_h, 1.0 - margin - half_h)

    for i, row in to_label.iterrows():
        color = color_map[str(row["category"])]
        x, y = float(row["log_prop"]), float(row["log_yoloe"])
        ax.annotate(
            str(row["object"]),
            xy=(x, y),
            xycoords="data",
            xytext=(float(pos[i, 0]), float(pos[i, 1])),
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.7,
                "alpha": 0.96,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "0.5",
                "lw": 0.6,
                "shrinkA": 0,
                "shrinkB": 3,
            },
            zorder=4,
            clip_on=True,
        )


def plot_scatter(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    """Log-log YOLOE vs VQA frame-prevalence scatter with in-axes labels."""
    # Editable text in Illustrator / Inkscape.
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 13,
            "axes.labelsize": 13.5,
            "axes.titlesize": 14,
            "legend.fontsize": 11.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    sub = df.copy()
    sub["log_prop"] = np.log(sub["vqa_frame_prevalence"])
    sub["log_yoloe"] = np.log(sub["yoloe_frame_prevalence"])
    sub["category"] = np.where(sub["in_valid85"], "high-precision subset", "other overlap")

    slope, intercept, _, _, _ = stats.linregress(sub["log_prop"], sub["log_yoloe"])
    sub["resid"] = np.abs(sub["log_yoloe"] - (intercept + slope * sub["log_prop"]))
    sub["is_outlier"] = False
    top_idx = sub.sort_values("resid", ascending=False).head(12).index
    sub.loc[top_idx, "is_outlier"] = True

    color_map = {"high-precision subset": "#1f78b4", "other overlap": "#7a7a7a"}
    point_size = 78

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    for cat, label in (
        ("other overlap", "other overlap"),
        ("high-precision subset", "high-precision subset"),
    ):
        pts = sub[sub["category"] == cat]
        if pts.empty:
            continue
        ax.scatter(
            pts["log_prop"],
            pts["log_yoloe"],
            s=point_size,
            alpha=0.5 if cat == "other overlap" else 0.55,
            color=color_map[cat],
            linewidths=0,
            label=label,
            zorder=3,
        )

    xmin, xmax = float(sub["log_prop"].min()), float(sub["log_prop"].max())
    ymin, ymax = float(sub["log_yoloe"].min()), float(sub["log_yoloe"].max())
    xpad = 0.12 * (xmax - xmin)
    ypad = 0.12 * (ymax - ymin)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    xline = np.linspace(xmin, xmax, 100)
    ax.plot(
        xline,
        intercept + slope * xline,
        color="gray",
        alpha=0.45,
        linewidth=1.2,
        zorder=1,
        label="all-overlap fit",
    )

    hps = sub[sub["category"] == "high-precision subset"]
    if len(hps) >= 2:
        s85, i85, _, _, _ = stats.linregress(hps["log_prop"], hps["log_yoloe"])
        ax.plot(
            xline,
            i85 + s85 * xline,
            color="#1f78b4",
            alpha=0.45,
            linewidth=1.2,
            zorder=1,
            label="high-precision fit",
        )

    ax.set_xlabel("log(frame prevalence) from VQA model detections")
    ax.set_ylabel("log(frame prevalence) from YOLOE detections")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")

    legend = ax.legend(frameon=False, loc="lower right", scatterpoints=1, markerscale=1.0)
    if legend:
        for handle in legend.legend_handles:
            if hasattr(handle, "set_alpha"):
                handle.set_alpha(1.0)
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([point_size])

    # Draw once so transforms are valid, then place labels inside axes.
    fig.canvas.draw()
    outliers = sub[sub["is_outlier"]].reset_index(drop=True)
    _annotate_nonoverlapping(ax, outliers, color_map, fontsize=10, max_labels=12)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def append_writeup_stats(summary: dict) -> None:
    table_path = CCN_DIR / "valid7018_writeup_stats_table.csv"
    if not table_path.is_file():
        return
    rows = [
        ("yoloe_vqa_correlation", "definition", "frame prevalence both sides", "log-log Pearson on overlapping categories"),
        (
            "yoloe_vqa_correlation",
            "all_overlap",
            f"r={summary['all_overlap']['log_log_pearson_r']:.3f}, n={summary['all_overlap']['n_categories']}",
            f"p={summary['all_overlap']['log_log_pearson_p']:.2e}; Spearman rho={summary['all_overlap']['spearman_rho']:.3f}",
        ),
        (
            "yoloe_vqa_correlation",
            "valid85_overlap",
            f"r={summary['valid85_overlap']['log_log_pearson_r']:.3f}, n={summary['valid85_overlap']['n_categories']}",
            f"p={summary['valid85_overlap']['log_log_pearson_p']:.2e}; high-precision YOLOE subset",
        ),
        (
            "yoloe_vqa_correlation",
            "legacy_detection_proportion",
            f"r={summary['legacy_yoloe_detection_proportion']['log_log_pearson_r']:.3f}, n={summary['legacy_yoloe_detection_proportion']['n_categories']}",
            "sensitivity: old YOLOE detection-instance proportion vs VQA frame prevalence",
        ),
    ]
    existing = table_path.read_text().splitlines()
    existing = [line for line in existing if not line.startswith("yoloe_vqa_correlation,")]
    existing.extend(",".join(row) for row in rows)
    table_path.write_text("\n".join(existing) + "\n")


def main() -> int:
    df = load_overlap_table()
    all_stats = correlation_block(df, "vqa_frame_prevalence", "yoloe_frame_prevalence")
    v85_stats = correlation_block(
        df[df["in_valid85"]], "vqa_frame_prevalence", "yoloe_frame_prevalence"
    )
    legacy_stats = correlation_block(
        df, "vqa_frame_prevalence", "yoloe_detection_proportion"
    )

    summary = {
        "frequency_definition": "frame_prevalence",
        "vqa_source": str(VQA_COUNTS.relative_to(REPO_ROOT)),
        "yoloe_source": str(YOLOE_FREQ.relative_to(REPO_ROOT)),
        "overlap_source": str(ALL_OBJECTS.relative_to(REPO_ROOT)),
        "all_overlap": all_stats,
        "valid85_overlap": v85_stats,
        "legacy_yoloe_detection_proportion": legacy_stats,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    df.to_csv(OUT_CSV, index=False)
    plot_scatter(df, OUT_FIG_PNG, OUT_FIG_PDF)

    for src, dst in (
        (OUT_FIG_PNG, OUT_FIG_LEGACY_PNG),
        (OUT_FIG_PDF, OUT_FIG_LEGACY_PDF),
        (OUT_FIG_PNG, OUT_FIG_SI_PNG),
        (OUT_FIG_PDF, OUT_FIG_SI_PDF),
    ):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    append_writeup_stats(summary)

    print(f"YOLOE vs VQA (frame prevalence): r={all_stats['log_log_pearson_r']:.3f}, n={all_stats['n_categories']}")
    print(f"  valid85 subset: r={v85_stats['log_log_pearson_r']:.3f}, n={v85_stats['n_categories']}")
    print(f"  legacy YOLOE detection proportion: r={legacy_stats['log_log_pearson_r']:.3f}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_FIG_PNG}")
    print(f"Wrote {OUT_FIG_PDF}")
    print(f"Copied to {OUT_FIG_LEGACY_PNG} and {OUT_FIG_LEGACY_PDF}")
    print(f"Copied to {OUT_FIG_SI_PNG} and {OUT_FIG_SI_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
