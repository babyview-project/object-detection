#!/usr/bin/env python3
"""Write a shareable PDF of the 868-hour vs 2025.2 comparison canvas."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from PIL import Image

MS = Path(__file__).resolve().parents[1]
FIG868 = MS / "main_results_valid129s_04302026" / "figures"
FIG252 = MS / "main_results_valid129s_2025.2" / "figures"
OUT = MS / "868h_vs_2025.2_comparison.pdf"

PAGE = (11.0, 8.5)  # landscape letter
INK = "#1a1a1a"
MUTED = "#4a4a4a"
RULE = "#d0d0d0"
WARN_BG = "#fff4d6"
WARN_EDGE = "#c9a227"
HEADER_BG = "#f3f3f3"


def _new(pdf: PdfPages, title: str):
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.subplots_adjust(left=0.045, right=0.97, top=0.90, bottom=0.055)
    fig.text(0.045, 0.955, "868-hour vs 2025.2  ·  BabyView Objects", fontsize=8, color=MUTED)
    fig.text(0.97, 0.955, "2 Sep 2026", fontsize=8, color=MUTED, ha="right")
    fig.text(0.045, 0.915, title, fontsize=14, color=INK, fontweight="bold", va="center")
    fig.lines.append(plt.Line2D([0.045, 0.97], [0.895, 0.895], transform=fig.transFigure, color=RULE, lw=0.8))
    fig.text(0.045, 0.028, "CLIP + DINOv3  ·  valid129 unless noted  ·  2025.2 is the detected 2025 slice, not the full extract", fontsize=7, color=MUTED)
    return fig


def _wrap(ax, x, y, text, *, width=108, fontsize=8.5, color=INK, weight="normal", va="top"):
    import textwrap

    t = "\n".join(textwrap.wrap(text, width=width))
    ax.text(x, y, t, fontsize=fontsize, color=color, fontweight=weight, va=va, ha="left", transform=ax.transAxes, linespacing=1.35)
    return t.count("\n") + 1


def _table(ax, headers, rows, col_widths=None, fontsize=7.5):
    ax.set_axis_off()
    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper left",
        cellLoc="left",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color(INK)
        else:
            cell.set_facecolor("white" if r % 2 else "#fafafa")
            cell.get_text().set_color(INK)
    return tbl


def _load(path: Path, max_side: int = 1800) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(im)


def _pair(fig, rects, file_name: str, captions=("868-hour (published slice)", "2025.2 (detected 2025 slice)")):
    imgs = [_load(FIG868 / file_name), _load(FIG252 / file_name)]
    for rect, img, cap in zip(rects, imgs, captions):
        ax = fig.add_axes(rect)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(cap, fontsize=8, color=INK, pad=4, loc="left")


def page_cover(pdf: PdfPages) -> None:
    fig = _new(pdf, "868-hour vs 2025.2 manuscript results")
    ax = fig.add_axes([0.045, 0.06, 0.925, 0.80])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box = FancyBboxPatch((0.0, 0.78), 1.0, 0.20, boxstyle="round,pad=0.008,rounding_size=0.01", linewidth=0.8, edgecolor=WARN_EDGE, facecolor=WARN_BG, transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(0.015, 0.96, "2025.2 analyses are not the full extract", fontsize=10, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    _wrap(
        ax,
        0.015,
        0.91,
        "YOLOE/CDI was run on 2,969 of 8,566 extracted videos (23 of 37 subjects) — almost only 2025 recordings, plus a thin slice of late 2024. The other 5,597 videos have frames but no detections; 5,459 of those (97.5%) are the 868-hour sessions re-extracted under new Sxxxx_…_rec… IDs. This comparison is published 868-hour (2023–2024) vs a later, largely non-overlapping 2025 slice — not a rerun of the same videos, and not the ~1,500-hour 2025.2 extract.",
        width=145,
        fontsize=8,
    )

    stats = [
        ("2,969 / 8,566", "Videos with CDI detections"),
        ("23 / 37", "Subjects with any detections"),
        ("5,459", "Undetected videos matching 868-hour sessions"),
        ("1 / 2,969", "CDI videos that overlap an 868-hour session"),
        ("0.669 → 0.660", "BV–THINGS RDM r (CLIP, 129)"),
        ("0.549 → 0.555", "BV–THINGS RDM r (DINOv3, 129)"),
        ("−0.031", "Mean category cosine shift (both models)"),
        ("r = 0.90", "Category-cosine rank preserved across runs"),
    ]
    for i, (val, lab) in enumerate(stats):
        x = 0.01 + (i % 4) * 0.25
        y = 0.68 if i < 4 else 0.52
        ax.text(x, y, val, fontsize=13, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
        ax.text(x, y - 0.07, lab, fontsize=7.5, color=MUTED, va="top", transform=ax.transAxes)

    ax.text(0.0, 0.38, "What ran", fontsize=11, fontweight="bold", color=INK, transform=ax.transAxes)
    tax = fig.add_axes([0.045, 0.07, 0.925, 0.32])
    _table(
        tax,
        ["Analysis", "868-hour", "2025.2", "Comparable?"],
        [
            ["01 long-tail / frame prevalence", "yes", "yes", "same lists; different video pools"],
            ["02 category cosine vs THINGS", "CLIP, DINO, BabyDINO", "CLIP, DINO", "yes (category-level)"],
            ["03 BV–THINGS RDM (category pairs)", "3 models", "CLIP, DINO", "yes (category-level)"],
            ["03 / 05 CDI cluster coloring", "9 domains", "9 domains", "yes"],
            ["04 individual RDMs (8 kids)", "yes", "reran 2026-09-02", "thinner overlap on 2025 slice"],
            ["Binary CDI-template vs RDM", "yes", "yes (9-domain map)", "yes"],
            ["CLIP threshold 0.26 / 0.27 / 0.28", "yes", "yes", "yes"],
            ["08 top-8 pairwise cosine", "yes", "yes", "thinner overlap"],
            ["09 local / global k=5", "not in this tree", "yes", "2025.2 only"],
            ["10 animal depiction", "yes (valid85)", "no", "—"],
        ],
        col_widths=[0.32, 0.20, 0.22, 0.26],
        fontsize=7,
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_longtail(pdf: PdfPages) -> None:
    fig = _new(pdf, "01 — frame prevalence / long tail")
    ax = fig.add_axes([0.045, 0.78, 0.925, 0.10])
    ax.set_axis_off()
    _wrap(
        ax,
        0,
        1,
        "Unique CLIP-pass frames: 1,715,977 (868-hour) vs 1,410,513 (2025.2 detected slice). That drop is expected: 2025.2 analyses never see the 2023–2024 sessions that make up 868-hour. Rank Pearson 0.92 / Spearman 0.95. Power-law α 1.90 → 1.84. The 2025 slice is more multi-label (sum of prevalences 1.38 → 2.33). Largest gains: cup +0.082, lamp +0.078, bottle +0.070, toy +0.066.",
        width=150,
        fontsize=8,
    )
    _pair(
        fig,
        ([0.045, 0.34, 0.45, 0.42], [0.52, 0.34, 0.45, 0.42]),
        "long_tailed_all_included_categories_filtered-0.27_valid129.png",
    )
    tax = fig.add_axes([0.045, 0.06, 0.55, 0.26])
    _table(
        tax,
        ["Rank", "868-hour", "prop", "2025.2", "prop"],
        [
            ["1", "chair", "0.131", "lamp", "0.183"],
            ["2", "lamp", "0.106", "cup", "0.137"],
            ["3", "table", "0.094", "chair", "0.124"],
            ["4", "couch", "0.087", "toy", "0.121"],
            ["5", "pillow", "0.068", "table", "0.117"],
            ["6", "cup", "0.055", "bottle", "0.103"],
        ],
        col_widths=[0.12, 0.22, 0.16, 0.22, 0.16],
    )
    note = fig.add_axes([0.62, 0.06, 0.35, 0.26])
    note.set_axis_off()
    _wrap(
        note,
        0,
        1,
        "valid85 868-hour used the 6,982-frame annotation/VQA pool; 2025.2 valid85 uses the full CLIP-passer pool on the detected slice, so precision–prevalence r (0.33 → 0.24) is not the same pool. valid129 precision vs prevalence is 0.35 → 0.37.",
        width=48,
        fontsize=8,
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_rdm(pdf: PdfPages) -> None:
    fig = _new(pdf, "Representational geometry (the paper’s core claim)")
    ax = fig.add_axes([0.045, 0.80, 0.58, 0.08])
    ax.set_axis_off()
    _wrap(
        ax,
        0,
        1,
        "BV–THINGS RDM correlation is essentially unchanged on the later slice. CLIP vs DINOv3 agreement on BabyView is unchanged (0.907 → 0.904). THINGS z-scored centroids are bit-identical (mean cosine = 1.000).",
        width=90,
        fontsize=8,
    )
    bx = fig.add_axes([0.66, 0.78, 0.30, 0.10])
    cats = ["CLIP 129", "DINOv3 129", "CLIP 85", "DINOv3 85"]
    x = np.arange(len(cats))
    w = 0.38
    bx.bar(x - w / 2, [0.669, 0.549, 0.703, 0.624], w, label="868-hour", color="#4c78a8")
    bx.bar(x + w / 2, [0.660, 0.555, 0.695, 0.617], w, label="2025.2", color="#f58518")
    bx.set_xticks(x, cats, fontsize=6.5)
    bx.set_ylim(0.50, 0.75)
    bx.set_ylabel("Pearson r", fontsize=7)
    bx.tick_params(labelsize=6.5)
    bx.legend(fontsize=6.5, frameon=False)
    bx.set_title("BV vs THINGS category-RDM Pearson r", fontsize=8, loc="left", color=INK)
    for spine in ("top", "right"):
        bx.spines[spine].set_visible(False)
    _pair(
        fig,
        ([0.045, 0.06, 0.45, 0.68], [0.52, 0.06, 0.45, 0.68]),
        "bv_things_rdm_comparison_2x2_bv_first_clip_dinov3_filtered-0.27_valid129.png",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_cosine_cluster(pdf: PdfPages) -> None:
    fig = _new(pdf, "Category cosine vs THINGS  ·  CDI cluster strength")
    ax = fig.add_axes([0.045, 0.80, 0.925, 0.08])
    ax.set_axis_off()
    _wrap(
        ax,
        0,
        1,
        "Mean cosine dropped by ~0.03 on both backbones; ranking is preserved (CLIP Pearson 0.90 / Spearman 0.88). 71 of 129 CLIP categories dropped by >0.02; 13 rose. Mix shift on a 2025-only slice is a better explanation than “less curated frames.”",
        width=155,
        fontsize=8,
    )
    t1 = fig.add_axes([0.045, 0.58, 0.48, 0.22])
    _table(
        t1,
        ["Set / model", "Mean 868h", "Mean 2025.2", "Δ", "Across-run r"],
        [
            ["valid129 CLIP", "0.609", "0.578", "−0.031", "0.90"],
            ["valid129 DINOv3", "0.480", "0.449", "−0.031", "0.90"],
            ["valid85 CLIP", "0.615", "0.587", "−0.028", "0.92"],
            ["valid85 DINOv3", "0.496", "0.459", "−0.036", "0.91"],
        ],
        col_widths=[0.28, 0.18, 0.20, 0.16, 0.18],
        fontsize=7,
    )
    bx = fig.add_axes([0.56, 0.58, 0.40, 0.22])
    cats = ["animals", "body", "cloth.", "food", "furn.", "house.", "out.", "toys", "veh."]
    x = np.arange(len(cats))
    w = 0.38
    bx.bar(x - w / 2, [0.698, 0.626, 0.56, 0.579, 0.612, 0.624, 0.624, 0.599, 0.628], w, label="868-hour", color="#4c78a8")
    bx.bar(x + w / 2, [0.654, 0.616, 0.544, 0.553, 0.581, 0.582, 0.564, 0.583, 0.618], w, label="2025.2", color="#f58518")
    bx.set_xticks(x, cats, fontsize=6.5)
    bx.set_ylim(0.50, 0.75)
    bx.set_ylabel("Mean CLIP cosine", fontsize=7)
    bx.legend(fontsize=6.5, frameon=False, loc="lower left")
    bx.set_title("CLIP valid129 cosine by CDI domain", fontsize=8, loc="left")
    for spine in ("top", "right"):
        bx.spines[spine].set_visible(False)
    t2 = fig.add_axes([0.045, 0.34, 0.48, 0.22])
    _table(
        t2,
        ["Set / model", "BV Δ 868h", "BV Δ 2025.2", "THINGS Δ", "BV−TH 868h", "BV−TH 2025.2"],
        [
            ["valid129 CLIP", "0.239", "0.235", "0.183", "0.056", "0.051"],
            ["valid129 DINOv3", "0.181", "0.184", "0.069", "0.112", "0.115"],
            ["valid85 CLIP", "0.230", "0.243", "0.182", "0.048", "0.061"],
            ["valid85 DINOv3", "0.174", "0.195", "0.083", "0.090", "0.111"],
        ],
        col_widths=[0.22, 0.15, 0.16, 0.15, 0.16, 0.16],
        fontsize=6.5,
    )
    t3 = fig.add_axes([0.54, 0.34, 0.21, 0.22])
    _table(
        t3,
        ["Drop", "868h", "2025.2", "Δ"],
        [
            ["melon", "0.74", "0.54", "−0.20"],
            ["pizza", "0.78", "0.58", "−0.20"],
            ["swing", "0.58", "0.40", "−0.18"],
            ["apple", "0.58", "0.43", "−0.15"],
        ],
        col_widths=[0.30, 0.23, 0.25, 0.22],
        fontsize=6.5,
    )
    t4 = fig.add_axes([0.76, 0.34, 0.21, 0.22])
    _table(
        t4,
        ["Gain", "868h", "2025.2", "Δ"],
        [
            ["banana", "0.66", "0.75", "+0.10"],
            ["carrot", "0.38", "0.47", "+0.09"],
            ["pickle", "0.41", "0.47", "+0.06"],
            ["muffin", "0.51", "0.57", "+0.06"],
        ],
        col_widths=[0.30, 0.23, 0.25, 0.22],
        fontsize=6.5,
    )
    _pair(
        fig,
        ([0.045, 0.055, 0.45, 0.27], [0.52, 0.055, 0.45, 0.27]),
        "bv_vs_things_cluster_delta_bars_2panel_valid129.png",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_children(pdf: PdfPages) -> None:
    fig = _new(pdf, "Where the 2025 slice differs: children (notebook 04)")
    ax = fig.add_axes([0.045, 0.80, 0.50, 0.08])
    ax.set_axis_off()
    _wrap(
        ax,
        0,
        1,
        "Same eight densest subjects, but each child is only observed in 2025 sessions here. White stripes in 2025.2 RDMs are missing categories. DINOv3 inter-child agreement drops more than CLIP.",
        width=78,
        fontsize=8,
    )
    t1 = fig.add_axes([0.52, 0.72, 0.45, 0.16])
    _table(
        t1,
        ["", "868h", "2025.2"],
        [
            ["CLIP mean Pearson r", "0.797", "0.772"],
            ["CLIP mean Spearman ρ", "0.776", "0.726"],
            ["DINOv3 mean Pearson r", "0.816", "0.730"],
            ["DINOv3 mean Spearman ρ", "0.787", "0.683"],
            ["Mean shared categories", "128.8", "117.0"],
            ["Mean RDM pairs", "8,224", "6,812"],
        ],
        col_widths=[0.50, 0.25, 0.25],
        fontsize=7,
    )
    _pair(
        fig,
        ([0.045, 0.06, 0.45, 0.64], [0.52, 0.06, 0.45, 0.64]),
        "individual_rdms_top8_densest_bv_semantic_order_clip_filtered-0.27_valid129.png",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_pair_takeaways(pdf: PdfPages) -> None:
    fig = _new(pdf, "Pairwise child RDMs  ·  threshold  ·  takeaways")
    _pair(
        fig,
        ([0.045, 0.52, 0.45, 0.34], [0.52, 0.52, 0.45, 0.34]),
        "individual_rdm_pairwise_top8_heatmap_hist_clip_filtered-0.27_valid129.png",
    )
    t1 = fig.add_axes([0.045, 0.30, 0.58, 0.18])
    _table(
        t1,
        ["τ", "868h boxes", "2025.2 boxes", "CLIP r 868h", "CLIP r 2025.2", "DINO r 868h", "DINO r 2025.2"],
        [
            ["0.26", "5,879,098", "5,646,163", "0.665", "0.624", "0.550", "0.555"],
            ["0.27", "3,625,482", "3,816,220", "0.665", "0.660", "0.550", "0.555"],
            ["0.28", "1,997,987", "2,334,219", "0.698", "0.692", "0.565", "0.573"],
        ],
        col_widths=[0.10, 0.16, 0.16, 0.14, 0.16, 0.14, 0.16],
        fontsize=6.5,
    )
    t2 = fig.add_axes([0.64, 0.30, 0.33, 0.18])
    _table(
        t2,
        ["Model", "BV mean cos", "BV min", "THINGS"],
        [
            ["CLIP", "0.917", "0.609 (dish)", "1.000"],
            ["DINOv3", "0.888", "0.409 (dish)", "1.000"],
        ],
        col_widths=[0.22, 0.24, 0.32, 0.22],
        fontsize=6.5,
    )
    ax = fig.add_axes([0.045, 0.055, 0.925, 0.23])
    ax.set_axis_off()
    ax.text(0, 1.0, "Takeaways", fontsize=11, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    items = [
        ("Different videos, same category geometry.", "2025.2 CDI covers the new 2025 recordings, not the 868-hour sessions sitting in the same extract. BV–THINGS RDM r, CLIP–DINOv3 agreement, and stronger BabyView cluster structure still come back within ~0.01."),
        ("Do not read unique-frame counts as “2025.2 is smaller.”", "The extract is ~1,500 hours / 8,566 videos. Analyses only see 2,969 detected videos (then 2,876 after dropping person / picture / 00270001)."),
        ("Small systematic cosine drop.", "Mean category cosine vs THINGS is ~0.03 lower. Outdoor / rare objects move the most. Mix shift on a 2025-only slice is the better explanation."),
        ("Individual analyses are not interchangeable.", "The same eight kids share ~117 categories instead of ~129. DINOv3 inter-child Spearman falls from 0.79 to 0.68. Do not quote 868-hour child numbers as 2025.2 numbers."),
    ]
    y = 0.82
    for head, body in items:
        ax.text(0, y, head, fontsize=8, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
        _wrap(ax, 0.01, y - 0.07, body, width=155, fontsize=7.5, color=MUTED)
        y -= 0.20
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        pdf.infodict()["Title"] = "868-hour vs 2025.2 manuscript results"
        pdf.infodict()["Author"] = "BabyView Objects"
        page_cover(pdf)
        page_longtail(pdf)
        page_rdm(pdf)
        page_cosine_cluster(pdf)
        page_children(pdf)
        page_pair_takeaways(pdf)
    print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
