"""UNR talk: gold set v2 results. RAs (two) corrected the full-run Qwen pre-fill on BabyView frames: deleted wrong boxes,
fixed box / name / format, drew every missed object. Reads annotation/gold_set/out/private/ra_scores (score_ra_qwen_t1.py).

    python fig_annotation_talk.py [a01_scorecard a02_category_pr a03_format_confusion a04_examples]
      -> figures/a0*.png (aggregates only); figures/private/a04_corrections.png (BabyView frames, people-free)
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

import talk_style as ts

Q = ts.ANALYSIS / "qwen-open-vocab"
sys.path.insert(0, str(Q))
import fig_style as FS  # noqa: E402

GOLD = ts.ANALYSIS.parent / "annotation" / "gold_set" / "out" / "private"
RS = GOLD / "ra_scores"
DOM = pd.read_csv(Q / "out" / "private" / "open_renamed" / "categories.csv", index_col=0).domain.to_dict()
KEPT, DELETED, ADDED = "#2E9E6B", "#D64545", "#F0A030"
NAVY, TEAL = ts.DS_COLOR["bv"], "#4DB8A8"
# Never shown in example frames: any frame where either Qwen or an RA boxed a person or a face part.
PEOPLE = {"person", "child", "baby", "man", "woman", "girl", "boy", "face", "head", "hair", "eye", "eyes", "mouth", "nose",
          "ear", "people", "toddler", "adult", "lips", "teeth", "beard", "reflection", "mirror", "photo", "photograph"}
# ... nor any frame with a body part (the child's own arm, a parent's hand): the scoring script's BODY_PARTS list.
_src = (ts.ANALYSIS.parent / "annotation" / "gold_set" / "score_ra_qwen_t1.py").read_text()
exec(_src[_src.index("BODY_PARTS = "):_src.index("\n\n", _src.index("BODY_PARTS = "))])
NO_SHOW = PEOPLE | BODY_PARTS  # noqa: F821


def summary():
    s = pd.read_csv(RS / "summary.csv", index_col=0)
    ir = pd.read_csv(RS / "interrater.csv").iloc[0]
    return s, ir


def a01_scorecard():
    """Validation scorecard: one dot per child (size = frames), pooled value with ±1 SE (10,000 resamples of the children;
    annotation/gold_set/validation_uncertainty.py). No bars: every child is shown."""
    s, ir = summary()
    U = pd.read_csv(RS / "scorecard_uncertainty.csv").set_index(["set", "measure"])
    K = pd.read_csv(RS / "scorecard_by_child.csv")
    rows = [("Precision", "all", "precision", "Qwen boxes people kept"),
            ("Recall", "all", "recall", "people's objects Qwen found"),
            ("Recall, without body parts", "without body parts", "recall", ""),
            ("Name kept", "all", "name_kept", "of kept boxes"),
            ("Format kept", "all", "format_kept", "real / toy / picture / screen"),
            ("Qwen vs people (F1)", "all", "f1", ""),
            ("Person vs person (F1)", "shared frames", "interrater_f1", f"{int(ir.shared_frames_both_reviewed)} shared frames")]
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H * 0.92))
    y = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(0)
    for yi, (lab, st, m, sub) in zip(y, rows):
        k = K[(K.set == st)]
        ax.scatter(k[m], yi + rng.uniform(-.18, .18, len(k)), s=8 + k.frames * 0.5, color=ts.MUTED, alpha=.45, lw=0, zorder=1)
        u = U.loc[(st, m)]
        ax.errorbar(u.value, yi, xerr=u.se, fmt="D", ms=7, color=ts.INK, capsize=4, lw=1.8, zorder=3)
        ax.text(1.02, yi, f"{u.value:.2f} ± {u.se:.3f}", va="center", fontsize=12, fontweight="bold", transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [r[0] + (f"\n({r[3]})" if r[3] else "") for r in rows], fontsize=11)
    ax.set_xlim(0.5, 1.005); ax.set_xlabel("proportion"); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-.7, len(rows) - .4)
    rv = pd.read_csv(RS / "reviewed_frames.csv")
    n_frames = int(rv.frames_reviewed.sum() - ir.shared_frames_both_reviewed)        # shared frames counted once
    a = s.loc["all"]
    ax.set_title(f"{n_frames:,} BabyView frames from {int(U.loc[('all', 'precision')].children)} children · "
                 f"{int(a.qwen_boxes):,} Qwen boxes · {int(a.ra_boxes):,} after correction\n"
                 "dots: children (size = frames); black: all frames pooled, ± 1 SE (children resampled)",
                 fontsize=11, color=ts.MUTED, loc="left")
    fig.tight_layout()
    ts.save(fig, "a01_scorecard")


def a02_category_pr(min_n=60):
    b = pd.read_csv(RS / "by_name.csv", index_col=0)
    b = b[b.qwen_boxes >= min_n].copy(); b["dom"] = [DOM.get(n, "other") for n in b.index]
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W, ts.SLIDE_H))
    for n, r in b.iterrows():
        ax.scatter(r.recall, r.precision, s=25 + r.qwen_boxes * 0.6, color=FS.DOMAIN_COLOR.get(r.dom, "#8B9A9E"),
                   alpha=0.85, edgecolor="white", linewidth=1, zorder=3)
    lab = set(b.nsmallest(6, "recall").index) | set(b.nsmallest(4, "precision").index) | set(b.nlargest(4, "qwen_boxes").index)
    for n in lab:
        r = b.loc[n]
        dx, dy, ha = {"box": (-8, 6, "right"), "paper": (-8, -14, "right"), "ball": (6, -12, "left"), "bottle": (-8, 6, "right")}.get(n, (6, 4, "left"))
        ax.annotate(n, (r.recall, r.precision), xytext=(dx, dy), textcoords="offset points", fontsize=11,
                    fontweight="bold", ha=ha)
    ax.set_xlabel("recall (found by Qwen / boxed by people)"); ax.set_ylabel("precision (kept by people)")
    ax.set_xlim(0.45, 1.0); ax.set_ylim(0.65, 1.0)
    doms = [d for d in ts.DOMAIN_ORDER if d in set(b.dom)] + (["other"] if "other" in set(b.dom) else [])
    for d in doms:
        ax.scatter([], [], color=FS.DOMAIN_COLOR.get(d, "#8B9A9E"), s=60,
                   label=ts.DOMAIN_LABEL.get(d, "other (incl. generic “toy”)"))
    ax.legend(loc="lower left", ncol=2, fontsize=10, handletextpad=0.2, columnspacing=0.8)
    ax.set_title(f"categories with ≥{min_n} Qwen boxes; dot size = boxes", fontsize=12, color=ts.MUTED, loc="left")
    ts.save(fig, "a02_category_pr")


def a03_format_confusion():
    c = pd.read_csv(RS / "format_confusion.csv", index_col=0).drop(index="All", columns="All")
    order = ["real", "toy", "picture", "screen"]; c = c.loc[order, order]; p = c.div(c.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(ts.SLIDE_W * 0.62, ts.SLIDE_H * 0.92))
    ax.imshow(p.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            v, n = p.values[i, j], c.values[i, j]
            if n:
                ax.text(j, i, f"{v:.0%}\n({n:,})" if i == j else f"{n}", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="white" if v > 0.5 else ts.INK)
    ax.set_xticks(range(4), order); ax.set_yticks(range(4), order)
    ax.set_xlabel("format after correction"); ax.set_ylabel("Qwen's format")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ts.save(fig, "a03_format_confusion")


def a04_examples(n=3, seed=3):
    b = pd.read_csv(RS / "boxes.csv"); fr = pd.read_csv(GOLD / "frames_qwen_t1.csv").set_index("frame_uid")
    hit = np.zeros(len(b), bool)
    for col in ("q_name", "ra_name"):
        nm = b[col].fillna("").str.lower().str.replace("'", " ").str.strip()
        hit |= nm.isin(NO_SHOW) | nm.str.split().str[-1].fillna("").isin(NO_SHOW)
    bad = b.loc[hit, "frame_uid"].unique()
    g = b[~b.frame_uid.isin(bad)].groupby("frame_uid").outcome.value_counts().unstack(fill_value=0)
    g = g[(g.get("deleted", 0) >= 1) & (g.get("added", 0) >= 2) & (g.sum(1).between(8, 20))]
    pick = g.sample(min(len(g), n), random_state=seed).index
    fig, axs = plt.subplots(1, n, figsize=(ts.SLIDE_W, ts.SLIDE_H * 0.98), layout="constrained")
    for ax, uid in zip(np.atleast_1d(axs), pick):
        ax.imshow(Image.open(fr.loc[uid, "frame_path"])); ax.axis("off")
        for r in b[b.frame_uid == uid].itertuples():
            if r.outcome == "added":
                x0, y0, x1, y1, c, ls = r.ra_x0, r.ra_y0, r.ra_x1, r.ra_y1, ADDED, "-"
            else:
                x0, y0, x1, y1 = r.q_x0, r.q_y0, r.q_x1, r.q_y1
                c, ls = (KEPT, "-") if r.outcome == "matched" else (DELETED, "--")
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=c, lw=2.2, ls=ls))
    for lab, c in (("kept", KEPT), ("deleted by people", DELETED), ("added by people (Qwen missed)", ADDED)):
        fig.axes[0].plot([], [], color=c, lw=3, label=lab)
    fig.legend(loc="outside lower center", ncol=3, fontsize=12, frameon=False)
    print("frames:", list(pick))
    ts.save(fig, "a04_corrections", private=True)


if __name__ == "__main__":
    ts.setup()
    fns = sys.argv[1:] or ["a01_scorecard", "a02_category_pr", "a03_format_confusion", "a04_examples"]
    for f in fns:
        globals()[f]()
