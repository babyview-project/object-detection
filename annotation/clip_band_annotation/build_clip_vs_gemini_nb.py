#!/usr/bin/env python3
"""Write clip_vs_gemini.ipynb (code only, no outputs).

The notebook shows crops and frames, so it is only ever executed into out/private/
(git-ignored); the copy next to this script stays output-free:

  python build_clip_vs_gemini_nb.py
  jupyter nbconvert --to notebook --execute clip_vs_gemini.ipynb \
      --ExecutePreprocessor.kernel_name=vislearnlabpy \
      --output-dir out/private --output clip_vs_gemini_executed.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# CLIP filter vs Gemini verify on the CLIP-band sample

**Contains crops and frames of children's homes and family members: keep the executed copy in
`out/private/` only (never GitHub, Slack, shared drives).**

Sample: 7,155 YOLOE crops from 2025.2 (129 categories × 7 CLIP bands, 8 per cell, 22 children).
Gemini 2.5 Flash-Lite (Vertex, temperature 0, `MEDIUM` media resolution) answered the gold-pilot
*verify* question for three shuffled words (YOLOE's label, a same-domain foil, an other-domain foil)
in two conditions:

* **crop**: the crop alone (what CLIP scores)
* **context**: the crop, then the full frame with the same object outlined in magenta

"Gemini accepts" = it says YOLOE's label is the object. Foil acceptances measure how lenient it is.
Population figures weight each item by its category × band cell size, i.e. they describe all
13.2 M scored 2025.2 crops. Gemini is **not ground truth** (pilot: 89% hit on human-verified crops,
1–2% foil false alarms); human labels on this sample will turn agreement into accuracy."""),
code("""import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display

HERE = Path.cwd()
sys.path.insert(0, str(HERE))
import compare_clip_gemini as cg

OUT = HERE / "out" / "private"
IMG = OUT / "images"
MONT = OUT / "montages"; MONT.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(OUT / "gemini" / "items_scored.csv")
rng = np.random.default_rng(0)
T = 0.27   # cut-off used for the agree/disagree montages
s["clip_keep"] = s.clip_sim >= T
print(f"{len(s)} items, {s.class_name.nunique()} categories; Gemini answered "
      f"crop {s.g_crop.notna().sum()}, context {s.g_context.notna().sum()}")

# Reference categorical palette (dataviz skill, slots 1-4 in documented order; adjacent pairs validated)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK2, GRID = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 110, "savefig.facecolor": "white"})"""),

md("""## 1. Gemini acceptance by CLIP band

Both conditions rise with CLIP score, so the two agree on the *ordering*. But Gemini accepts a large
share of crops in bands every threshold we used would discard, while foil acceptance stays at 2–5% in
every band: it is not simply saying yes."""),
code("""bb = cg.by_band(s, rng)
display(bb[["band", "n", "population", "accept_crop", "accept_context", "fa_same_crop", "fa_other_crop",
            "rescued_by_context", "dropped_by_context"]].round(3))

x = np.arange(len(bb)); labels = [b.replace("lt", "<").replace("ge", "≥") for b in bb.band]
fig, ax = plt.subplots(figsize=(8, 4.3))
for c, col, name in (("crop", BLUE, "Gemini, crop only"), ("context", ORANGE, "Gemini, crop + frame")):
    ax.fill_between(x, bb[f"accept_{c}_lo"], bb[f"accept_{c}_hi"], color=col, alpha=0.15, lw=0)
    ax.plot(x, bb[f"accept_{c}"], color=col, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=name)
fa = (bb.fa_same_crop + bb.fa_same_context) / 2
ax.plot(x, fa, color=GREY, lw=2, ls="--", marker="o", ms=5, mec="white", mew=1.2, label="foil (same domain), both")
for t in (0.26, 0.27, 0.28):   # band edges: 0.26 between idx 2|3, 0.27 between 3|4, 0.28 between 4|5
    xi = {0.26: 2.5, 0.27: 3.5, 0.28: 4.5}[t]
    ax.axvline(xi, color=INK2, lw=0.8, ls=":"); ax.text(xi, 0.97, f"{t}", ha="center", va="top", color=INK2, fontsize=8,
                                                      bbox=dict(fc="white", ec="none", pad=1))
ax.text(x[0] - 0.15, bb.accept_context.iloc[0], "crop + frame", color=INK2, fontsize=9, ha="right", va="center")
ax.text(x[0] - 0.15, bb.accept_crop.iloc[0], "crop", color=INK2, fontsize=9, ha="right", va="center")
ax.text(x[0] - 0.15, fa.iloc[0], "foil", color=INK2, fontsize=9, ha="right", va="center")
ax.set_xlim(-1.1, len(x) - 0.6); ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
ax.set_xlabel("CLIP similarity band (crop vs \\"a photo of a {label}\\")"); ax.set_ylabel("Gemini says the label is right")
ax.set_title("Gemini acceptance rises with CLIP score but stays high below every cut-off", loc="left", fontsize=11, color=INK)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.93)); fig.tight_layout()
fig.savefig(HERE / "results" / "fig_accept_by_band.pdf"); plt.show()"""),

md("""## 2. Agreement at each CLIP cut-off (population-weighted)

Share of all 13.2 M scored crops in each cell of CLIP keep/reject × Gemini accept/reject."""),
code("""bt = cg.by_threshold(s)
display(bt.round(3))
segs = [("both_accept", "both accept", BLUE), ("clip_only", "CLIP only", ORANGE),
        ("gemini_only", "Gemini only", AQUA), ("both_reject", "both reject", YELLOW)]
fig, ax = plt.subplots(figsize=(8, 3.6))
rows = [f"{r.threshold:.2f} · {'crop' if r.gemini == 'crop' else 'crop + frame'}" for r in bt.itertuples()]
y = np.arange(len(bt))[::-1]
left = np.zeros(len(bt))
for k, name, col in segs:
    v = bt[k].values
    ax.barh(y, v, left=left, color=col, height=0.62, edgecolor="white", linewidth=2, label=name)
    for yi, l, vi in zip(y, left, v):
        if vi > 0.06: ax.text(l + vi / 2, yi, f"{vi:.0%}", ha="center", va="center", fontsize=8.5, color=INK)
    left += v
ax.set_yticks(y, rows); ax.set_xlim(0, 1); ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
ax.grid(axis="y", visible=False); ax.set_xlabel("share of all scored 2025.2 crops")
fig.suptitle("At every cut-off, 'Gemini only' (CLIP discards, Gemini accepts) is the largest disagreement",
             x=0.01, ha="left", fontsize=11, color=INK)
ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(HERE / "results" / "fig_agreement_by_threshold.pdf"); plt.show()"""),

md("""## 3. Categories: one global threshold vs category-specific score ranges

Each dot is a category. x: its median CLIP score over all scored 2025.2 crops (not the sample,
whose bands are equal-sized by design). y: of its crops that CLIP discards at 0.27, the share Gemini
(crop + frame) accepts, population-weighted. Dot size: number of crops. Categories whose whole
score distribution sits low lose most of their crops at any cut-off. Gemini accepts fewer of those
discarded crops in low-median categories (Spearman ≈ 0.45), but the spread at any median is wide:
some low-median categories (e.g. toy, shirt, hair) are mostly correct below the cut-off."""),
code("""bc = cg.by_category(s)
y = bc["g_context_of_clip_rejected_0.27"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(bc.median_clip_pop, y, s=12 + 120 * np.sqrt(bc.n_population / bc.n_population.max()), color=BLUE,
           edgecolor="white", linewidth=1.2, alpha=0.9, zorder=3)
lab = pd.concat([bc.assign(y=y).nlargest(4, "n_population"), bc.assign(y=y).nsmallest(2, "y"),
                 bc.assign(y=y).nsmallest(2, "median_clip_pop"), bc.assign(y=y).nlargest(3, "median_clip_pop")])
lab = lab[~lab.index.duplicated()]
for i, (name, r) in enumerate(lab.iterrows()):
    ax.annotate(name, (r.median_clip_pop, r.y), xytext=(6, 6 if i % 2 else -10), textcoords="offset points",
                fontsize=8.5, color=INK2)
for t in (0.26, 0.27, 0.28):
    ax.axvline(t, color=INK2, lw=0.8, ls=":"); ax.text(t, 1.06, f"{t}", ha="center", fontsize=8, color=INK2)
ax.set_xlabel("category median CLIP score, all scored 2025.2 crops")
ax.set_ylabel("Gemini (crop + frame) accepts,\\nof crops CLIP discards at 0.27")
ax.set_ylim(-0.03, 1.1); ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
ax.set_title("Category medians span 0.23-0.30, so one cut-off discards very different shares per category",
             loc="left", fontsize=11, color=INK, pad=16)
fig.tight_layout(); fig.savefig(HERE / "results" / "fig_category_scatter.pdf"); plt.show()
print(f"category medians (population): {bc.median_clip_pop.min():.3f}-{bc.median_clip_pop.max():.3f}; "
      f"Spearman(median, y) = {bc.median_clip_pop.corr(y, method='spearman'):.2f}")
cols = ["n_population", "median_clip_pop", "pass_0.26_pop", "pass_0.27_pop", "pass_0.28_pop",
        "g_context_of_clip_rejected_0.27", "g_crop_below_0.26", "g_context_below_0.26"]
display(bc[cols].sort_values("g_context_of_clip_rejected_0.27", ascending=False).round(3).head(15))
display(bc[cols].sort_values("g_context_of_clip_rejected_0.27", ascending=False).round(3).tail(10))"""),

md("""## Montage helpers

Tiles are letterboxed crops with a caption: YOLOE label, CLIP score, Gemini's answer from the crop
alone and from crop + frame (✓ accepts, ✗ rejects). Items are picked deterministically, spread over categories (one per category
before any repeats). Montages are also saved to `out/private/montages/`."""),
code("""FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
MARK = {1.0: "✓", 0.0: "✗"}

def pick(d, n=30, seed=0):
    d = d.sample(frac=1, random_state=seed)
    d = d.assign(_k=d.groupby("class_name").cumcount()).sort_values(["_k"], kind="stable")
    return d.head(n).sort_values("clip_sim")

def fit(im, w, h, bg=(240, 239, 236)):
    im = im.copy(); im.thumbnail((w, h), Image.LANCZOS)
    c = Image.new("RGB", (w, h), bg); c.paste(im, ((w - im.width) // 2, (h - im.height) // 2)); return c

def tile(r, with_frame=False, W=170, H=170):
    crop = fit(Image.open(IMG / f"{r.item_id}_crop.jpg").convert("RGB"), W, H)
    if with_frame:
        fr = fit(Image.open(IMG / f"{r.item_id}_frame.jpg").convert("RGB"), int(H * 512 / 910), H)
        img = Image.new("RGB", (W + fr.width + 4, H), "white"); img.paste(crop, (0, 0)); img.paste(fr, (W + 4, 0))
    else:
        img = crop
    cap = Image.new("RGB", (img.width, 56), "white"); d = ImageDraw.Draw(cap)
    d.text((3, 2), r.class_name, font=FONT_B, fill=(11, 11, 11))
    d.text((3, 19), f"CLIP {r.clip_sim:.3f}", font=FONT, fill=(82, 81, 78))
    d.text((3, 36), f"Gemini crop {MARK.get(r.g_crop, '?')} · frame {MARK.get(r.g_context, '?')}", font=FONT,
           fill=(82, 81, 78))
    out = Image.new("RGB", (img.width, H + 56), "white"); out.paste(img, (0, 0)); out.paste(cap, (0, H)); return out

def montage(d, title, fname, cols=6, with_frame=False):
    tiles = [tile(r, with_frame) for r in d.itertuples()]
    if with_frame: cols = max(1, cols // 2 + 1)
    tw, th = tiles[0].size; g = 8; rows = -(-len(tiles) // cols)
    m = Image.new("RGB", (cols * (tw + g) + g, rows * (th + g) + g + 34), "white")
    ImageDraw.Draw(m).text((g, 8), title, font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17),
                           fill=(11, 11, 11))
    for i, t in enumerate(tiles):
        m.paste(t, (g + (i % cols) * (tw + g), 34 + g + (i // cols) * (th + g)))
    m.save(MONT / fname, quality=90); display(m)"""),

md(f"""## 4. Agreements and disagreements at CLIP ≥ 0.27 (Gemini crop condition)

Four cells of the 2×2. Within each montage tiles are sorted by CLIP score."""),
code("""gc = s.dropna(subset=["g_crop"])
cells = [
    ("A. Both accept (CLIP ≥ 0.27, Gemini ✓)", gc[gc.clip_keep & (gc.g_crop == 1)], "agree_both_accept.jpg"),
    ("B. Both reject (CLIP < 0.27, Gemini ✗)", gc[~gc.clip_keep & (gc.g_crop == 0)], "agree_both_reject.jpg"),
    ("C. CLIP keeps, Gemini rejects (CLIP ≥ 0.27, Gemini ✗)", gc[gc.clip_keep & (gc.g_crop == 0)], "disagree_clip_only.jpg"),
    ("D. CLIP discards, Gemini accepts (CLIP < 0.27, Gemini ✓)", gc[~gc.clip_keep & (gc.g_crop == 1)], "disagree_gemini_only.jpg"),
]
for title, d, f in cells:
    print(f"{title}: {len(d)} sample items")
    montage(pick(d), title, f)"""),

md("""## 5. The hardest disagreements: very low CLIP, Gemini accepts

CLIP < 0.24 (discarded at every cut-off we tried) yet Gemini accepts from the crop alone."""),
code("""d = gc[(gc.clip_sim < 0.24) & (gc.g_crop == 1)]
print(len(d), "items")
montage(pick(d, seed=1), "E. CLIP < 0.24 but Gemini accepts the crop", "gemini_only_very_low_clip.jpg")
d = gc[(gc.clip_sim >= 0.30) & (gc.g_crop == 0)]
print(len(d), "items")
montage(pick(d, seed=1), "F. CLIP ≥ 0.30 but Gemini rejects the crop", "clip_only_very_high_clip.jpg")"""),

md("""## 6. Does the frame change Gemini's answer?

Crop on the left, frame (magenta box) on the right. *Rescued*: rejected from the crop, accepted with the
frame. *Dropped*: the reverse. Rescues concentrate in low CLIP bands (13–14% of items below 0.26 vs 6% at
≥ 0.30), which is where missing context would hurt CLIP."""),
code("""both = s.dropna(subset=["g_crop", "g_context"])
resc = both[(both.g_crop == 0) & (both.g_context == 1)]
drop = both[(both.g_crop == 1) & (both.g_context == 0)]
print(f"rescued {len(resc)}, dropped {len(drop)} of {len(both)}")
display(pd.crosstab(both.band, [both.g_crop.map(MARK), both.g_context.map(MARK)], normalize="index")
        .reindex(cg.BAND_ORDER).round(3))
montage(pick(resc[resc.clip_sim < 0.26], n=18, seed=2), "G. Rescued by context (CLIP < 0.26): crop ✗ → frame ✓",
        "context_rescued.jpg", with_frame=True)
montage(pick(drop, n=12, seed=2), "H. Dropped with context: crop ✓ → frame ✗", "context_dropped.jpg", with_frame=True)"""),

md("""## 7. Category spotlights

Categories whose crops below 0.26 Gemini mostly accepts (top of the scatter in §3) and ones it mostly
rejects there."""),
code("""top = bc.head(4).index.tolist(); low = bc.dropna(subset=["g_context_below_0.26"]).tail(3).index.tolist()
print("mostly accepted below 0.26:", top, "| mostly rejected below 0.26:", low)
for c in top + low:
    d = s[(s.class_name == c) & (s.clip_sim < 0.26)].sort_values("clip_sim")
    montage(d.head(12), f"{c}: crops with CLIP < 0.26 (Gemini crop / context)", f"category_{c}.jpg")"""),

md("""## Caveats

* Gemini is a second model, not ground truth. Foils estimate leniency on unrelated words, but YOLOE's real
  errors are near misses (cup vs bowl), which are harder.
* `MEDIUM` media resolution (one 258-token tile per image) in both conditions; large crops are seen at
  lower detail than in the gold pilot.
* Body parts are included here (valid129); the 868 h study excluded them.
* Human labels (`out/private/human_labels.csv`) will add precision / recall per filter via
  `compare_clip_gemini.py` (`results/vs_human.csv`)."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "vislearnlabpy", "display_name": "Python (vislearnlabpy)", "language": "python"}})
nbf.write(nb, HERE / "clip_vs_gemini.ipynb")
print("wrote", HERE / "clip_vs_gemini.ipynb")
