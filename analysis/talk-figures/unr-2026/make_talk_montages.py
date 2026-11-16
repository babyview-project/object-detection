"""Private talk visuals (BabyView / VEDB frames; figures/private/montages/, git-ignored):

  cat_<a>_<b>.jpg    two categories, each a BabyView row above a VEDB row of crops (border colour = visual format)
  frames_<cat>.jpg   the same category in the full frame, BabyView vs VEDB, box outlined, boxes near each dataset's
                     median size for that category (typical, not hand-picked)
  glimpse.jpg        random frames from each dataset, one per video

Boxes from the Qwen3.8-27B stores (renamed labels). Frames in which Qwen found any person or face are skipped, and crops
must be at least MIN_SIDE px on their short side. Sampling is random (seeded), one box per video where possible.

    python make_talk_montages.py [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import re
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
Q = HERE.parents[1] / "qwen-open-vocab"
sys.path.insert(0, str(Q)); sys.path.insert(0, str(Q.parents[1] / "annotation" / "qwen_pilot"))
import os  # noqa: E402
os.environ.setdefault("QOV_LABELS", "renamed")
import variant as V  # noqa: E402
from build_open_means import basic_of  # noqa: E402
import talk_style as ts  # noqa: E402

OUT = ts.PRIVATE / "montages"
STORE = {"bv": Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1"),
         "vedb": Path("/data2/dataset/babyview/related_data/vedb_1612/qwen3.8-27b_setupB_t1_pixbv_adults")}
FRAMES = {"bv": lambda v, f: Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps") / v / f"{int(f):05d}.jpg",
          "vedb": lambda v, f: Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps_adults") / v / f"{int(f):08d}.jpg"}
NAME = {"bv": "BabyView (child)", "vedb": "VEDB (adult)"}
DSC = {"bv": (31, 92, 168), "vedb": (194, 81, 31)}   # fig_style.DS_COLOR (slate scheme)
FMT = {"real": (150, 150, 150), "toy": (126, 87, 194), "picture": (38, 166, 154), "screen": (249, 168, 37)}
PAIRS = [("dog", "cat"), ("bird", "duck"), ("bear", "fish"), ("car", "bus"), ("train", "truck"), ("airplane", "bicycle"),
         ("cup", "ball"), ("book", "bottle")]
FRAME_CATS = ["chair", "table"]
TILE, PER_ROW, LEFT, MIN_SIDE = 210, 8, 270, 48
BG = (251, 251, 248)
FONT = str(ts.HERE / "fonts" / "Lato-Bold.ttf") if (ts.HERE / "fonts" / "Lato-Bold.ttf").exists() else font_manager.findfont("DejaVu Sans")


def font(n):
    return ImageFont.truetype(FONT, n)


def load(key):
    ix = V.read_index(STORE[key], usecols=["video_id", "frame_number", "label", "format", "xmin", "ymin", "xmax", "ymax"])
    ix["label"] = ix.label.astype(str)
    b = basic_of(ix.label.unique()); ix["category"] = ix.label.map(b)
    dom = pd.read_csv(Q / "out/private/open_renamed/categories.csv", index_col=0).domain.to_dict()
    people = ix.label.str.contains(r"\b(?:person|face|man|woman|child|baby|girl|boy|adult|people|kid|toddler)\b", regex=True) | \
        (ix.category.map(dom) == "people")
    bad = set(map(tuple, ix.loc[people, ["video_id", "frame_number"]].drop_duplicates().values))
    ix = ix[[(v, f) not in bad for v, f in zip(ix.video_id, ix.frame_number)]]
    ix["fmt"] = ix.format.replace({"photo": "picture", "drawing": "picture"})
    ix["side"] = np.minimum(ix.xmax - ix.xmin, ix.ymax - ix.ymin)
    return ix


def tile(key, r, size=TILE, pad=0.12):
    im = Image.open(FRAMES[key](r.video_id, r.frame_number)).convert("RGB")
    # Uniform square tiles (as in the CCN deck): the square window slides to stay inside the frame and is capped at the
    # frame's short side, so no tile is padded with blank margins.
    s = min(max(r.xmax - r.xmin, r.ymax - r.ymin) * (1 + 2 * pad), im.width, im.height)
    cx, cy = (r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2
    L = int(min(max(cx - s / 2, 0), im.width - s)); T = int(min(max(cy - s / 2, 0), im.height - s))
    c = im.crop((L, T, L + int(s), T + int(s))).resize((size - 12, size - 12), Image.LANCZOS)
    t = Image.new("RGB", (size, size), FMT.get(r.fmt, FMT["real"])); t.paste(c, (6, 6))
    return t


def pick(d, n, rng):
    """Spread across children and videos: round-robin over children (BabyView subject id; VEDB has no participant id,
    so its unit is the session), and within a child one box per video before any video repeats."""
    d = d.sample(frac=1, random_state=int(rng.integers(1 << 30))).copy()
    d["unit"] = np.where(d.video_id.str.match(r"S\d{8}_"), d.video_id.str[:9], d.video_id)
    d["v_rank"] = d.groupby("video_id").cumcount()                       # 0 = first box of its video
    d = d.sort_values("v_rank", kind="stable")
    d["u_rank"] = d.groupby("unit").cumcount()                           # k-th box drawn from this child
    return d.sort_values(["u_rank", "v_rank"], kind="stable").head(n).drop(columns=["unit", "v_rank", "u_rank"])


def category_page(pair, data, rng):
    rows = []
    for cat in pair:
        for key in ("bv", "vedb"):
            g = data[key]; g = g[(g.category == cat) & (g.side >= MIN_SIDE)]
            ims = []
            for r in pick(g, PER_ROW * 3, rng).itertuples():
                if (str(r.video_id), str(r.frame_number)) in EXCLUDE:
                    continue
                try:
                    ims.append(tile(key, r))
                except Exception:
                    continue
                MANIFEST.append((PAGE, key, cat, len(ims) - 1, r.video_id, r.frame_number))
                if len(ims) == PER_ROW:
                    break
            rows.append((cat, key, ims))
    W = LEFT + PER_ROW * (TILE + 8) + 20; Hh = 40 + len(pair) * 58 + len(rows) * (TILE + 14)
    img = Image.new("RGB", (W, Hh), "white"); d = ImageDraw.Draw(img); y = 20
    for i, (cat, key, ims) in enumerate(rows):
        if key == "bv":
            d.text((20, y), cat, fill=(34, 34, 34), font=font(44)); y += 58
        d.text((20, y + TILE // 2 - 18), NAME[key].split(" ")[0], fill=DSC[key], font=font(34))
        for j, t in enumerate(ims):
            img.paste(t, (LEFT + j * (TILE + 8), y))
        if not ims:
            d.text((LEFT, y + TILE // 2 - 15), "none in VEDB", fill=(150, 150, 150), font=font(30))
        y += TILE + 14
    img = img.crop((0, 0, W, y + 10))
    lx = 20                                            # format legend
    leg = Image.new("RGB", (W, 60), "white"); dl = ImageDraw.Draw(leg)
    for f, col in FMT.items():
        dl.rectangle((lx, 18, lx + 34, 46), fill=col); dl.text((lx + 44, 14), f, fill=(34, 34, 34), font=font(30)); lx += 210
    full = Image.new("RGB", (W, img.height + 60), "white"); full.paste(img, (0, 0)); full.paste(leg, (0, img.height))
    full.save(OUT / f"cat_{pair[0]}_{pair[1]}.jpg", quality=88)


def outline(key, r, h):
    im = Image.open(FRAMES[key](r.video_id, r.frame_number)).convert("RGB")
    d = ImageDraw.Draw(im); lw = max(4, im.width // 120)
    d.rectangle((r.xmin, r.ymin, r.xmax, r.ymax), outline=(255, 214, 0), width=lw)
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


def frames_page(cat, data, rng, n=3, h=520):
    cols = []
    for key in ("bv", "vedb"):
        g = data[key]; g = g[(g.category == cat) & (g.fmt == "real")]
        area = (g.xmax - g.xmin) * (g.ymax - g.ymin) / (2048 * 1536 if key == "vedb" else 1)
        if key == "bv":                                       # BabyView frames vary in size: share of the frame
            area = area / (g.xmax.max() * g.ymax.max())
        lo, hi = area.quantile([.4, .6])
        g = g[(area >= lo) & (area <= hi)]
        ims = []
        for r in pick(g, n * 4, rng).itertuples():
            try:
                ims.append(outline(key, r, h))
            except Exception:
                continue
            if len(ims) == n:
                break
        cols.append((key, ims))
    W = max(sum(i.width for i in ims) + 20 * (len(ims) - 1) for _, ims in cols) + 40
    img = Image.new("RGB", (W, 2 * (h + 80) + 20), "white"); d = ImageDraw.Draw(img); y = 10
    for key, ims in cols:
        d.text((20, y), NAME[key], fill=DSC[key], font=font(40)); y += 60; x = 20
        for im in ims:
            img.paste(im, (x, y)); x += im.width + 20
        y += h + 20
    img.save(OUT / f"frames_{cat}.jpg", quality=88)


TEXTY = r"\b(?:paper|note|notes|sign|label|card|letter|text|poster|whiteboard|document|envelope|mail|calendar|book|screen|monitor|phone|laptop|computer|tablet|television)\b"


def glimpse(data, rng, n=4, h=430):
    rows = []
    for key in ("bv", "vedb"):
        g = data[key]
        texty = set(map(tuple, g.loc[g.label.str.contains(TEXTY, regex=True), ["video_id", "frame_number"]].drop_duplicates().values))
        fr = g[["video_id", "frame_number"]].drop_duplicates()
        fr = fr[[(v, f) not in texty for v, f in zip(fr.video_id, fr.frame_number)]]
        fr = fr.sample(frac=1, random_state=int(rng.integers(1 << 30))).drop_duplicates("video_id")
        ims = []
        for r in fr.itertuples():
            if (str(r.video_id), str(r.frame_number)) in EXCLUDE:
                continue
            try:
                im = Image.open(FRAMES[key](r.video_id, r.frame_number)).convert("RGB")
            except Exception:
                continue
            MANIFEST.append((PAGE, key, "frame", len(ims), r.video_id, r.frame_number))
            ims.append(im.resize((round(im.width * h / im.height), h), Image.LANCZOS))
            if len(ims) == (n + 1 if key == "bv" else n - 1):
                break
        rows.append((key, ims))
    W = max(sum(i.width for i in ims) + 16 * (len(ims) - 1) for _, ims in rows) + 40
    img = Image.new("RGB", (W, 2 * (h + 76) + 10), "white"); d = ImageDraw.Draw(img); y = 8
    for key, ims in rows:
        d.text((20, y), NAME[key], fill=DSC[key], font=font(40)); y += 58; x = 20
        for im in ims:
            img.paste(im, (x, y)); x += im.width + 16
        y += h + 18
    img.save(OUT / "glimpse.jpg", quality=88)


def scale_strip(data, rng, n=14, unit=1100, maxh=420):
    """Random real objects (any category) drawn at their size relative to the frame: side = sqrt(area share) x unit."""
    rows = []
    for key in ("bv", "vedb"):
        g = data[key]; g = g[(g.fmt == "real") & (g.side >= 24)]
        ims = []
        for r in pick(g, n * 3, rng).itertuples():
            try:
                im = Image.open(FRAMES[key](r.video_id, r.frame_number)).convert("RGB")
            except Exception:
                continue
            share = (r.xmax - r.xmin) * (r.ymax - r.ymin) / (im.width * im.height)
            c = im.crop((int(r.xmin), int(r.ymin), int(r.xmax), int(r.ymax)))
            k = np.sqrt(share) * unit / np.sqrt(c.width * c.height)
            c = c.resize((max(8, round(c.width * k)), max(8, round(c.height * k))), Image.LANCZOS)
            if c.height > maxh:
                c = c.resize((round(c.width * maxh / c.height), maxh), Image.LANCZOS)
            ims.append(c)
            if len(ims) == n:
                break
        rows.append((key, ims))
    W = 2100; img = Image.new("RGB", (W, 2 * (maxh + 90) + 20), "white"); d = ImageDraw.Draw(img); y = 10
    for key, ims in rows:
        d.text((20, y), NAME[key], fill=DSC[key], font=font(40)); y += 60; x = 20; base = y + maxh
        for c in ims:
            if x + c.width > W - 20:
                break
            img.paste(c, (x, base - c.height)); x += c.width + 14
        d.line((20, base + 4, W - 20, base + 4), fill=(200, 200, 200), width=3)
        y = base + 26
    img.save(OUT / "scale_strip.jpg", quality=88)


FRAME_AREA = {"bv": 512 * 910, "vedb": 2048 * 1536}
SIZE_CATS = ["cup", "spoon", "bowl", "book", "shoe", "chair", "car"]


def size_pairs(data, rng, cats=SIZE_CATS, unit=1250, maxh=470, gap=40):
    """One real object per dataset and category, near that dataset's median size, drawn at true relative scale
    (displayed side = sqrt(share of frame) x unit)."""
    cells = {}
    for cat in cats:
        for key in ("bv", "vedb"):
            g = data[key]; g = g[(g.category == cat) & (g.fmt == "real")]
            sh = (g.xmax - g.xmin) * (g.ymax - g.ymin) / FRAME_AREA[key]
            lo, hi = sh.quantile([.45, .55]); g = g[(sh >= lo) & (sh <= hi)]
            for r in pick(g, 10, rng).itertuples():
                try:
                    im = Image.open(FRAMES[key](r.video_id, r.frame_number)).convert("RGB")
                except Exception:
                    continue
                c = im.crop((int(r.xmin), int(r.ymin), int(r.xmax), int(r.ymax)))
                share = c.width * c.height / FRAME_AREA[key]
                k = np.sqrt(share) * unit / np.sqrt(c.width * c.height)
                c = c.resize((max(6, round(c.width * k)), max(6, round(c.height * k))), Image.LANCZOS)
                if c.height > maxh:
                    c = c.resize((round(c.width * maxh / c.height), maxh), Image.LANCZOS)
                cells[cat, key] = (c, share); break
    colw = [max(cells[c, k][0].width for k in ("bv", "vedb") if (c, k) in cells) for c in cats]
    W = 230 + sum(colw) + gap * len(cats); H = 2 * (maxh + 70) + 90
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    for row, key in enumerate(("bv", "vedb")):
        base = 60 + (row + 1) * (maxh + 70) - 40
        d.text((20, base - maxh // 2 - 20), NAME[key].split(" ")[0], fill=DSC[key], font=font(40))
        x = 230
        for cat, w in zip(cats, colw):
            if (cat, key) in cells:
                c, share = cells[cat, key]
                img.paste(c, (x + (w - c.width) // 2, base - c.height))
                d.text((x + w // 2, base + 8), f"{share * 100:.1f}%", fill=(90, 90, 90), font=font(26), anchor="mt")
            if row == 0:
                d.text((x + w // 2, 20), cat, fill=(34, 34, 34), font=font(40), anchor="mt")
            x += w + gap
    img.save(OUT / "size_pairs.jpg", quality=90)


GRID_PAIRS = [("duck", "bird"), ("bus", "truck"), ("car", "tractor"), ("sun", "fire hydrant")]


def grid5_page(pair, data, rng, n=5, t=150, g=3):
    """CCN-deck style: per category a 5 x 5 tile grid for BabyView and one for VEDB, category name in its domain colour."""
    dom = pd.read_csv(Q / "out/private/open_renamed/categories.csv", index_col=0).domain.to_dict()
    sys.path.insert(0, str(Q)); import fig_style as FS
    gw = n * t + (n - 1) * g; colgap, left, top = 70, 30, 20
    W = left + 2 * gw + colgap + 30; H = top + len(pair) * (gw + 120) + 10
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img); y = top
    for cat in pair:
        col = tuple(int(FS.DOMAIN_COLOR.get(dom.get(cat, "other"), "#8B9A9E")[i:i + 2], 16) for i in (1, 3, 5))
        d.text((left, y), cat, fill=col, font=font(52)); y += 70
        for j, key in enumerate(("bv", "vedb")):
            x0 = left + j * (gw + colgap)
            d.text((x0, y), NAME[key], fill=DSC[key], font=font(30))
            gg = data[key]; gg = gg[(gg.category == cat) & (gg.side >= MIN_SIDE)]
            for k, tl in enumerate(_tiles(pick(gg, n * n * 4, rng), t, key=key, n=n * n)):
                img.paste(tl, (x0 + (k % n) * (t + g), y + 44 + (k // n) * (t + g)))
        y += gw + 50
    img.save(OUT / f"grid_{pair[0].replace(' ', '_')}_{pair[1].replace(' ', '_')}.jpg", quality=90)


def _dom_rgb(cat):
    dom = pd.read_csv(Q / "out/private/open_renamed/categories.csv", index_col=0).domain.to_dict()
    sys.path.insert(0, str(Q)); import fig_style as FS
    h = FS.DOMAIN_COLOR.get(dom.get(cat, "other"), "#8B9A9E")
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _grid(rows, n, t=120, g=3, cols=None):
    """rows: list of PIL tiles (already t x t); returns an n-column grid image."""
    cols = cols or n; k = len(rows); r_ = -(-k // cols)
    if k == 0:
        return None
    img = Image.new("RGB", (cols * t + (cols - 1) * g, r_ * t + (r_ - 1) * g), "white")
    for i, tl in enumerate(rows):
        img.paste(tl, ((i % cols) * (t + g), (i // cols) * (t + g)))
    return img


def _dhash(im, k=12):
    """Difference hash of a tile (k x k bits): near-identical crops have hashes a few bits apart."""
    g = np.asarray(im.convert("L").resize((k + 1, k), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).ravel()


_DINO = {}


def _embed(im):
    """DINOv3 ViT-B/16 global embedding of a tile (CPU), L2-normalized; loaded once."""
    if not _DINO:
        os.environ.setdefault("HF_HOME", "/ccn2/u/khaiaw/.cache/huggingface"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoImageProcessor, AutoModel
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        _DINO["p"] = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        _DINO["m"] = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").eval(); _DINO["t"] = torch
    with _DINO["t"].inference_mode():
        v = _DINO["m"](**_DINO["p"](images=[im], return_tensors="pt")).pooler_output[0].numpy()
    return v / np.linalg.norm(v)


# Manual privacy review: tiles a reviewer flagged (people Qwen gave an object label, e.g. a child boxed as "dog";
# automatic person checks fire on toys and cartoons at this crop size). Rows: video_id,frame_number. Kept private.
EXCLUDE_CSV = OUT / "exclude_frames.csv"
EXCLUDE = (set(map(tuple, pd.read_csv(EXCLUDE_CSV).astype(str).values)) if EXCLUDE_CSV.exists() else set())
PAGE = ""
MANIFEST = []   # (page, key, category, tile index, video_id, frame_number) per drawn tile; written to OUT/manifest.csv


def _tiles(rows, t, key="bv", pad=0.08, n=None, max_bits=30, max_cos=None):
    """Tiles chosen for spread, not just at random: the same plush, print or toy recurs across sessions and homes, and
    its DINOv3 similarity shifts with the background, so a fixed cut-off misses it. Greedy max-min instead: each pick is
    the candidate least similar to the tiles already kept, with small penalties for a child (VEDB: session) or video
    already shown and for late draw order; near-blank crops are dropped. max_bits=-1 keeps draw order (viewpoint)."""
    cand = []
    for r in rows.itertuples():
        if (str(r.video_id), str(r.frame_number)) in EXCLUDE:
            continue
        try:
            tl = tile(key, r, size=t + 12, pad=pad).crop((6, 6, t + 6, t + 6))
        except Exception:
            continue
        if max_bits < 0:
            L_ = np.asarray(tl.convert("L"), dtype=float)
            if L_.mean() < 45 or L_.std() < 10:                    # dark or blank moments (camera covered, lights off)
                continue
            cand.append(tl); MANIFEST.append((PAGE, key, r.category, len(cand) - 1, r.video_id, r.frame_number))
            if n and len(cand) == n:
                break
            continue
        if np.asarray(tl.convert("L"), dtype=float).std() < 10:
            continue
        u = r.video_id[:9] if re.match(r"S\d{8}_", r.video_id) else r.video_id
        cand.append((tl, u, r.video_id, r.frame_number, r.category))
    if max_bits < 0:
        return cand
    n = n or len(cand)
    if len(cand) <= n:
        keep = list(range(len(cand)))
    else:
        keep = None
    if keep is None:
        keep = _maxmin(cand, n)
    for j, i in enumerate(keep):
        MANIFEST.append((PAGE, key, cand[i][4], j, cand[i][2], cand[i][3]))
    return [cand[i][0] for i in keep]


def _maxmin(cand, n, atypical=0.25):
    """Greedy max-min over DINOv3 embeddings. Max-min favours outliers, which are often mislabels or people boxed as
    objects, so the least typical quarter (lowest cosine to the category mean) is set aside first."""
    E = np.stack([_embed(c[0]) for c in cand]); S = E @ E.T
    order = np.arange(len(cand)) / len(cand)
    typ = E @ (E.mean(0) / np.linalg.norm(E.mean(0)))
    near = np.where(typ < np.quantile(typ, atypical) if len(cand) * (1 - atypical) >= n else False, np.inf, -1.0); taken = np.zeros(len(cand), bool); units, vids = {}, set(); keep = []
    for _ in range(n):
        pen = np.array([0.15 * units.get(u, 0) + 0.1 * (v in vids) for _, u, v, *_ in cand])
        score = np.where(taken, np.inf, near + pen + 0.1 * order)
        i = int(np.argmin(score)); taken[i] = True; keep.append(i)
        near = np.maximum(near, S[i]); units[cand[i][1]] = units.get(cand[i][1], 0) + 1; vids.add(cand[i][2])
    return sorted(keep)


def bv_grid7(cats, data, rng, n=7, t=128, prefix="bv7", pool=4):
    """BabyView only: one n x n grid per category (random boxes, one per video first), two categories per image."""
    b = data["bv"]
    for i in range(0, len(cats), 2):
        pair = cats[i:i + 2]; panels = []
        for cat in pair:
            gg = b[(b.category == cat) & (b.side >= MIN_SIDE)]
            panels.append((cat, _grid(_tiles(pick(gg, n * n * pool, rng), t, n=n * n), n, t)))
        W = sum(p_.width for _, p_ in panels) + 60 * (len(panels) - 1) + 40; H = max(p_.height for _, p_ in panels) + 90
        img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img); x = 20
        for cat, p_ in panels:
            d.text((x, 10), cat, fill=_dom_rgb(cat), font=font(56)); img.paste(p_, (x, 84)); x += p_.width + 60
        img.save(OUT / f"{prefix}_{'_'.join(c.replace(' ', '_') for c in pair)}.jpg", quality=90)


def _panel_page(panels, name, ncol=4, title_px=46):
    """panels: list of (cat, grid image); laid out ncol per row with category names in domain colours."""
    panels = [(c, p_) for c, p_ in panels if p_ is not None]
    gw = max(p_.width for _, p_ in panels); gh = max(p_.height for _, p_ in panels)
    nrow = -(-len(panels) // ncol); W = ncol * gw + (ncol - 1) * 50 + 40; H = nrow * (gh + title_px + 40) + 20
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    for i, (cat, p_) in enumerate(panels):
        x = 20 + (i % ncol) * (gw + 50); y = 10 + (i // ncol) * (gh + title_px + 40)
        d.text((x, y), cat, fill=_dom_rgb(cat), font=font(title_px)); img.paste(p_, (x, y + title_px + 12))
    img.save(OUT / f"{name}.jpg", quality=90)


def var_format(cats, data, rng, n=5, t=110):
    """Variation across visual format: per animal category a 5 x 5 grid sampled round-robin over the formats present."""
    b = data["bv"]; panels = []
    for cat in cats:
        gg = b[(b.category == cat) & (b.side >= MIN_SIDE)]
        by = {f: pick(x, n * n, rng) for f, x in gg.groupby("fmt") if len(x) >= 3}
        order = []
        for k in range(n * n):
            for f in ("real", "toy", "picture", "screen"):
                if f in by and k < len(by[f]):
                    order.append(by[f].iloc[k])
        rows = pd.DataFrame(order[: n * n * 4])
        panels.append((cat, _grid(_tiles(rows, t, n=n * n), n, t)))
    _panel_page(panels, "var_format")


def var_exemplar(cats, data, rng, n=4, t=130):
    """Variation across exemplars: one box per video, spread across children (likely different objects)."""
    b = data["bv"]; panels = []
    for cat in cats:
        gg = b[(b.category == cat) & (b.side >= 64) & (b.fmt == "real")].copy()
        panels.append((cat, _grid(_tiles(pick(gg, n * n * 4, rng), t, n=n * n), n, t)))
    _panel_page(panels, "var_exemplar")


def var_viewpoint(cats, data, rng, n=4, t=130):
    """Variation across viewpoints: n x n moments from ONE video (likely the same object), spread over the video."""
    b = data["bv"]; panels = []
    for cat in cats:
        gg = b[(b.category == cat) & (b.side >= 64) & (b.fmt == "real")]
        cnt = gg.groupby("video_id").frame_number.nunique()
        if cnt.empty or cnt.max() < n * n:
            continue
        ok = gg[gg.video_id.isin(cnt[cnt >= n * n].index)].copy()          # prefer videos where the object's size and
        ok["cx"] = (ok.xmin + ok.xmax) / 2 / 512; ok["ls"] = np.log(ok.side)  # position in frame vary most (more viewpoints)
        score = ok.groupby("video_id").agg(a=("ls", "std"), b=("cx", "std")).sum(1).sort_values(ascending=False)
        vid = score.head(5).sample(1, random_state=int(rng.integers(1 << 30))).index[0]
        v = gg[gg.video_id == vid].sort_values("side", ascending=False).drop_duplicates("frame_number").sort_values("frame_number")
        idx = np.unique(np.linspace(0, len(v) - 1, min(len(v), n * n * 3)).round().astype(int))
        rows = v.iloc[idx]; rows = rows.iloc[np.argsort(np.arange(len(rows)) % 3, kind="stable")]  # spread first, extras after
        tl = _tiles(rows, t, pad=0.1, n=n * n, max_bits=-1)
        panels.append((cat, _grid(tl, n, t)))
    _panel_page(panels, "var_viewpoint")


THINGS_IMG = Path("/data2/dataset/babyview/related_data/things/object_images")


IMAGENET_IMG = Path("/data2/j7yang/imagenet-1k/extracted/train")
CD_MAP = Q.parents[0] / "comparison-datasets" / "outputs" / "mapping"
OTHER_NAME = {"things": "THINGS", "imagenet": "ImageNet", "vedb": "VEDB (adult)", "lvis": "LVIS"}
OTHER_RGB = {"things": (59, 125, 196), "imagenet": (125, 98, 184), "vedb": (224, 123, 57), "lvis": (46, 150, 110)}
_IMNET, _LVIS = {}, {}


def _square(im, t, box=None, pad=0.08):
    """Square t x t tile: the box (or the whole image) padded, the square window kept inside the image."""
    W_, H_ = im.size
    if box is None:
        x0, y0, x1, y1 = 0, 0, W_, H_; s_ = min(W_, H_)
    else:
        x0, y0, x1, y1 = box; s_ = min(max(x1 - x0, y1 - y0) * (1 + 2 * pad), W_, H_)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L = int(min(max(cx - s_ / 2, 0), W_ - s_)); T = int(min(max(cy - s_ / 2, 0), H_ - s_))
    return im.crop((L, T, L + int(s_), T + int(s_))).resize((t, t), Image.LANCZOS)


def _other_images(other, cat, k, rng, t):
    """k images of category cat from a comparison dataset, as t x t tiles."""
    if other == "things":
        ids = pd.read_csv(Q / "mapping/map_things.csv").query("category == @cat").id
        ps = sorted(p_ for c_ in ids for p_ in (THINGS_IMG / c_).glob("*.jpg"))
        return [_square(Image.open(p_).convert("RGB"), t) for p_ in (rng.choice(ps, min(k, len(ps)), replace=False) if ps else [])]
    if other == "imagenet":
        if not _IMNET:
            for f in os.listdir(IMAGENET_IMG):
                _IMNET.setdefault(f.split("_", 1)[0], []).append(IMAGENET_IMG / f)
        ps = [p_ for w in pd.read_csv(Q / "mapping/map_imagenet.csv").query("category == @cat").id for p_ in _IMNET.get(w, [])[:400]]
        return [_square(Image.open(p_).convert("RGB"), t) for p_ in (rng.choice(ps, min(k, len(ps)), replace=False) if ps else [])]
    if other == "lvis":
        if not _LVIS:
            old = pd.read_csv(CD_MAP / "lvis_crop_manifest.csv").merge(pd.read_csv(CD_MAP / "lvis_crop_metadata.csv.gz",
                  usecols=["crop_id", "category_id"]), left_on="image_id", right_on="crop_id")
            c0 = old[["category_id", "path", "xmin", "ymin", "xmax", "ymax"]]
            ex = np.load(Q / "out/crossdataset/lvis_extra_crops_sample.npz", allow_pickle=True)     # categories the
            b_ = ex["bbox"]                                                                       # sample lacks
            c1 = pd.DataFrame({"category_id": ex["category_id"], "path": ex["path"], "xmin": b_[:, 0], "ymin": b_[:, 1],
                               "xmax": b_[:, 2], "ymax": b_[:, 3]})
            _LVIS["c"] = pd.concat([c0, c1], ignore_index=True)
        ids = set(pd.read_csv(Q / "mapping/map_lvis.csv").query("category == @cat").id)
        c = _LVIS["c"][_LVIS["c"].category_id.isin(ids)]
        c = c[(c.xmax - c.xmin) >= 48]
        c = c.sample(frac=1, random_state=int(rng.integers(1 << 30))).drop_duplicates("path").head(k)
        return [_square(Image.open(r.path).convert("RGB"), t, (r.xmin, r.ymin, r.xmax, r.ymax)) for r in c.itertuples()]
    raise ValueError(other)


STRATEGY = {"things": (("basketball", "washing machine", "banana", "fork", "suitcase"), ("car", "bird", "monkey", "plant", "magnet")),
            "imagenet": (("washing machine", "guitar", "banana", "teapot", "teddy bear"), ("monkey", "bird", "duck", "car", "box")),
            "vedb": (("box", "television", "fork", "pillow", "armchair"), ("duck", "train", "boat", "car", "truck")),
            "lvis": (("basketball", "guitar", "fork", "pot", "rabbit"), ("duck", "bear", "train", "car", "bus"))}


def strategy_page(data, rng, other="things", cols=5, rows=2, t=84, g=3):
    """CCN-deck layout (CCN slide 30): similar and dissimilar categories, BabyView vs one comparison dataset (DINOv3
    cosine of the category means, both z-scored over the shared categories); per category a cols x rows block of
    BabyView crops next to the same of the comparison dataset."""
    similar, dissimilar = STRATEGY[other]
    if other == "things":
        cos = pd.read_csv(Q / "results/open_renamed/open_categorywise_bv_things.csv").query("model == 'dinov3'")
    else:
        cos = pd.read_csv(Q / f"results/crossdataset_renamed_core/categorywise_bv_vs_{other}.csv")
    cos = cos.set_index("category").cosine
    k = cols * rows
    b = data["bv"]; strip = cols * t + (cols - 1) * g; colw = 2 * strip + 16
    blockh = rows * t + (rows - 1) * g
    W = 40 + 2 * colw + 70; rowh = blockh + 36; H = 92 + 5 * rowh + 10
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    for j, (head, cats, col) in enumerate((("Similar categories", similar, (76, 160, 110)),
                                           ("Dissimilar categories", dissimilar, (140, 110, 200)))):
        x0 = 20 + j * (colw + 70)
        d.text((x0, 8), head, fill=col, font=font(34))
        d.text((x0 + strip // 2, 70), "BabyView", fill=DSC["bv"], font=font(18), anchor="mm")
        d.text((x0 + strip + 16 + strip // 2, 70), OTHER_NAME[other], fill=OTHER_RGB[other], font=font(18), anchor="mm")
        for i, cat in enumerate(cats):
            y = 92 + i * rowh
            d.text((x0 + colw // 2, y + 2), f"{cat}, sim = {cos[cat]:.2f}", fill=_dom_rgb(cat), font=font(20), anchor="mt")
            gg = b[(b.category == cat) & (b.side >= MIN_SIDE)]
            for q, tl in enumerate(_tiles(pick(gg, k * 5, rng), t, n=k)):
                img.paste(tl, (x0 + (q % cols) * (t + g), y + 28 + (q // cols) * (t + g)))
            if other == "vedb":
                gv = data["vedb"]; gv = gv[(gv.category == cat) & (gv.side >= MIN_SIDE)]
                oth = _tiles(pick(gv, k * 5, rng), t, key="vedb", n=k)
            else:
                oth = _other_images(other, cat, k, rng, t)
            for q, tl in enumerate(oth):
                img.paste(tl, (x0 + strip + 16 + (q % cols) * (t + g), y + 28 + (q // cols) * (t + g)))
    img.save(OUT / ("strategy_similar_dissimilar.jpg" if other == "things" else f"strategy_similar_dissimilar_{other}.jpg"), quality=90)


def pipeline_page(data, rng, video_id="S00320003_2025-05-10_2_recGiSPRvzAcCp0SN", frame_number=0, cat="book", t=92, g=3, name="pipeline_crop_embed_average"):
    """CCN-deck layout (CCN slide 26): one Qwen-annotated frame (boxes coloured by format, the category's box in yellow)
    -> crops of that category (this frame's first, then other children) -> DINOv3 -> one vector per crop, averaged."""
    b = data["bv"]
    fb = b[(b.video_id == video_id) & (b.frame_number == frame_number)]
    fr = Image.open(FRAMES["bv"](video_id, frame_number)).convert("RGB"); sc = 600 / fr.height
    fr = fr.resize((int(fr.width * sc), 600), Image.LANCZOS)
    cv = Image.new("RGB", fr.size, "white"); cv.paste(fr); fr = cv; dr = ImageDraw.Draw(fr)   # canvas: labels stay editable
    COL = {"real": (46, 158, 107), "toy": (240, 160, 48), "picture": (217, 123, 158), "screen": (91, 155, 213)}
    for r in sorted(fb.itertuples(), key=lambda r: r.category == cat):
        hi = r.category == cat; c = (240, 200, 40) if hi else COL.get(r.fmt, COL["real"])
        box = [r.xmin * sc, r.ymin * sc, r.xmax * sc, r.ymax * sc]
        dr.rectangle(box, outline=c, width=5 if hi else 2)
        lab = f"{r.category} · {r.fmt}"; f_ = font(15 if hi else 12); w_ = dr.textlength(lab, font=f_)
        y_ = max(0, box[1] - (19 if hi else 15))
        dr.rectangle((box[0], y_, box[0] + w_ + 6, y_ + (19 if hi else 15)), fill=c)
        dr.text((box[0] + 3, y_), lab, fill="white" if not hi else (30, 30, 30), font=f_)
    gg = b[(b.category == cat) & (b.side >= MIN_SIDE) & (b.video_id != video_id)]
    own = fb[fb.category == cat].sort_values("side", ascending=False).head(1)
    tiles = _tiles(own, t, n=1, max_bits=-1) + _tiles(pick(gg, 60, rng), t, n=8)
    grid = _grid(tiles, 3, t, g)
    W = fr.width + 70 + grid.width + 60 + 220 + 40; H = 620
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    img.paste(fr, (10, 10))
    gx = fr.width + 80; gy = 60; img.paste(grid, (gx, gy))
    d.rectangle((gx - 2, gy - 2, gx + t + 1, gy + t + 1), outline=(240, 200, 40), width=4)
    d.text((gx + grid.width // 2, gy - 12), f"every '{cat}' crop", fill=(40, 40, 40), font=font(22), anchor="mb")
    ay = gy + grid.height // 2
    d.line((fr.width + 18, ay, gx - 12, ay), fill=(240, 200, 40), width=7)
    d.polygon([(gx - 10, ay), (gx - 30, ay - 14), (gx - 30, ay + 14)], fill=(240, 200, 40))
    bx = gx + grid.width + 60; by = gy + 20
    d.rounded_rectangle((bx, by, bx + 200, by + 110), radius=12, fill=(20, 30, 70))
    d.text((bx + 100, by + 40), "DINOv3", fill="white", font=font(32), anchor="mm")
    d.text((bx + 100, by + 80), "self-supervised", fill=(190, 200, 230), font=font(18), anchor="mm")
    d.line((gx + grid.width + 8, ay - 40, bx - 8, by + 55), fill=(60, 170, 110), width=5)
    vy = by + 200; cols = [(13, 71, 161), (77, 184, 168), (232, 165, 76), (155, 126, 200), (107, 171, 122),
                           (217, 123, 158), (91, 155, 213), (232, 122, 95)]
    d.line((bx + 100, by + 115, bx + 100, vy - 30), fill=(60, 170, 110), width=5)
    d.polygon([(bx + 100, vy - 14), (bx + 88, vy - 34), (bx + 112, vy - 34)], fill=(60, 170, 110))
    for q, c in enumerate(cols):
        d.rectangle((bx + q * 24, vy, bx + q * 24 + 20, vy + 20), fill=c)
    d.text((bx + 8 * 24 + 4, vy + 10), "…", fill=(40, 40, 40), font=font(26), anchor="lm")
    d.text((bx, vy - 6), "1", fill=(40, 40, 40), font=font(16), anchor="lb")
    d.text((bx + 200, vy - 6), "768", fill=(40, 40, 40), font=font(16), anchor="rb")
    d.text((bx + 100, vy + 52), "mean per category", fill=(40, 40, 40), font=font(20), anchor="mm")
    img.save(OUT / f"{name}.jpg", quality=92)


def format_grids(data, rng, key="bv", n=10, t=92, formats=("real", "toy", "drawing", "photo", "screen")):
    """One n x n page per Qwen format label (raw: real / toy / drawing / photo / screen): random boxes of any category,
    spread across children and videos (round-robin), near-duplicates skipped, body parts left out."""
    b = data[key]; dom = pd.read_csv(Q / "out/private/open_renamed/categories.csv", index_col=0).domain
    b = b[(b.side >= MIN_SIDE) & (b.category.map(dom) != "body_parts")]
    share = data[key].format.value_counts(normalize=True)
    for f in formats:
        gg = b[b.format == f]
        tl = _tiles(pick(gg, n * n * 3, rng), t, key=key, n=n * n)
        g = _grid(tl, n, t)
        img = Image.new("RGB", (g.width + 40, g.height + 100), "white"); d = ImageDraw.Draw(img)
        d.text((20, 12), f"{f}", fill=(40, 40, 40), font=font(52))
        d.text((20 + d.textlength(f, font=font(52)) + 24, 34), f"{share.get(f, 0):.1%} of {NAME[key]} boxes",
               fill=(110, 110, 110), font=font(26))
        img.paste(g, (20, 90))
        img.save(OUT / f"format10_{key}_{f}.jpg", quality=90)



def _maxmin_hash(cand, n):
    """Greedy max-min over difference hashes (no model needed): each pick is the candidate whose closest kept tile is
    farthest in Hamming distance, so near-duplicate prints and toys are not shown twice. Starts from the first draw."""
    H = np.stack([_dhash(c[0]) for c in cand]); keep = [0]
    d = (H != H[0]).sum(1).astype(float)
    while len(keep) < n:
        d[keep] = -1; i = int(np.argmax(d)); keep.append(i); d = np.minimum(d, (H != H[i]).sum(1))
    return sorted(keep)


def most_depicted(data, rng, n_cat=8, n_bv=6, n_vedb=3, t=150, min_bv=1000, min_vedb=50, min_dep=.95):
    """Manuscript Fig. 3B: categories BabyView sees almost only as depictions while adults see them real. One row per
    category: among categories >= min_dep depicted in BabyView with >= min_bv BabyView and >= min_vedb VEDB boxes
    (people left out), the n_cat with the largest BabyView - VEDB gap in % depicted; name in its domain colour, %
    depicted in each dataset, then n_bv BabyView crops of depicted boxes and n_vedb VEDB crops of any format. Tile
    border = Qwen's format. Crops are random draws spread over children and videos (one box per child, then per video,
    before any repeats) and pass the same privacy filters as the
    other pages (no frame with a person or face; manual exclusion list)."""
    global PAGE
    PAGE = "most_depicted"
    g = pd.read_csv(Q / "results/crossdataset_renamed/depiction_gap_bv_vs_vedb.csv")
    g = g[(g.bv_n >= min_bv) & (g.vedb_n >= min_vedb) & (g.domain != "people") & (g.bv_depicted >= min_dep)]
    g = g.sort_values("gap", ascending=False).head(n_cat)

    def strip(key, cat, n, depicted_only):
        d = data[key]; d = d[(d.category == cat) & (d.side >= MIN_SIDE)]
        if depicted_only:
            d = d[d.fmt != "real"]
        cand = []
        for r in pick(d, n * 5, rng).itertuples():
            if (str(r.video_id), str(r.frame_number)) in EXCLUDE:
                continue
            try:
                tl = tile(key, r, size=t)
            except Exception:
                continue
            L_ = np.asarray(tl.convert("L"), dtype=float)
            if L_.std() < 10 or L_.mean() < 55:                          # blank or dark (night, camera covered)
                continue
            u = r.video_id[:9] if re.match(r"S\d{8}_", r.video_id) else r.video_id
            cand.append((tl, u, r.video_id, r.frame_number, r.category))
        keep = range(min(n, len(cand)))                                  # draw order: spread over children / videos
        for j, i in enumerate(keep):
            MANIFEST.append((PAGE, key, cat, j, cand[i][2], cand[i][3]))
        return [cand[i][0] for i in keep]

    LEFT_W, GAP, G, ROW, HEAD = 330, 34, 4, t + 16, 92
    W = LEFT_W + n_bv * (t + G) + GAP + n_vedb * (t + G) + 20
    H = HEAD + len(g) * ROW + 70
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    x_bv, x_v = LEFT_W, LEFT_W + n_bv * (t + G) + GAP
    d.text((x_bv, 20), "BabyView (child): depicted examples", fill=DSC["bv"], font=font(30))
    d.text((x_v, 20), "VEDB (adult)", fill=DSC["vedb"], font=font(30))
    for k, r in enumerate(g.itertuples()):
        y = HEAD + k * ROW
        d.text((16, y + 18), r.category, fill=_dom_rgb(r.category), font=font(38))
        d.text((16, y + 70), f"child: {r.bv_depicted:.0%} depicted", fill=DSC["bv"], font=font(24))
        d.text((16, y + 100), f"adult: {r.vedb_depicted:.0%} depicted", fill=DSC["vedb"], font=font(24))
        for j, tl in enumerate(strip("bv", r.category, n_bv, True)):
            img.paste(tl, (x_bv + j * (t + G), y))
        for j, tl in enumerate(strip("vedb", r.category, n_vedb, False)):
            img.paste(tl, (x_v + j * (t + G), y))
    x = 16; yl = H - 50                                              # format legend (tile border colours)
    d.text((x, yl), "tile border = format:", fill=(80, 80, 80), font=font(26)); x += 270
    for f in ("toy", "picture", "screen", "real"):
        d.rectangle((x, yl + 4, x + 26, yl + 30), fill=FMT[f]); d.text((x + 36, yl), f, fill=(60, 60, 60), font=font(26)); x += 160
    img.save(OUT / "most_depicted.jpg", quality=92)
    pd.DataFrame([m for m in MANIFEST if m[0] == PAGE], columns=["page", "dataset", "category", "tile", "video_id", "frame_number"]) \
        .to_csv(OUT / "manifest_most_depicted.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: load(k) for k in ("bv", "vedb")}
    print({k: len(v) for k, v in data.items()}, flush=True)
    rng = np.random.default_rng(a.seed)
    if not a.only or "cats" in a.only:
        for p in PAIRS:
            category_page(p, data, rng); print("cat", p, flush=True)
    if not a.only or "frames" in a.only:
        for c in FRAME_CATS:
            frames_page(c, data, rng); print("frames", c, flush=True)
    if not a.only or "glimpse" in a.only:
        glimpse(data, rng); print("glimpse", flush=True)
    if not a.only or "scale" in a.only:
        scale_strip(data, rng); print("scale", flush=True)
    if not a.only or "grid" in a.only:
        for p_ in GRID_PAIRS:
            grid5_page(p_, data, rng); print("grid", p_, flush=True)
    if not a.only or "size" in a.only:
        size_pairs(data, rng); print("size", flush=True)
    if a.only and "most_depicted" in a.only:
        most_depicted(data, rng); print("most_depicted", flush=True)
