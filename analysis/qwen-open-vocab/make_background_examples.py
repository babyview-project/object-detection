#!/usr/bin/env python3
"""Examples for the background control (Reviewer 1, Response Figures 4 and 5; SI 5). Private: BabyView / VEDB frames.

  R1d_examples_lvis_crop_vs_whole.jpg   LVIS: the whole image (object outlined) next to the crop, one category per domain
  R1j_examples_sam3.jpg                 BabyView and VEDB: Qwen's box | SAM 3's box | SAM 3's box, background removed;
                                        random objects among those used in the analysis, one per domain
  R1j_examples_sam3_quality.jpg         random objects per SAM 3 score band (how good are the masks?), BabyView and VEDB

Objects come from the analysis sample (out/private/bv_vedb_renamed/sample_dinov3.npz) and the SAM 3 outputs
(annotation/qwen_pilot/sam3_crop_sample.py); background removal is the same as in embed_masked_sample.py (grey 128
outside the mask). Frames in which Qwen found a person or face are skipped (as in the talk montages). Seeded.

    python make_background_examples.py      # -> manuscript-2026/revision_private/figures/
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
TALK = HERE.parent / "talk-figures" / "unr-2026"
sys.path.insert(0, str(TALK))
import make_talk_montages as MT  # noqa: E402  (fonts, people filter, LVIS crop table)

OUT = HERE.parent / "manuscript-2026" / "revision_private" / "figures"
SAM = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/sam3_crop_sample_bv_vedb_adults")
SAMPLE = HERE / "out" / "private" / "bv_vedb_renamed" / "sample_dinov3.npz"
CORE = {l.strip() for l in open(HERE.parents[1] / "data" / "included_categories_qwen_core.txt") if l.strip()}
DOM = pd.read_csv(HERE / "out/private/open_renamed/categories.csv", index_col=0).domain.to_dict()
DOMS = ["animals", "vehicles", "food_drink", "toys", "furniture_rooms", "household", "clothing", "outside"]
DLAB = {"animals": "animal", "vehicles": "vehicle", "food_drink": "food", "toys": "toy", "furniture_rooms": "furniture",
        "household": "household", "clothing": "clothing", "outside": "outside"}
GREY, T, G = 128, 150, 4
INK, FAINT = (40, 40, 40), (120, 120, 120)
DSC = {"bv": (13, 71, 161), "vedb": (224, 123, 57)}
NAME = {"bv": "BabyView (child)", "vedb": "VEDB (adult)"}


def sam():
    parts = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(str(SAM / "w*_*.npz")))]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    S = {k: cat(k) for k in ("idx", "score", "box", "crop", "mask_shape")}
    S["mask"] = np.concatenate([p["mask"] for p in parts])
    e = np.load(SAM / "embeddings_dinov3.npz")
    used = set(e["idx"][np.isfinite(e["masked"][:, 0].astype(float))])
    S["used"] = np.array([i in used for i in S["idx"]])
    return S


def frame(v, f):
    p = MT.FRAMES["vedb" if v.startswith("session_") else "bv"](v, f)
    return Image.open(p).convert("RGB")


def square(im, box, t=T, pad=0.08):
    return MT._square(im, t, box, pad)


def triplet(S, z, r):
    """(Qwen box, SAM 3 box, SAM 3 box with background grey) as t x t tiles, same square window for the last two."""
    i = S["idx"][r]; v, f = str(z["video_id"][i]), int(z["frame_number"][i])
    im = frame(v, f); W, H = im.size; a = np.asarray(im)
    L, Tp, R, B = (int(x) for x in S["crop"][r]); h, w = (int(x) for x in S["mask_shape"][r])
    m = np.unpackbits(S["mask"][r])[: h * w].reshape(h, w).astype(bool)
    m = np.asarray(Image.fromarray(m).resize((max(1, R - L), max(1, B - Tp)), Image.NEAREST))
    full = np.zeros((H, W), bool); full[Tp:Tp + m.shape[0], L:L + m.shape[1]] = m[: H - Tp, : W - L]
    masked = Image.fromarray(np.where(full[..., None], a, GREY).astype(np.uint8))
    qb = tuple(z["bbox"][i]); sb = tuple(S["box"][r])
    return square(im, qb), square(im, sb), square(masked, sb, pad=0.0 if False else 0.08)


def pick_rows(S, z, ok_frames, ds, rng, cond, n_per=1, by_domain=True):
    cat = z["category"][S["idx"]]; dset = z["dataset"][S["idx"]]
    vid = z["video_id"][S["idx"]]; fno = z["frame_number"][S["idx"]]
    base = (dset == ds) & cond
    out = []
    groups = DOMS if by_domain else [None]
    for d in groups:
        sel = base & (np.array([DOM.get(c) == d for c in cat]) if d else True) & np.isin(cat, list(CORE))
        cand = rng.permutation(np.flatnonzero(sel))
        k = 0
        for r in cand:
            if (str(vid[r]), int(fno[r])) in ok_frames[ds]:
                continue
            q = z["bbox"][S["idx"][r]]
            if min(q[2] - q[0], q[3] - q[1]) < 40:                  # legible tiles
                continue
            out.append(r); k += 1
            if k == n_per:
                break
    return out


def label(d, xy, text, size=17, fill=INK, anchor="la"):
    d.text(xy, text, fill=fill, font=MT.font(size), anchor=anchor)


def sam3_examples(S, z, bad, rng):
    """Two blocks (BabyView, VEDB), 8 objects each (one per domain) in 2 rows of 4 triplets."""
    cols, tw = 4, 3 * T + 2 * G
    cw = tw + 28; W = 20 + cols * cw
    rowh = T + 52; blockh = 46 + 2 * rowh
    img = Image.new("RGB", (W, 60 + 2 * blockh), "white"); d = ImageDraw.Draw(img)
    label(d, (20, 12), "Qwen box  |  SAM 3 box  |  background removed (what the control embeds)", 22)
    for b, ds in enumerate(("bv", "vedb")):
        y0 = 60 + b * blockh
        label(d, (20, y0), NAME[ds], 24, DSC[ds])
        rows = pick_rows(S, z, bad, ds, rng, S["used"])
        for k, r in enumerate(rows):
            x = 20 + (k % cols) * cw; y = y0 + 40 + (k // cols) * rowh
            c = str(z["category"][S["idx"][r]])
            label(d, (x, y), f"{c} ({DLAB[DOM[c]]}) · score {S['score'][r]:.2f}", 16)
            for j, tl in enumerate(triplet(S, z, r)):
                img.paste(tl, (x + j * (T + G), y + 24))
    img.save(OUT / "R1j_examples_sam3.jpg", quality=90)


def sam3_quality(S, z, bad, rng, n=6):
    """Random objects (any category in the core set) in three SAM 3 score bands, BabyView and VEDB."""
    dset = z["dataset"][S["idx"]]
    bands = [("score ≥ 0.5", (S["score"] >= 0.5) & S["used"]),
             ("0 < score < 0.5", (S["score"] > 0) & (S["score"] < 0.5) & S["used"]),
             ("no object found (score 0; left out)", S["score"] == 0)]
    tw = 2 * T + G; cw = tw + 22; W = 20 + n * cw
    rowh = T + 50
    img = Image.new("RGB", (W, 40 + 2 * (44 + 3 * (rowh + 34))), "white"); d = ImageDraw.Draw(img)
    y = 10
    for ds in ("bv", "vedb"):
        label(d, (20, y), NAME[ds] + ": random objects by SAM 3 score (left: Qwen box; right: background removed)", 22, DSC[ds])
        y += 44
        for name, cond in bands:
            share = cond[dset == ds].mean()
            label(d, (20, y), f"{name}: {share:.0%} of sampled objects", 18, FAINT)
            rows = pick_rows(S, z, bad, ds, rng, cond, n_per=n, by_domain=False)
            for k, r in enumerate(rows):
                x = 20 + k * cw; c = str(z["category"][S["idx"][r]])
                label(d, (x, y + 28), f"{c} · {S['score'][r]:.2f}", 15)
                a, _, m = triplet(S, z, r) if S["score"][r] > 0 else (triplet_plain(z, S, r), None, None)
                img.paste(a, (x, y + 50))
                if m is not None:
                    img.paste(m, (x + T + G, y + 50))
            y += rowh + 34
    img.save(OUT / "R1j_examples_sam3_quality.jpg", quality=90)


def triplet_plain(z, S, r):
    i = S["idx"][r]
    return square(frame(str(z["video_id"][i]), int(z["frame_number"][i])), tuple(z["bbox"][i]))


# one category per domain, all among the 92 categories of the LVIS whole-vs-crop analysis
LVIS_CATS = [("dog", "animals"), ("car", "vehicles"), ("banana", "food_drink"), ("doll", "toys"),
             ("chair", "furniture_rooms"), ("cup", "household"), ("shoe", "clothing"), ("flower", "outside")]


def lvis_examples(rng, n=2):
    """Per category n LVIS images: whole image (box outlined) | crop, 4 categories per row."""
    MT._other_images("lvis", "dog", 1, rng, 10)                      # builds MT._LVIS (crop table)
    C = MT._LVIS["c"]; mp = pd.read_csv(MT.Q / "mapping/map_lvis.csv")
    th = 170; cols = 4
    cw = 2 * (n * (th + G)) + 10; W = 20 + cols * (cw + 20)
    W = 20 + cols * ((th * 4 // 3 + th + 3 * G) + 26)
    cellw = th * 4 // 3 + th + 3 * G + 26
    img = Image.new("RGB", (W, 60 + 2 * n * (th + G) + 2 * 40 + 40), "white"); d = ImageDraw.Draw(img)
    label(d, (20, 12), "LVIS: whole image (object outlined)  |  crop", 22)
    for k, (cat, dom) in enumerate(LVIS_CATS):
        ids = set(mp.query("category == @cat").id)
        c = C[C.category_id.isin(ids) & ((C.xmax - C.xmin) >= 60)]
        c = c.sample(frac=1, random_state=int(rng.integers(1 << 30))).drop_duplicates("path").head(n)
        x = 20 + (k % cols) * cellw; y = 60 + (k // cols) * (n * (th + G) + 40)
        label(d, (x, y), f"{cat} ({DLAB[dom]})", 17, tuple(int(MT._dom_rgb(cat)[i]) for i in range(3)))
        for j, r in enumerate(c.itertuples()):
            im = Image.open(r.path).convert("RGB")
            bw = th * 4 // 3; s = min(bw / im.width, th / im.height)          # whole image fitted, no cropping
            wim = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
            dw = ImageDraw.Draw(wim); dw.rectangle([r.xmin * s, r.ymin * s, r.xmax * s, r.ymax * s], outline=(255, 210, 0), width=3)
            img.paste(wim, (x + (bw - wim.width) // 2, y + 26 + j * (th + G) + (th - wim.height) // 2))
            img.paste(MT._square(im, th, (r.xmin, r.ymin, r.xmax, r.ymax)), (x + th * 4 // 3 + G, y + 26 + j * (th + G)))
    img.save(OUT / "R1d_examples_lvis_crop_vs_whole.jpg", quality=90)


def main():
    rng = np.random.default_rng(1)
    lvis_examples(rng); print("lvis")
    if "lvis" in sys.argv[1:]:
        return
    z = np.load(SAMPLE, allow_pickle=True)
    S = sam()
    bad = {}
    for ds in ("bv", "vedb"):                                         # frames with a person or face (any Qwen box)
        ix = MT.V.read_index(MT.STORE[ds], usecols=["video_id", "frame_number", "label"])
        lab = ix.label.astype(str)
        b = MT.basic_of(lab.unique()); cat = lab.map(b)
        p = lab.str.contains(r"\b(?:person|face|man|woman|child|baby|girl|boy|adult|people|kid|toddler)\b", regex=True) | \
            (cat.map(DOM) == "people")
        bad[ds] = set(zip(ix.video_id[p].astype(str), ix.frame_number[p].astype(int)))
        print(ds, "frames with people", len(bad[ds]), flush=True)
    sam3_examples(S, z, bad, rng); print("sam3 examples")
    sam3_quality(S, z, bad, rng); print("sam3 quality")


if __name__ == "__main__":
    main()
