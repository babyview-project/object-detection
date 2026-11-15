#!/usr/bin/env python3
"""Dataset overview figure: one row of example images per dataset (companion to the datasets table).

Rows run from curated, object-centric photographs to naturalistic egocentric video, matching tab:datasets:
ImageNet, THINGS, LVIS, VEDB, BabyView. Images are whole images (not crops) at their native aspect ratio, all
scaled to one row height, so the formats differ visibly (square THINGS crops, 4:3 LVIS/VEDB, portrait BabyView).
Each row is filled to the same width by adding images and trimming every image by the same horizontal center crop.

Picks live in a JSON file ({dataset: [image paths]}); the first run samples them at random (one image per
class / concept / session / video, dark or flat frames skipped) and writes the file so you can edit it.
BabyView and VEDB frames show people, so outputs go to revision_private/ (git-ignored). Before publishing,
replace picks with frames from sessions whose consent allows publication.

  python make_dataset_overview_figure.py                      # figure from picks (samples picks on first run)
  python make_dataset_overview_figure.py --resample --seed 3  # new random picks
  python make_dataset_overview_figure.py --candidates 40      # numbered contact sheets to hand-pick from

Outputs: revision_private/figures/dataset_overview/{dataset_overview.pdf,.png,picks.json,candidates_<dataset>.jpg}
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "revision_private" / "figures" / "dataset_overview"

ROOTS = {
    "ImageNet": Path("/data2/j7yang/imagenet-1k/extracted/train"),
    "THINGS": Path("/data2/dataset/babyview/related_data/things/object_images"),
    "LVIS": Path("/data2/j7yang/lvis/extracted/train2017"),
    "VEDB": Path("/ccn2a/dataset/babyview/related_data/vedb_1612/extracted_frames_1fps_adults"),
    "BabyView": Path("/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps"),
}
ORDER = list(ROOTS)
LABELS = {  # name, two short descriptor lines (details are in the table)
    "ImageNet": ("ImageNet", "Web photographs", "object-centric"),
    "THINGS": ("THINGS", "Web photographs", "object-centered"),
    "LVIS": ("LVIS", "Web photographs", "multi-object scenes"),
    "VEDB": ("VEDB", "Head-mounted video", "egocentric, adult"),
    "BabyView": ("BabyView", "Head-mounted video", "egocentric, child"),
}


def ok_frame(p: Path) -> bool:
    """Skip black, washed-out or flat frames (lens covered, camera in a bag)."""
    try:
        a = np.asarray(Image.open(p).convert("L").reduce(8), dtype=np.float32)
    except Exception:
        return False
    return 35 < a.mean() < 215 and a.std() > 30


REPO = HERE.parents[2]
IMAGENET_CDI = REPO / "analysis" / "comparison-datasets" / "outputs" / "mapping" / "imagenet_class_to_cdi.csv"
THINGS_MAP = REPO / "analysis" / "qwen-open-vocab" / "mapping" / "map_things.csv"
LVIS_ANN = Path("/data2/j7yang/lvis/annotations/lvis_v1_train.json")
VEDB_DET = Path("/ccn2a/dataset/babyview/related_data/vedb_1612/outputs/object_detections/cdi_adults")
BV_DET = Path("/data2/dataset/babyview/2025.2/outputs/object_detections/qwen3.8-27b_setupB_t1/boxes_index.csv.gz")
PERSON = {"person", "people", "man", "woman", "men", "women", "child", "children", "kid", "girl", "boy", "baby", "infant",
          "toddler", "adult", "lady", "mother", "father", "mom", "dad", "face", "head", "hair", "human", "teenager"}


def video_frames(ds: str, min_objects: int) -> dict[str, list[tuple[str, int]]]:
    """{video: [(frame path, n distinct objects)]} for frames with >= min_objects distinct detected classes and no
    person-like detection (any confidence for VEDB/YOLOE; any box for BabyView/Qwen). Faces still need a visual check."""
    import pandas as pd
    if ds == "VEDB":
        dfs = []
        for f in sorted(VEDB_DET.glob("*/bounding_box_predictions.csv")):
            dfs.append(pd.read_csv(f, usecols=["confidence", "class_name", "original_frame_path"]))
        df = pd.concat(dfs)
        df["path"], df["label"] = df.original_frame_path, df.class_name.astype(str).str.lower()
        df = df[(df.confidence > 0.3) | df.label.isin(PERSON)]
    else:
        df = pd.read_csv(BV_DET, usecols=["video_id", "frame_number", "label"])
        df["label"] = df.label.astype(str).str.lower()
        df["path"] = [str(ROOTS["BabyView"] / v / f"{int(k):05d}.jpg") for v, k in zip(df.video_id, df.frame_number)]
    bad = set(df.path[df.label.isin(PERSON) | df.label.str.split().str[-1].isin(PERSON)])
    g = df[~df.path.isin(bad)].groupby("path").label.nunique()
    g = g[g >= min_objects]
    out: dict[str, list[tuple[str, int]]] = {}
    for p, k in g.items():
        out.setdefault(Path(p).parent.name, []).append((p, int(k)))
    return out


def sample(ds: str, n: int, rng: random.Random) -> list[str]:
    """Candidate images: object-centric for ImageNet/THINGS (everyday CDI-mapped classes), busy multi-object scenes
    for LVIS, object-rich frames without detected people for VEDB/BabyView."""
    import pandas as pd
    root = ROOTS[ds]
    picks: list[str] = []
    if ds == "ImageNet":                              # flat dir: <wnid>_<n>_<wnid>.JPEG; one image per CDI-mapped class
        keep = set(pd.read_csv(IMAGENET_CDI).dropna(subset=["cdi_most_specific"]).wnid)
        by = {}
        for f in os.listdir(root):
            w = f.split("_", 1)[0]
            if f.endswith(".JPEG") and w in keep:
                by.setdefault(w, []).append(f)
        for w in rng.sample(sorted(by), min(n, len(by))):
            picks.append(str(root / rng.choice(by[w])))
    elif ds == "THINGS":                              # one image per CDI-mapped concept
        ids = sorted(set(pd.read_csv(THINGS_MAP).dropna(subset=["category"]).id) & set(os.listdir(root)))
        for c in rng.sample(ids, min(n, len(ids))):
            picks.append(str(root / c / rng.choice(sorted(f for f in os.listdir(root / c) if f.endswith(".jpg")))))
    elif ds == "LVIS":                                # cluttered scenes: >= 8 categories, >= 20 instances, no person
        ann = json.load(open(LVIS_ANN))
        person = {c["id"] for c in ann["categories"] if c["name"] == "person"}
        cats, inst = {}, {}
        for a in ann["annotations"]:
            cats.setdefault(a["image_id"], set()).add(a["category_id"])
            inst[a["image_id"]] = inst.get(a["image_id"], 0) + 1
        ok = sorted(i for i, c in cats.items() if len(c) >= 8 and inst[i] >= 20 and not c & person)
        name = {im["id"]: im["coco_url"].rsplit("/", 1)[1] for im in ann["images"]}
        picks = [str(root / name[i]) for i in rng.sample(ok, min(n, len(ok))) if (root / name[i]).exists()]
    else:                                             # video: one frame per session / child
        frames = video_frames(ds, min_objects=5 if ds == "VEDB" else 6)
        vids = sorted(frames)
        rng.shuffle(vids)
        subj = set()
        for v in vids:
            s = v.split("_", 1)[0] if ds == "BabyView" else v
            if s in subj:
                continue
            cand = sorted(frames[v], key=lambda t: -t[1])[:20]   # among the 20 most object-rich frames
            rng.shuffle(cand)
            for p, _ in cand[:5]:
                if ds == "BabyView" and Image.open(p).width > Image.open(p).height:
                    break                             # older landscape sessions; keep the current portrait format
                if ok_frame(Path(p)):
                    picks.append(p)
                    subj.add(s)
                    break
            if len(picks) == n:
                break
    return picks


def fill_row(paths: list[str], H: int, W: int, gap: int) -> list[Image.Image]:
    """Scale to height H; take images until the row reaches W; center-crop all widths by one factor to fit W."""
    ims, total = [], 0
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = im.resize((round(im.width * H / im.height), H), Image.LANCZOS)
        ims.append(im)
        total += im.width
        if total + gap * (len(ims) - 1) >= W:
            break
    f = min(1.0, (W - gap * (len(ims) - 1)) / total)
    out = []
    for im in ims:
        w = round(im.width * f)
        x = (im.width - w) // 2
        out.append(im.crop((x, 0, x + w, H)))
    slack = W - gap * (len(out) - 1) - sum(i.width for i in out)   # rounding; absorb in the last image
    if slack and f < 1:
        last = Image.open(paths[len(out) - 1]).convert("RGB")
        last = last.resize((round(last.width * H / last.height), H), Image.LANCZOS)
        w = out[-1].width + slack
        x = (last.width - w) // 2
        out[-1] = last.crop((x, 0, x + w, H))
    return out


def figure(picks: dict, width_in: float, row_in: float, dpi: int, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

    label_in, gap_in, vgap_in = 1.35, 0.03, 0.06
    img_w_in = width_in - label_in
    H, W, gap = round(row_in * dpi), round(img_w_in * dpi), round(gap_in * dpi)
    rows = [d for d in ORDER if picks.get(d)]
    fig_h = len(rows) * row_in + (len(rows) - 1) * vgap_in
    fig = plt.figure(figsize=(width_in, fig_h))
    for k, ds in enumerate(rows):
        strip = fill_row(picks[ds], H, W, gap)
        canvas = Image.new("RGB", (W, H), "white")
        x = 0
        for im in strip:
            canvas.paste(im, (x, 0))
            x += im.width + gap
        y0 = 1 - (k * (row_in + vgap_in) + row_in) / fig_h
        ax = fig.add_axes([label_in / width_in, y0, img_w_in / width_in, row_in / fig_h])
        ax.imshow(canvas, interpolation="none")
        ax.axis("off")
        name, l1, l2 = LABELS[ds]
        yc = y0 + row_in / fig_h / 2
        fig.text(0.0, yc + 0.17 * row_in / fig_h, name, fontsize=9, fontweight="bold", va="center")
        fig.text(0.0, yc - 0.06 * row_in / fig_h, l1, fontsize=7, color="#52514e", va="center")
        fig.text(0.0, yc - 0.24 * row_in / fig_h, l2, fontsize=7, color="#52514e", va="center")
    # curated -> naturalistic arrow along the left edge
    fig.add_artist(matplotlib.patches.FancyArrowPatch((label_in / width_in - 0.02, 0.99), (label_in / width_in - 0.02, 0.01),
                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, color="#8a8984", lw=0.8))
    for y, txt, va in ((0.99, "curated", "top"), (0.03, "naturalistic", "bottom")):   # end labels, set on the arrow
        fig.text(label_in / width_in - 0.02, y, txt, rotation=90, fontsize=6.5, color="#52514e", ha="center", va=va,
                 style="italic", bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"dataset_overview.{ext}", dpi=dpi)
    plt.close(fig)


def candidates(n: int, seed: int, out: Path):
    """Numbered contact sheet per dataset; copy the paths you want from candidates.json into picks.json."""
    rng = random.Random(seed)
    font = ImageFont.truetype(__import__("matplotlib.font_manager").font_manager.findfont("DejaVu Sans"), 18)
    allc = {}
    for ds in ORDER:
        ps = sample(ds, n, rng)
        allc[ds] = ps
        T, cols = 220, 10
        sheet = Image.new("RGB", (cols * T, -(-len(ps) // cols) * T), "white")
        for i, p in enumerate(ps):
            im = Image.open(p).convert("RGB")
            im.thumbnail((T - 6, T - 6))
            sheet.paste(im, ((i % cols) * T + 3, (i // cols) * T + 3))
            ImageDraw.Draw(sheet).text(((i % cols) * T + 6, (i // cols) * T + 4), str(i), fill="yellow", font=font,
                                       stroke_width=2, stroke_fill="black")
        sheet.save(out / f"candidates_{ds}.jpg", quality=85)
        print("wrote", out / f"candidates_{ds}.jpg")
    (out / "candidates.json").write_text(json.dumps(allc, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picks", type=Path, default=OUT / "picks.json")
    ap.add_argument("--resample", action="store_true", help="overwrite picks with a new random sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=14, help="picks per dataset (rows use as many as fit)")
    ap.add_argument("--candidates", type=int, default=0, help="write N-image contact sheets instead of the figure")
    ap.add_argument("--width", type=float, default=7.0, help="figure width, inches (table* textwidth)")
    ap.add_argument("--row", type=float, default=0.95, help="row height, inches")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.candidates:
        candidates(a.candidates, a.seed, OUT)
        return
    if a.resample or not a.picks.exists():
        rng = random.Random(a.seed)
        a.picks.write_text(json.dumps({d: sample(d, a.n, rng) for d in ORDER}, indent=1))
        print("sampled picks ->", a.picks)
    figure(json.loads(a.picks.read_text()), a.width, a.row, a.dpi, OUT)
    print("wrote", OUT / "dataset_overview.pdf")


if __name__ == "__main__":
    main()
