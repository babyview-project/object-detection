#!/usr/bin/env python3
"""Side-by-side boxes on the pilot's gold frames: Gemini detect vs YOLOE + CLIP filter.

Left:  Gemini (`detect` task, no YOLOE input): every box it returned, with its label.
Middle (when out/private/<model>__detect_open__p1_montage72.jsonl exists): Gemini
       `detect_open`, open vocabulary (the most precise everyday name per object); then
       `detect_combo` (open label, closest CDI word, format; box colour = format) if run.
       Person / picture are dropped here too; words not in the vocabulary carry a "*".
Right: every YOLOE box in the frame (person / picture excluded, as in the pipeline),
       scored with the pipeline's CLIP filter (openai/clip-vit-base-patch32, cosine of
       the crop vs "a photo of a {label}"): solid orange = kept at --threshold (0.27),
       dashed grey = dropped. Labels show the CLIP score.
Both:  the rater-validated gold object is marked with a white "gold" tag on its YOLOE box.

The 868 h CLIP scores were on /data2 (gone), so they are recomputed here on CPU in
float32 (the pipeline ran bfloat16 on GPU; scores agree to about +-0.003).

Frames: one per category (72), seeded; --n to change. Output is private (frames show
children's homes and family members):
  out/private/box_compare/<model>/page_XX.jpg      6 frame pairs per page
  out/private/box_compare/<model>/frames.csv       per frame: counts per pipeline
  out/private/box_compare/<model>/yoloe_clip_scores.csv

  python compare_boxes_montage.py [--model gemini-2.5-flash] [--n 72] [--threshold 0.27]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"
YOLOE = Path("/ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_allframes_1fps")
EXCLUDE = {"person", "picture"}
FW, FH = 512, 910
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BLUE, ORANGE, GREY, WHITE = (42, 120, 214), (235, 104, 52), (150, 150, 150), (255, 255, 255)
GREEN = (20, 140, 98)
FORMAT_COLORS = {"real": (20, 140, 98), "toy": (201, 133, 0), "photo": (227, 73, 72),
                 "drawing": (213, 81, 129), "screen": (42, 120, 214)}


def combo_panel(frame, rec, title):
    """detect_combo output: box colour = format; text 'label (cdi word) · format'."""
    boxes = []
    if rec and rec["error"] is None:
        for o in json.loads(rec["raw"]).get("objects", []):
            b = o.get("box_2d", [])
            if len(b) != 4:
                continue
            y0, x0, y1, x1 = (min(1000, max(0, v)) for v in b)
            if x1 <= x0 or y1 <= y0:
                continue
            lab, cdi, fm = (str(o.get(k, "")).strip().lower() for k in ("label", "cdi_word", "format"))
            text = lab + (f" ({cdi})" if cdi and cdi != lab else "") + f" · {fm}"
            boxes.append((text, (x0 * FW / 1000, y0 * FH / 1000, x1 * FW / 1000, y1 * FH / 1000),
                          FORMAT_COLORS.get(fm, GREY), False, 3))
    n = {k: sum(b[2] == c for b in boxes) for k, c in FORMAT_COLORS.items()}
    sub = (f"{len(boxes)} objects · " + ", ".join(f"{v} {k}" for k, v in n.items() if v)) if rec else "no reply"
    return draw_panel(frame, boxes, title, sub)
SEED = 20261019
VOCAB: set = set()


def yoloe_index():
    return {p.name.lower().removesuffix("_processed"): p for p in YOLOE.iterdir()}


def load_frames(n):
    g = pd.read_csv(PRIV / "gold_frames.csv", dtype={"frame_id": str})
    g["unit_id"] = g.video + "/" + g.frame_id
    g = g.drop_duplicates("unit_id")
    return g.sample(frac=1, random_state=SEED).groupby("category").head(1).head(n).sort_values("category")


def gemini_boxes(raw):
    out = []
    for o in json.loads(raw).get("objects", []):
        b = o.get("box_2d", [])
        if len(b) != 4:
            continue
        y0, x0, y1, x1 = (min(1000, max(0, v)) for v in b)
        if x1 > x0 and y1 > y0:  # 0-1000 of the image sent (288x512), same aspect as the frame
            out.append((str(o.get("label", "")), (x0 * FW / 1000, y0 * FH / 1000, x1 * FW / 1000, y1 * FH / 1000)))
    return out


class Clip:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(1)
        self.torch = torch
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self.text = {}

    def _vec(self, out):
        out = out if self.torch.is_tensor(out) else (getattr(out, "pooler_output", None)
                                                      if getattr(out, "pooler_output", None) is not None else out[0])
        return self.torch.nn.functional.normalize(out.float(), dim=-1)

    def score(self, crops, labels):
        torch = self.torch
        new = sorted({l for l in labels if l not in self.text})
        with torch.inference_mode():
            if new:
                t = self.proc(text=[f"a photo of a {l}" for l in new], return_tensors="pt", padding=True, truncation=True)
                v = self._vec(self.model.get_text_features(**t))
                self.text.update({l: v[i] for i, l in enumerate(new)})
            im = self._vec(self.model.get_image_features(**self.proc(images=crops, return_tensors="pt")))
        return [float(im[i] @ self.text[l]) for i, l in enumerate(labels)]


def draw_panel(frame, boxes, title, sub):
    """boxes: list of (label_text, box, color, dashed, width)."""
    W, top = FW, 48
    img = Image.new("RGB", (W, FH + top), WHITE)
    img.paste(frame, (0, top))
    d = ImageDraw.Draw(img)
    f, fb = ImageFont.truetype(FONT, 12), ImageFont.truetype(FONT_B, 15)
    d.text((6, 4), title, font=fb, fill=(11, 11, 11))
    d.text((6, 26), sub, font=ImageFont.truetype(FONT, 12), fill=(82, 81, 78))
    for text, (x0, y0, x1, y1), col, dashed, w in boxes:
        y0, y1 = y0 + top, y1 + top
        below = text == "gold"  # gold tag on the bottom edge so the box's own label stays visible
        if dashed:
            for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // 8))
                for k in range(0, n, 2):
                    p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                    q = (a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n)
                    d.line([p, q], fill=col, width=w)
        else:
            d.rectangle((x0, y0, x1, y1), outline=col, width=w)
        if text:
            tw = d.textlength(text, font=f)
            tx = min(max(0, x0), W - tw - 4)
            ty = min(y1 + 1, FH + top - 15) if below else max(top, y0 - 15)
            d.rectangle((tx, ty, tx + tw + 4, ty + 14), fill=col)
            d.text((tx + 2, ty), text, font=f, fill=WHITE if col != WHITE else (11, 11, 11))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-2.5-flash", choices=["gemini-2.5-flash", "gemini-2.5-flash-lite"])
    ap.add_argument("--n", type=int, default=72)
    ap.add_argument("--threshold", type=float, default=0.27)
    ap.add_argument("--per-page", type=int, default=6)
    args = ap.parse_args()
    out = PRIV / "box_compare" / args.model
    out.mkdir(parents=True, exist_ok=True)

    recs = {}
    for line in open(PRIV / f"{args.model}__detect__p1.jsonl"):
        r = json.loads(line)
        recs[r["unit_id"]] = r  # last record wins (retries)
    frames = load_frames(args.n)
    frames = frames[frames.unit_id.map(lambda u: u in recs and recs[u]["error"] is None)]
    global VOCAB
    VOCAB = {d["name"] for d in json.load(open(HERE.parent / "cvat_cdi_labels.json"))}
    open_path = PRIV / f"{args.model}__detect_open__p1_montage72.jsonl"
    opn = {}
    if open_path.exists():
        for line in open(open_path):
            o = json.loads(line)
            opn[o["unit_id"]] = o
    combo = {}
    cpath = PRIV / f"{args.model}__detect_combo__p1_montage72.jsonl"
    if cpath.exists():
        for line in open(cpath):
            o = json.loads(line)
            combo[o["unit_id"]] = o
    yi, clip = yoloe_index(), Clip()

    cache = {}
    scores, summary, pairs = [], [], []
    for r in frames.itertuples():
        if r.video not in cache:
            b = pd.read_csv(yi[r.video] / "bounding_box_predictions.csv", dtype={"frame_id": str})
            b = b[b.xmin.notna()].copy()
            b["frame_id"] = b.frame_id.str.zfill(5)
            b["class_name"] = b.class_name.astype(str).str.strip().str.lower()
            cache = {r.video: b}  # one video at a time
        yb = cache[r.video]
        yb = yb[(yb.frame_id == r.frame_id) & ~yb.class_name.isin(EXCLUDE)].reset_index(drop=True)
        frame = Image.open(r.frame_path).convert("RGB")
        assert frame.size == (FW, FH), frame.size
        rgb = np.asarray(frame)
        crops, boxes_i = [], []
        for y in yb.itertuples():
            x0, x1 = sorted((max(0, min(FW, y.xmin)), max(0, min(FW, y.xmax))))
            y0, y1 = sorted((max(0, min(FH, y.ymin)), max(0, min(FH, y.ymax))))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            crops.append(rgb[y0:y1, x0:x1]); boxes_i.append((y.class_name, (x0, y0, x1, y1), y.confidence))
        sims = clip.score(crops, [c for c, _, _ in boxes_i]) if crops else []
        gold = (r.xmin, r.ymin, r.xmax, r.ymax)
        right, kept = [], 0
        for (cls, bx, conf), s in zip(boxes_i, sims):
            keep = s >= args.threshold
            kept += keep
            scores.append({"unit_id": r.unit_id, "class_name": cls, "yolo_conf": conf, "clip_sim": s, "kept": keep,
                           "is_gold": bool(cls == r.category and max(abs(bx[0] - gold[0]), abs(bx[1] - gold[1])) <= 2)})
            right.append((f"{cls} {s:.2f}", bx, ORANGE if keep else GREY, not keep, 3 if keep else 2))
        right.sort(key=lambda b: b[2] == ORANGE)  # draw kept boxes on top
        gem = [(l.strip().lower(), b) for l, b in gemini_boxes(recs[r.unit_id]["raw"])]
        gem = [(l, b) for l, b in gem if l not in EXCLUDE]  # same exclusions as the pipeline
        left = [(lab if lab in VOCAB else f"{lab}*", bx, BLUE, False, 3) for lab, bx in gem]
        tag = ("gold", gold, WHITE, False, 2)
        lp = draw_panel(frame, left + [tag], f"Gemini {args.model.replace('gemini-', '')}",
                        f"{len(gem)} objects (* = word not in the vocabulary) · gold: {r.category}")
        rp = draw_panel(frame, right + [tag], f"YOLOE + CLIP ≥ {args.threshold}",
                        f"{len(right)} YOLOE boxes · {kept} kept (solid) · {len(right) - kept} dropped (dashed)")
        panels = [lp]
        n_open, open_labels = None, ""
        if opn:
            o = opn.get(r.unit_id)
            ob = gemini_boxes(o["raw"]) if o and o["error"] is None else []
            ob = [(l.strip().lower(), b) for l, b in ob]
            n_open, open_labels = len(ob), ";".join(sorted({l for l, _ in ob}))
            panels.append(draw_panel(frame, [(l, b, GREEN, False, 3) for l, b in ob] + [tag],
                                     f"Gemini {args.model.replace('gemini-', '')} · open vocabulary",
                                     f"{len(ob)} objects, everyday names (people = person)"))
        if combo:
            panels.append(combo_panel(frame, combo.get(r.unit_id),
                                      f"Gemini {args.model.replace('gemini-', '')} · combined (label, CDI, format)"))
        panels.append(rp)
        pair = Image.new("RGB", (len(panels) * (FW + 12) - 12, FH + 48), WHITE)
        for k, im in enumerate(panels):
            pair.paste(im, (k * (FW + 12), 0))
        pairs.append(pair)
        summary.append({"unit_id": r.unit_id, "gold_category": r.category, "n_gemini": len(gem),
                        "n_open": n_open, "open_labels": open_labels,
                        "n_yoloe": len(right), "n_kept": kept,
                        "gemini_labels": ";".join(sorted({l for l, _ in gem})),
                        "yoloe_kept_labels": ";".join(sorted({c for (c, _, _), s in zip(boxes_i, sims) if s >= args.threshold}))})

    pd.DataFrame(scores).to_csv(out / "yoloe_clip_scores.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "frames.csv", index=False)
    cols, g = 2, 16
    if opn or combo:
        cols = 1 if combo else cols
        args.per_page = min(args.per_page, 4)
    for p in range(0, len(pairs), args.per_page):
        chunk = pairs[p:p + args.per_page]
        pw, ph = chunk[0].size
        rows = -(-len(chunk) // cols)
        page = Image.new("RGB", (cols * (pw + g) + g, rows * (ph + g) + g), (236, 235, 230))
        for i, im in enumerate(chunk):
            page.paste(im, (g + (i % cols) * (pw + g), g + (i // cols) * (ph + g)))
        page.save(out / f"page_{p // args.per_page + 1:02d}.jpg", quality=88)
    s = pd.DataFrame(summary)
    print(f"{len(s)} frames -> {out}  | per frame: Gemini {s.n_gemini.mean():.1f}, YOLOE {s.n_yoloe.mean():.1f}, "
          f"kept by CLIP {s.n_kept.mean():.1f}; gold box kept by CLIP: "
          f"{pd.DataFrame(scores).query('is_gold').kept.mean():.0%}")


if __name__ == "__main__":
    main()
