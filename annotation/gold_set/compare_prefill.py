#!/usr/bin/env python3
"""Compare Gemini pre-fill versions on the gold set v2 frames, for choosing what RAs correct.

Versions (gemini_pilot/out/private, all cap 30, full frame at 512x910, thinking off):
  A  2.5 Flash       detect_combo        open label + CDI word + format
  B  2.5 Flash       detect_open_format  open label + format (chosen: the pre-fill RAs get)
  C  2.5 Flash-Lite  detect_open_format  open label + format
Boxes are filtered as in build_cvat_tasks.py (not discrete, parts, duplicates); photo and
drawing are shown as picture, as RAs see them.

Writes
  out/private/compare_prefill/summary.csv     per version: boxes, frames at the cap, formats, cost
  out/private/compare_prefill/<frame_uid>.jpg  the three versions side by side
  out/private/compare_prefill/index.html       local page: summary + the sampled frames
Frames stay local: open index.html from the server's disk, never upload it.

Usage
  python compare_prefill.py [--n 60]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from build_cvat_tasks import PRIV, VALID129, EXCLUDE, dedup, load_gemini
from make_overlays import font

GP = Path(__file__).resolve().parent.parent / "gemini_pilot" / "out" / "private"
VERSIONS = {
    "A": ("2.5 Flash · open + CDI + format", "gemini-2.5-flash__detect_combo__p1_gold_v2_frames_cap30"),
    "B": ("2.5 Flash · open + format", "gemini-2.5-flash__detect_open_format__p1_gold_v2_frames_cap30"),
    "C": ("2.5 Flash-Lite · open + format", "gemini-2.5-flash-lite__detect_open_format__p1_gold_v2_frames_cap30"),
}
PRICES = {"gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40)}
FMT_COLORS = {"real": (0, 170, 0), "toy": (230, 120, 0), "picture": (200, 0, 200), "screen": (0, 120, 255)}
OUT = PRIV / "compare_prefill"
SEED = 20261021


def panel(path, boxes, title, fnt, tfnt):
    im = Image.open(path).convert("RGB")
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for b in boxes.itertuples():
        c = FMT_COLORS.get(b.gemini_format, (255, 255, 255))
        x0, y0, x1, y1 = (v * 2 for v in (b.xmin, b.ymin, b.xmax, b.ymax))
        d.rectangle((x0, y0, x1, y1), outline=c, width=3)
        t = f"{b.name} | {b.gemini_format or '-'}"
        tw, th = d.textbbox((0, 0), t, font=fnt)[2:]
        ty = y0 - th - 4 if y0 - th - 4 >= 0 else y0
        tx = max(0, min(x0, im.width - tw - 6))
        d.rectangle((tx, ty, tx + tw + 6, ty + th + 4), fill=c)
        d.text((tx + 3, ty + 1), t, fill=(255, 255, 255), font=fnt)
    head = Image.new("RGB", (im.width, 44), (30, 30, 30))
    ImageDraw.Draw(head).text((10, 8), f"{title} · {len(boxes)} boxes", fill=(255, 255, 255), font=tfnt)
    out = Image.new("RGB", (im.width, im.height + 44))
    out.paste(head, (0, 0))
    out.paste(im, (0, 44))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="frames to draw (stratified: random and depiction)")
    args = ap.parse_args()
    classes = {l.strip().lower() for l in VALID129.read_text().splitlines() if l.strip()} - EXCLUDE
    frames = pd.read_csv(PRIV / "frames.csv")
    boxes, rows = {}, []
    for k, (title, stem) in VERSIONS.items():
        g = load_gemini(frames, classes, GP / f"{stem}.jsonl")
        g = pd.concat([dedup(x) for _, x in g.groupby("frame_uid")], ignore_index=True)
        boxes[k] = g
        per = g.groupby("frame_uid").size().reindex(frames.frame_uid, fill_value=0)
        u = json.load(open(GP / f"usage_{stem}.json"))
        pi, po = PRICES[u["model"]]
        cost = (u["input_tokens"] * pi + u["output_tokens"] * po) / 1e6
        fmt = g.gemini_format.replace("", "none").value_counts()
        rows.append({"version": k, "setup": title, "boxes": len(g), "boxes_per_frame_median": per.median(),
                     "frames_empty": int((per == 0).sum()), "frames_at_cap": int((per >= 30).sum()),
                     "distinct_names": g.name.nunique(), "failed_calls": u.get("failures", 0),
                     **{f"fmt_{f}": int(fmt.get(f, 0)) for f in ["real", "toy", "picture", "screen"]},
                     "in_tok_per_frame": round(u["input_tokens"] / u["scored"]),
                     "out_tok_per_frame": round(u["output_tokens"] / u["scored"]),
                     "cost_1000_frames_online": round(cost, 2),
                     "full_run_bvmain_5s_batch": round(cost / u["scored"] * 5141074 / 5 / 2)})
    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.T.to_string())

    # Same frames for every version: depiction frames in proportion, fixed seed
    n_dep = round(args.n * (frames.stratum == "depiction").mean())
    pick = pd.concat([frames[frames.stratum == "depiction"].sample(n_dep, random_state=SEED),
                      frames[frames.stratum == "random"].sample(args.n - n_dep, random_state=SEED)])
    fnt, tfnt = font(18), font(24)
    legend = " ".join(f'<span style="border:3px solid rgb{c};padding:1px 6px">{f}</span>' for f, c in FMT_COLORS.items())
    parts = ["<!doctype html><meta charset=utf-8><title>Pre-fill comparison</title>"
             "<body style='font-family:sans-serif;margin:16px'><h1>Gemini pre-fill comparison, gold set v2</h1>",
             summary.set_index("version").T.to_html(), f"<p>Box text: name | format (photo and drawing shown as picture). Box color = format: {legend}</p>"]
    for r in pick.itertuples():
        ims = [panel(r.frame_path, boxes[k][boxes[k].frame_uid == r.frame_uid], f"{k}: {VERSIONS[k][0]}", fnt, tfnt)
               for k in VERSIONS]
        row = Image.new("RGB", (sum(i.width for i in ims) + 20 * (len(ims) - 1), ims[0].height), (255, 255, 255))
        x = 0
        for i in ims:
            row.paste(i, (x, 0))
            x += i.width + 20
        row.save(OUT / f"{r.frame_uid}.jpg", quality=88)
        parts.append(f"<h3>{r.frame_uid} · {r.stratum}</h3><a href='{r.frame_uid}.jpg'>"
                     f"<img src='{r.frame_uid}.jpg' loading=lazy style='width:100%;max-width:1600px'></a>")
    (OUT / "index.html").write_text("\n".join(parts) + "\n")
    print(f"{len(pick)} frames drawn; open {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
