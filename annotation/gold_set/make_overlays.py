#!/usr/bin/env python3
"""Draw the pre-filled boxes RAs get (Gemini's) on the frames, for checking before RAs start.

Reads out/private/prefill/merged.csv (build_cvat_tasks.py) and writes, per CVAT task,
  out/private/overlays/<rater>_batch<k>/<frame_uid>.jpg   frame at 2x, boxes colored by source,
                                                          labelled "name | Gemini format"
  out/private/overlays/index.html                         local page with every task's frames
Frames stay local: open index.html from the server's disk (or a lab machine), never upload it.

Usage
  python make_overlays.py [--tasks Mira_batch1 ...]
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from build_cvat_tasks import gemini_prefill

HERE = Path(__file__).resolve().parent
PRIV = HERE / "out" / "private"
OUT = PRIV / "overlays"
SCALE = 2
COLORS = {"both": (0, 200, 0), "yoloe_clip": (0, 120, 255), "gemini": (255, 140, 0),
          "both_person": (200, 0, 200), "yoloe_person": (255, 0, 120)}
LEGEND = {"both": "YOLOE+CLIP and Gemini", "yoloe_clip": "YOLOE+CLIP only", "gemini": "Gemini only",
          "both_person": "person, both", "yoloe_person": "person, YOLOE only"}


def font(size):
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw(path: Path, boxes: pd.DataFrame, out: Path, fnt):
    im = Image.open(path).convert("RGB")
    im = im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for b in boxes.itertuples():
        c = COLORS.get(b.source, (255, 255, 255))
        x0, y0, x1, y1 = (v * SCALE for v in (b.xmin, b.ymin, b.xmax, b.ymax))
        d.rectangle((x0, y0, x1, y1), outline=c, width=3)
        t = f"{b.name} | {b.gemini_format or '-'}"   # Gemini's format; "-" = YOLOE-only box
        tw, th = d.textbbox((0, 0), t, font=fnt)[2:]
        ty = y0 - th - 4 if y0 - th - 4 >= 0 else y0
        tx = max(0, min(x0, im.width - tw - 6))          # keep the label inside the frame
        d.rectangle((tx, ty, tx + tw + 6, ty + th + 4), fill=c)
        d.text((tx + 3, ty + 1), t, fill=(0, 0, 0) if sum(c) > 400 else (255, 255, 255), font=fnt)
    im.save(out, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", help="task names (default: all)")
    args = ap.parse_args()
    frames = pd.read_csv(PRIV / "frames.csv")
    m = pd.read_csv(PRIV / "prefill" / "merged.csv", keep_default_na=False)
    m = gemini_prefill(m)                                # what RAs see: Gemini boxes, Gemini coordinates
    tasks = sorted(d.name for d in (PRIV / "cvat").iterdir() if d.is_dir())
    tasks = [t for t in tasks if not args.tasks or t in args.tasks]
    fnt = font(18)
    legend = " ".join(f'<span style="border:3px solid rgb{c};padding:1px 6px">{LEGEND[s]}</span>'
                      for s, c in COLORS.items())
    parts = [f"<!doctype html><meta charset=utf-8><title>Pre-fill check</title><body style='font-family:sans-serif'>"
             f"<h1>Pre-fill check</h1><p>{legend}</p><p>The boxes RAs get: Gemini's only, in Gemini's "
             f"coordinates (YOLOE-only boxes are hidden). Box text: name | Gemini's format (real, toy, "
             f"picture, screen), pre-filled for RAs to correct.</p>"]
    for t in tasks:
        uids = sorted(p.stem for p in (PRIV / "cvat" / t / "images").glob("*.jpg"))
        (OUT / t).mkdir(parents=True, exist_ok=True)
        fr = frames.set_index("frame_uid").loc[uids]
        parts.append(f"<h2 id='{t}'>{t} ({len(uids)} frames)</h2><div style='display:flex;flex-wrap:wrap;gap:8px'>")
        for uid in uids:
            bx = m[m.frame_uid == uid]
            draw(Path(fr.at[uid, "frame_path"]), bx, OUT / t / f"{uid}.jpg", fnt)
            meta = f"{uid} · {fr.at[uid, 'stratum']} · {len(bx)} boxes"
            parts.append(f"<figure style='margin:0;width:420px'><a href='{t}/{uid}.jpg'>"
                         f"<img src='{t}/{uid}.jpg' loading=lazy style='width:100%'></a>"
                         f"<figcaption>{html.escape(meta)}</figcaption></figure>")
        parts.append("</div>")
        print(f"{t}: {len(uids)} frames")
    (OUT / "index.html").write_text("\n".join(parts) + "\n")
    print(f"open {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
