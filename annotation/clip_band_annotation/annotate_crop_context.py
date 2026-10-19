#!/usr/bin/env python3
"""Two-step crop annotation: crop alone, then crop with its frame.

For each item the rater answers "Is this a {label}?" twice:
  step 1  the crop alone (what CLIP sees)            -> crop_answer
  step 2  the crop next to the full frame, box drawn -> frame_answer
  step 3  only if frame_answer is "no": what is it?  -> actual_label
Answers: y = yes, n = no, u = can't tell.

The crop answer is locked once the frame has been shown, so it is never
influenced by the frame. Items come in the rater's assignment order; CLIP
scores and bands are never shown.

Controls
  y / n / u  answer the current step
  Enter      submit the "what is it?" label (blank is allowed)
  ← / p      previous item (review; only the frame answer can be changed)
  →          next item (only after the item is complete)
  + / -      zoom

Usage
  python annotate_crop_context.py --rater A
Results: out/private/annotations_<rater>.csv (saved after every answer).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "private"
IMG = OUT / "images"
LABELS_JSON = HERE.parent / "cvat_cdi_labels.json"
EXTRA_LABELS = ["(not a CDI object)", "(no object / background)", "(person or body part)"]
FIELDS = ["item_id", "rater", "class_name", "crop_answer", "frame_answer", "actual_label",
          "label_done", "crop_rt_s", "frame_rt_s", "label_rt_s", "answered_at"]
ANSWER = {"y": "yes", "n": "no", "u": "cant_tell"}
CROP_MIN_PX, CROP_MAX_PX, FRAME_H_PX = 300, 520, 620


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def save_rows(path: Path, rows: dict[str, dict], order: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows[i] for i in order if i in rows)
    os.replace(tmp, path)


def complete(row: dict | None) -> bool:
    if not row or row.get("crop_answer") not in ANSWER.values():
        return False
    if row.get("frame_answer") not in ANSWER.values():
        return False
    return row["frame_answer"] != "no" or row.get("label_done") == "1"


class App:
    def __init__(self, rater: str):
        self.rater = rater
        with open(OUT / f"assignments_{rater}.csv", newline="") as f:
            a = list(csv.DictReader(f))
        self.order = [r["item_id"] for r in sorted(a, key=lambda r: int(r["order"]))]
        self.label = {r["item_id"]: r["class_name"] for r in a}
        self.csv_path = OUT / f"annotations_{rater}.csv"
        self.rows = load_rows(self.csv_path)
        vocab = sorted(d["name"] for d in json.loads(LABELS_JSON.read_text()))
        self.vocab = EXTRA_LABELS + vocab
        self.zoom = 1.0
        self.idx = next((i for i, k in enumerate(self.order) if not complete(self.rows.get(k))), 0)

        self.root = tk.Tk()
        self.root.title(f"Crop + context annotation — rater {rater}")
        self.root.geometry("1250x860")
        self._build()
        self._show()

    # ---- UI -----------------------------------------------------------------
    def _build(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)
        self.status = tk.StringVar()
        ttk.Label(top, textvariable=self.status).pack(side=tk.LEFT)
        self.canvas = tk.Canvas(self.root, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.prompt = tk.StringVar()
        tk.Label(self.root, textvariable=self.prompt, font=("", 16, "bold"), anchor=tk.W, padx=10,
                 pady=8).pack(fill=tk.X)
        self.entry_row = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        ttk.Label(self.entry_row, text="What is it?").pack(side=tk.LEFT)
        self.entry_var = tk.StringVar()
        self.entry = ttk.Combobox(self.entry_row, textvariable=self.entry_var, values=self.vocab, width=40)
        self.entry.pack(side=tk.LEFT, padx=8)
        self.entry.bind("<KeyRelease>", self._filter_vocab)
        self.entry.bind("<Return>", lambda e: self._submit_label())
        ttk.Label(self.entry_row, text="type to filter · Enter to submit (blank allowed)",
                  foreground="gray").pack(side=tk.LEFT)
        self.hint = ttk.Label(self.root, text="Y = yes · N = no · U = can't tell · ← previous · → next · +/- zoom",
                              foreground="gray")
        self.hint.pack(pady=4)
        for k, v in (("y", "y"), ("n", "n"), ("u", "u")):
            self.root.bind(f"<KeyPress-{k}>", lambda e, v=v: self._answer(v))
            self.root.bind(f"<KeyPress-{k.upper()}>", lambda e, v=v: self._answer(v))
        self.root.bind("<Left>", lambda e: self._move(-1))
        self.root.bind("<KeyPress-p>", lambda e: self._move(-1))
        self.root.bind("<Right>", lambda e: self._move(1))
        self.root.bind("<plus>", lambda e: self._zoom(0.25))
        self.root.bind("<equal>", lambda e: self._zoom(0.25))
        self.root.bind("<minus>", lambda e: self._zoom(-0.25))
        self.root.bind("<Configure>", lambda e: self._draw())

    def _filter_vocab(self, event):
        if event.keysym in ("Return", "Up", "Down"):
            return
        t = self.entry_var.get().strip().lower()
        self.entry["values"] = [w for w in self.vocab if t in w.lower()] if t else self.vocab

    # ---- state --------------------------------------------------------------
    @property
    def item(self) -> str:
        return self.order[self.idx]

    def _row(self) -> dict:
        k = self.item
        if k not in self.rows:
            self.rows[k] = {f: "" for f in FIELDS} | {"item_id": k, "rater": self.rater,
                                                       "class_name": self.label[k]}
        return self.rows[k]

    def _step(self) -> int:
        r = self.rows.get(self.item)
        if not r or not r.get("crop_answer"):
            return 1
        if not r.get("frame_answer"):
            return 2
        if r["frame_answer"] == "no" and r.get("label_done") != "1":
            return 3
        return 4  # complete

    def _show(self):
        k = self.item
        self.crop = Image.open(IMG / f"{k}_crop.jpg").convert("RGB")
        self.frame = Image.open(IMG / f"{k}_frame.jpg").convert("RGB")
        self.t0 = time.monotonic()
        step = self._step()
        n_done = sum(complete(self.rows.get(i)) for i in self.order)
        self.status.set(f"Item {self.idx + 1} / {len(self.order)} · completed {n_done} · id {k}")
        lab = self.label[k]
        r = self.rows.get(k, {})
        if step == 1:
            self.prompt.set(f"Crop only — is this a  “{lab}”?   (Y / N / U)")
        elif step == 2:
            self.prompt.set(f"With the frame — is the boxed object a  “{lab}”?   (Y / N / U)")
        elif step == 3:
            self.prompt.set(f"Not a “{lab}”. What is it?  (pick or type, then Enter)")
        else:
            lbl = f" · it is: {r.get('actual_label') or '—'}" if r.get("frame_answer") == "no" else ""
            self.prompt.set(f"Done — crop: {r.get('crop_answer')} · frame: {r.get('frame_answer')}{lbl}"
                            f"   (Y / N / U changes the frame answer · → next)")
        if step == 3:
            self.entry_var.set(r.get("actual_label", ""))
            self.entry["values"] = self.vocab
            self.entry_row.pack(fill=tk.X, before=self.hint)
            self.entry.focus_set()
        else:
            self.entry_row.pack_forget()
            self.root.focus_set()
        self._draw()

    def _draw(self):
        if not hasattr(self, "crop"):
            return
        c = self.canvas
        c.delete("all")
        cw, ch = max(c.winfo_width(), 400), max(c.winfo_height(), 300)
        w, h = self.crop.size
        s = min(CROP_MAX_PX / max(w, h), max(CROP_MIN_PX / max(w, h), 1.0)) * self.zoom
        crop = self.crop.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)
        self._ph_crop = ImageTk.PhotoImage(crop)
        show_frame = self._step() >= 2
        if not show_frame:
            c.create_image(cw // 2, ch // 2, image=self._ph_crop)
            return
        fw, fh = self.frame.size
        fs = min(FRAME_H_PX, ch - 20) / fh * self.zoom
        frame = self.frame.resize((round(fw * fs), round(fh * fs)), Image.Resampling.LANCZOS)
        self._ph_frame = ImageTk.PhotoImage(frame)
        c.create_image(cw // 3, ch // 2, image=self._ph_crop)
        c.create_image(2 * cw // 3 + 40, ch // 2, image=self._ph_frame)

    def _save(self):
        save_rows(self.csv_path, {k: {f: v.get(f, "") for f in FIELDS} for k, v in self.rows.items()},
                  self.order)
        self.status.set(self.status.get().split(" · saved")[0] + f" · saved {datetime.now():%H:%M:%S}")

    # ---- actions ------------------------------------------------------------
    def _answer(self, key: str):
        step = self._step()
        if step == 3:
            return  # typing into the label box
        r = self._row()
        rt = f"{time.monotonic() - self.t0:.2f}"
        r["answered_at"] = datetime.now().isoformat(timespec="seconds")
        if step == 1:
            r["crop_answer"], r["crop_rt_s"] = ANSWER[key], rt
        else:  # step 2, or re-answering the frame question on a completed item
            r["frame_answer"], r["frame_rt_s"] = ANSWER[key], rt
            if key != "n":
                r["actual_label"], r["label_rt_s"], r["label_done"] = "", "", "1"
            else:
                r["label_done"] = ""
        self._save()
        if self._step() == 4:
            self._move(1)
        else:
            self._show()

    def _submit_label(self):
        if self._step() != 3:
            return
        r = self._row()
        r["actual_label"] = self.entry_var.get().strip()
        r["label_rt_s"] = f"{time.monotonic() - self.t0:.2f}"
        r["label_done"] = "1"
        self._save()
        self._move(1)

    def _move(self, d: int):
        if d > 0 and not complete(self.rows.get(self.item)):
            return
        new = self.idx + d
        if 0 <= new < len(self.order):
            self.idx = new
            self._show()
        elif new >= len(self.order):
            self.prompt.set("All items done. Thank you!")

    def _zoom(self, d: float):
        self.zoom = min(4.0, max(0.25, self.zoom + d))
        self._draw()

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rater", required=True)
    App(ap.parse_args().rater).run()


if __name__ == "__main__":
    main()
