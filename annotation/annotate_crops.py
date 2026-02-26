#!/usr/bin/env python3
"""
Simple annotation GUI for cropped object images.
Requires: Pillow (pip install Pillow)

- User selects the image folder (e.g. annotation/sampled_object_crops).
- For each image, category is taken from the parent folder name (first part of path).
- Asks: (1) Is the crop the correct category? (2) If not, is it in the correct CDI semantic category?
- Saves results incrementally to a CSV with columns:
  category_name, correct_detection, correct_cdi, image_filename

Controls:
  y / n     : Answer yes / no to the current question
  p / ←     : Go back to previous image
  n / →     : Go to next image (skip without answering)
  s         : Save annotations to CSV (auto-save every 30 s and on next image)
  + / =     : Zoom in
  -         : Zoom out
"""
from pathlib import Path
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

AUTO_SAVE_INTERVAL_MS = 30_000  # 30 seconds

# Default folder and CSV
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "annotation" / "sampled_object_crops"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def collect_images(root_dir: Path) -> list[tuple[Path, str, str]]:
    """Collect (full_path, category, relative_filename) for all images under root_dir."""
    root_dir = root_dir.resolve()
    out = []
    for ext in IMAGE_EXTENSIONS:
        for path in root_dir.rglob(f"*{ext}"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(root_dir)
            except ValueError:
                continue
            # Category = parent folder name; if image is in root, use first field of filename
            parts = rel.parts
            if len(parts) > 1:
                category = parts[0]
            else:
                stem = path.stem
                category = stem.split("_")[0] if "_" in stem else stem
            rel_str = str(rel).replace("\\", "/")
            out.append((path, category, rel_str))
    out.sort(key=lambda x: (x[1], x[2]))
    return out


def load_existing_annotations(csv_path: Path) -> dict[str, dict]:
    """Load existing CSV into dict keyed by image_filename."""
    result = {}
    if not csv_path.exists():
        return result
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("image_filename", "").strip()
            if fn:
                result[fn] = {
                    "category_name": row.get("category_name", "").strip(),
                    "correct_detection": row.get("correct_detection", "").strip(),
                    "correct_cdi": row.get("correct_cdi", "").strip(),
                    "image_filename": fn,
                }
    return result


def save_annotations(csv_path: Path, annotations: dict[str, dict], image_list: list[tuple]) -> None:
    """Write all annotations for image_list to CSV (merge with existing keys)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_annotations(csv_path)
    for _path, _cat, rel_str in image_list:
        if rel_str in annotations:
            existing[rel_str] = annotations[rel_str]
    rows = [existing[rel_str] for _path, _cat, rel_str in image_list if rel_str in existing]
    # Also include any existing rows for images not in current list (e.g. different run)
    seen = {r["image_filename"] for r in rows}
    for fn, row in existing.items():
        if fn not in seen:
            rows.append(row)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category_name", "correct_detection", "correct_cdi", "image_filename"])
        w.writeheader()
        w.writerows(rows)


class AnnotateApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Crop annotation")
        self.root.geometry("900x700")

        self.image_dir: Path | None = None
        self.csv_path: Path | None = None
        self.image_list: list[tuple[Path, str, str]] = []
        self.index = 0
        self.annotations: dict[str, dict] = {}  # image_filename -> row dict
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 5.0
        self.zoom_step = 0.25
        self.current_photo = None
        self.pil_image = None
        self.question_step = 0  # 0 = show Q1, 1 = show Q2 (only if Q1 was no)
        self._auto_save_after_id: str | None = None

        self._build_ui()
        self._bind_keys()
        self._ask_folder()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Choose image folder", command=self._choose_folder).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Save annotations", command=self._save).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Previous image", command=self._prev).pack(side=tk.LEFT, padx=4)
        self.next_btn = ttk.Button(top, text="Next image", command=self._next)
        self.next_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Zoom in", command=self._zoom_in).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Zoom out", command=self._zoom_out).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Choose an image folder to start.")
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=8)

        # Canvas with scrollbars for image
        frame_canvas = ttk.Frame(self.root, padding=4)
        frame_canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame_canvas, bg="gray20", highlightthickness=0)
        self.hbar = ttk.Scrollbar(frame_canvas, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(frame_canvas, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Question / info area
        self.info_var = tk.StringVar(value="")
        self.info_label = ttk.Label(self.root, textvariable=self.info_var, font=("", 12), padding=8)
        self.info_label.pack(fill=tk.X)

        self.prompt_var = tk.StringVar(value="")
        self.prompt_label = tk.Label(
            self.root,
            textvariable=self.prompt_var,
            font=("", 14, "bold"),
            fg="black",
            anchor=tk.W,
            padx=8,
            pady=8,
        )
        self.prompt_label.pack(fill=tk.X)

        hint = "Keys: Y / N = answer | P or ← = previous | → = next | S = save | + / - = zoom"
        ttk.Label(self.root, text=hint, foreground="gray").pack(pady=4)

    def _bind_keys(self):
        self.root.bind("<KeyPress-y>", lambda e: self._answer(1))
        self.root.bind("<KeyPress-Y>", lambda e: self._answer(1))
        self.root.bind("<KeyPress-n>", lambda e: self._answer(0))
        self.root.bind("<KeyPress-N>", lambda e: self._answer(0))
        self.root.bind("<KeyPress-p>", lambda e: self._prev())
        self.root.bind("<KeyPress-P>", lambda e: self._prev())
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<KeyPress-s>", lambda e: self._save())
        self.root.bind("<KeyPress-S>", lambda e: self._save())
        self.root.bind("<plus>", lambda e: self._zoom_in())
        self.root.bind("<equal>", lambda e: self._zoom_in())
        self.root.bind("<minus>", lambda e: self._zoom_out())

    def _ask_folder(self):
        """Optionally start with default folder or prompt."""
        if DEFAULT_IMAGE_DIR.exists():
            self.image_dir = DEFAULT_IMAGE_DIR
            self.csv_path = DEFAULT_IMAGE_DIR.parent / "annotation_results.csv"
            self._load_list()
        else:
            self.status_var.set("Default folder not found. Click 'Choose image folder'.")

    def _choose_folder(self):
        start = DEFAULT_IMAGE_DIR if DEFAULT_IMAGE_DIR.exists() else PROJECT_ROOT
        path = filedialog.askdirectory(title="Select image folder", initialdir=str(start))
        if not path:
            return
        self.image_dir = Path(path)
        self.csv_path = self.image_dir.parent / "annotation_results.csv"
        self.annotations = load_existing_annotations(self.csv_path)
        self._load_list()

    def _load_list(self):
        if not self.image_dir or not self.image_dir.is_dir():
            messagebox.showerror("Error", "Invalid image folder.")
            return
        self.image_list = collect_images(self.image_dir)
        self.index = 0
        self.status_var.set(f"Loaded {len(self.image_list)} images. CSV: {self.csv_path}")
        if self.image_list:
            self._show_current()
            self._schedule_auto_save()
        else:
            if self._auto_save_after_id is not None:
                self.root.after_cancel(self._auto_save_after_id)
                self._auto_save_after_id = None
            self.next_btn.state(["disabled"])
            self.info_var.set("No images found.")
            self.prompt_var.set("")
            self.canvas.delete("all")

    def _current_entry(self) -> tuple[Path, str, str] | None:
        if not self.image_list or self.index < 0 or self.index >= len(self.image_list):
            return None
        return self.image_list[self.index]

    def _show_current(self):
        entry = self._current_entry()
        if not entry:
            return
        path, category, rel_str = entry
        self.question_step = 0
        existing = self.annotations.get(rel_str)

        # Load and display image
        try:
            self.pil_image = Image.open(path).convert("RGB")
        except Exception as e:
            self.info_var.set(f"Error loading image: {e}")
            self.pil_image = None
        self._draw_image()

        # Progress and category
        self.info_var.set(f"Image {self.index + 1} / {len(self.image_list)}  |  Category: {category}  |  File: {rel_str}")

        if existing is not None:
            d, c = existing.get("correct_detection", ""), existing.get("correct_cdi", "")
            self.prompt_var.set(f"Already annotated: correct_detection={d}, correct_cdi={c}. Press Y/N to re-annotate or P to go back.")
            self.next_btn.state(["!disabled"])
        else:
            self.prompt_var.set("(1) Is this crop the correct category? Press Y or N")
            self.next_btn.state(["disabled"])

    def _draw_image(self):
        self.canvas.delete("all")
        if self.pil_image is None:
            return
        w, h = self.pil_image.size
        if w <= 0 or h <= 0:
            return
        zoomed_w = max(1, int(w * self.zoom))
        zoomed_h = max(1, int(h * self.zoom))
        img = self.pil_image.resize((zoomed_w, zoomed_h), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img)
        # Use a region at least as large as the canvas so we can center the crop
        self.root.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        region_w = max(cw, zoomed_w)
        region_h = max(ch, zoomed_h)
        self.canvas.create_image(region_w // 2, region_h // 2, anchor=tk.CENTER, image=self.current_photo)
        self.canvas.configure(scrollregion=(0, 0, region_w, region_h))
        # Center the view on the image when it's larger than the canvas
        if region_w > cw:
            self.canvas.xview_moveto(0.5)
        if region_h > ch:
            self.canvas.yview_moveto(0.5)

    def _zoom_in(self):
        self.zoom = min(self.max_zoom, self.zoom + self.zoom_step)
        if self.pil_image is not None:
            self._draw_image()
        self.status_var.set(f"Zoom: {self.zoom:.2f}x")

    def _zoom_out(self):
        self.zoom = max(self.min_zoom, self.zoom - self.zoom_step)
        if self.pil_image is not None:
            self._draw_image()
        self.status_var.set(f"Zoom: {self.zoom:.2f}x")

    def _answer(self, yes: int):
        entry = self._current_entry()
        if not entry:
            return
        _path, category, rel_str = entry

        if self.question_step == 0:
            if yes == 1:
                # Correct detection => assume correct CDI
                row = {
                    "category_name": category,
                    "correct_detection": "1",
                    "correct_cdi": "1",
                    "image_filename": rel_str,
                }
                self.annotations[rel_str] = row
                self.prompt_var.set("Recorded: correct_detection=1, correct_cdi=1. Moving to next.")
                self._next()
            else:
                self.question_step = 1
                self.prompt_var.set("(2) Is it in the correct CDI semantic category? Press Y or N")
        else:
            correct_cdi = yes
            row = {
                "category_name": category,
                "correct_detection": "0",
                "correct_cdi": "1" if correct_cdi else "0",
                "image_filename": rel_str,
            }
            self.annotations[rel_str] = row
            self.prompt_var.set(f"Recorded: correct_detection=0, correct_cdi={row['correct_cdi']}. Moving to next.")
            self._next()

    def _next(self):
        if not self.image_list:
            return
        # Require annotation before advancing (skip check when called after _answer)
        entry = self._current_entry()
        if entry:
            _path, _cat, rel_str = entry
            if rel_str not in self.annotations:
                if self.question_step == 0:
                    q = "(1) Is this crop the correct category? Press Y or N"
                else:
                    q = "(2) Is it in the correct CDI semantic category? Press Y or N"
                self.prompt_var.set(f"{q} — Please answer before going to next.")
                return
        self.index += 1
        if self.index >= len(self.image_list):
            self.status_var.set("Done with all images. You can go back (P) or save (S).")
            self.index = len(self.image_list) - 1
            return
        self._auto_save()
        self._show_current()

    def _prev(self):
        if not self.image_list:
            return
        self.index = max(0, self.index - 1)
        self._show_current()

    def _auto_save(self) -> bool:
        """Save to CSV without dialog. Returns True if saved, False if skipped or error."""
        if not self.csv_path or not self.image_list:
            return False
        try:
            save_annotations(self.csv_path, self.annotations, self.image_list)
            self.status_var.set(f"Auto-saved at {datetime.now().strftime('%H:%M:%S')}")
            return True
        except Exception:
            return False

    def _schedule_auto_save(self) -> None:
        """Schedule the next auto-save in AUTO_SAVE_INTERVAL_MS."""
        if self._auto_save_after_id is not None:
            self.root.after_cancel(self._auto_save_after_id)
        if self.image_list and self.csv_path:
            self._auto_save_after_id = self.root.after(
                AUTO_SAVE_INTERVAL_MS,
                self._run_scheduled_auto_save,
            )

    def _run_scheduled_auto_save(self) -> None:
        self._auto_save_after_id = None
        self._auto_save()
        self._schedule_auto_save()

    def _save(self):
        if not self.csv_path or not self.image_list:
            messagebox.showinfo("Save", "No folder loaded or no images.")
            return
        try:
            save_annotations(self.csv_path, self.annotations, self.image_list)
            messagebox.showinfo("Saved", f"Annotations saved to:\n{self.csv_path}")
            self.status_var.set(f"Saved to {self.csv_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


def main():
    app = AnnotateApp()
    app.run()


if __name__ == "__main__":
    main()
