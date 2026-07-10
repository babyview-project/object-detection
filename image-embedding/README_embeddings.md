# BabyView image embeddings — structure & how-to

Everything produced by [`create_image_embeddings.py`](create_image_embeddings.py): frozen-backbone
image embeddings for several models over three frame sets, plus a spatial `grid4x4` readout for
region-MIL. This doc is the map.

---

## 1. The three frame sets

| set | frames | manifest | output root |
|---|---|---|---|
| **babyview** (128k) | 128,395 | `frame_ids_128395.txt` | `/ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview/` |
| **babyview_877k** | 877,802 | `frame_ids_877802.txt` | `.../image_embeddings/babyview_877k/` |
| **eval (Konkle + LEVANTE)** | 1,020 + 683 | `eval_manifest.csv` | `.../image_embeddings/eval_konkle_mcfrank/` and `.../eval_levante_mcfrank/` |

- **babyview_877k** is the topline set: the 877,802 unique `(video_id, frame_idx)` pairs in
  `grid_baseline_train.parquet` (T1/T2 are subsets, so this union covers all three toplines).
  Use this for anything keyed to `grid_baseline_train`.
- **babyview (128k)** is an *older, different* 128,395-frame subset drawn from the full 5.4M-frame
  `extracted_frames_1fps/` tree. It overlaps the 877k train set by only **20,848 frames (16%)** —
  do **not** use it for train-set-keyed analyses. Kept because it's complete and internally paired.
- **eval** frames live at `/data2/mcfrank/vlm-headcam/eval_frames_mcfrank/`. ⚠️ `/data2` is
  **node-local** — that path exists **only on node14**. Eval jobs must run on node14 (or stage to
  `/ccn2` first).

Source frames for the two babyview sets: `/ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/<video_id>/<frame_idx:05d>.jpg`
(5,419,919 files across 8,566 video dirs).

---

## 2. Models

| model | dir prefix | params | blocks | dim | readout |
|---|---|---|---|---|---|
| DINOv3 ViT-B/16 (off-the-shelf) | `facebook_dinov3-vitb16-pretrain-lvd1689m` | ~86M | — | 768 | CLS |
| DINOv3 ViT-L (BabyView) | `awwkl_dinov3-vitl-babyview` | 303M | 24 | 1024 | CLS |
| V-JEPA2 ViT-L (BabyView) | `awwkl_vjepa2-vitl-fpc16-256-babyview-bs3072-e140` | 304M | 24 | 1024 | patch tokens |
| ZWM 170M (BabyView) | `awwkl_zwm-babyview-170m` | 170M | 24 | 768 | patch tokens |
| ZWM 1B (BabyView) | `awwkl_zwm-babyview-1b` | 947M | 48 | 1280 | patch tokens |
| DINOv2 ViT-B (off-the-shelf) | `facebook_dinov2-base` | ~86M | — | 768 | CLS (128k only, 100k subset) |

Only the DINO models have a **CLS token**. V-JEPA2 and ZWM have no CLS and no pooler, so they are
read out by **pooling patch tokens** at several depths.

---

## 3. Readouts (= directory suffixes)

Every readout is one directory `<model_tag><suffix>/`, holding one `<image_id>.npy` per frame.

| suffix | what | shape | which models |
|---|---|---|---|
| *(none)* | CLS token after the backbone's final LayerNorm (`pooler_output`) | `[D]` | DINO only |
| `_meanpatch` | patch tokens mean-pooled to 1 vector (param-free LayerNorm first) | `[D]` | DINO only |
| `_layer<L>` | patch tokens at 0-indexed block L, mean-pooled to 1 vector | `[D]` | V-JEPA2, ZWM |
| `_grid4x4` | patch tokens adaptive-avg-pooled to a **4×4 grid** = 16 region vectors | `[16, D]` | DINO |
| `_layer<L>_grid4x4` | same, at block L | `[16, D]` | V-JEPA2, ZWM |

**Dtype** is `float16` throughout. `_meanpatch` exists so DINO can be compared apples-to-apples
against the pooler-less models (all mean-pooled).

### Layer indexing
0-indexed block outputs. A 24-block model runs `0..23`, a 48-block model `0..47`. The **last** block
is read out **after the model's own final norm** (ZWM's `ln_f`; for V-JEPA2 `hidden_states[n]` is
bit-identical to `last_hidden_state`). `-1` is accepted on the CLI as an alias for the last block but
directories always carry the real number (`_layer23`, `_layer47`) — never `_layer-1`.

> ⚠️ This differs from the ZWM spelke-seg evals, where `--feature_layer 23` is the *raw* final block
> and `-1` is the normed one. Here `_layer23` is the **normed** one.

### Default layer sweep
Evenly spaced through depth, plus the last block — mirrors the ZWM eval sweep:
`[0,4,8,12,16,20,23]` for 24-block models, `[0,8,16,24,32,40,47]` for ZWM-1B.

---

## 4. grid4x4 & region-MIL — what it's for

The `_layer<L>` / CLS readouts give **one vector per frame**, discarding *where* things are. A frame
labeled "airplane" gets one embedding for the whole scene even though the airplane fills ~1/6 of it.

`grid4x4` keeps the spatial layout. It takes the **same** patch tokens already computed and
adaptive-avg-pools them to a 4×4 grid instead of a single mean:

```
patch grid [Hp × Wp × D]  ──adaptive_avg_pool2d(→4×4)──▶  [4 × 4 × D] = [16, D]
(16×16 for V-JEPA2, 32×32 for ZWM, 14×14 for DINOv3-vitl)      row-major: top row L→R first
```

**Region-MIL** (Multiple-Instance Learning): each image is a *bag* of 16 region instances; the head
is trained with only an image-level label and learns to **attend** to the region containing the
labeled object — no region-level supervision. The MIL input is **17 regions = 16 grid cells +
1 global**. The global vector is your existing `_layer<L>` (or `_meanpatch`) readout; `_grid4x4`
adds the 16 local ones.

Verified: the 16 grid cells average back to the global mean-pool (`cos ≈ 0.999–1.000`), and the grid
is spatially correct row-major (unit-tested on positional data; ZWM confirmed on a brightness probe).

**Which layer gets grid4x4:** on the huge **877k** set, only the single recommended layer per model
(disk — see §7). On the tiny **eval** set, **every** layer.

Recommended readout layer (from a same-video separability probe): ZWM mid-depth (**170M→layer12**,
**1B→layer24**), V-JEPA2 **last (layer23)**, DINO single readout. Rationale: ZWM reconstructs pixels,
so its late blocks drift back toward low-level appearance and mid-depth is most semantic; V-JEPA2
predicts in latent space (`skip_predictor=True`), so its final layer stays most abstract.

---

## 5. Complete directory inventory

`<root>/<model_tag><suffix>/<image_id>.npy`. Base readouts are `[D]`; `_grid4x4` are `[16, D]`.

### babyview/ (128k) — EXISTING, no grid4x4
```
facebook_dinov2-base/                                 (100,000 — DIFFERENT subset, see §7)
facebook_dinov3-vitb16-pretrain-lvd1689m/             (no _meanpatch — original Sept run)
awwkl_dinov3-vitl-babyview/                _meanpatch/
awwkl_vjepa2-...-e140_layer{0,4,8,12,16,20,23}/       (7)
awwkl_zwm-babyview-170m_layer{0,4,8,12,16,20,23}/     (7)
awwkl_zwm-babyview-1b_layer{0,8,16,24,32,40,47}/      (7)
frame_ids_128395.txt
```

### babyview_877k/ — base readouts DONE; grid4x4 to ADD (recommended layer only)
```
facebook_dinov3-vitb16-pretrain-lvd1689m/  _meanpatch/            [+ _grid4x4/  (NEW)]
awwkl_dinov3-vitl-babyview/                _meanpatch/            [+ _grid4x4/  (NEW)]
awwkl_vjepa2-...-e140_layer{0,4,8,12,16,20,23}/                   [+ _layer23_grid4x4/  (NEW)]
awwkl_zwm-babyview-170m_layer{0,4,8,12,16,20,23}/                 [+ _layer12_grid4x4/  (NEW)]
awwkl_zwm-babyview-1b_layer{0,8,16,24,32,40,47}/                  [+ _layer24_grid4x4/  (NEW)]
frame_ids_877802.txt   README.md
```

### eval_konkle_mcfrank/ (1,020) and eval_levante_mcfrank/ (683) — NEW, grid4x4 at ALL layers
Each of the two dirs contains, per model:
```
awwkl_dinov3-vitl-babyview/  _meanpatch/  _grid4x4/
facebook_dinov3-vitb16-pretrain-lvd1689m/  _meanpatch/  _grid4x4/
awwkl_vjepa2-...-e140_layer{0,4,8,12,16,20,23}/  + _layer{...}_grid4x4/   (7 + 7)
awwkl_zwm-babyview-170m_layer{0,4,8,12,16,20,23}/  + _layer{...}_grid4x4/ (7 + 7)
awwkl_zwm-babyview-1b_layer{0,8,16,24,32,40,47}/  + _layer{...}_grid4x4/  (7 + 7)
```

> Two eval dirs because `image_id` **collides across sets**: `ball_00000.jpg` (Konkle) and
> `ball_00000.webp` (LEVANTE) are different images with the same id. Splitting by `set` keeps both.

---

## 6. How to use

### Load one readout
```python
import numpy as np, os
d = ".../babyview_877k/awwkl_zwm-babyview-1b_layer24"
vec  = np.load(f"{d}/{image_id}.npy")            # [1280]        global mean-pool
grid = np.load(f"{d}_grid4x4/{image_id}.npy")    # [16, 1280]    region-MIL cells
mil_bag = np.concatenate([grid, vec[None]], 0)   # [17, 1280]    16 regions + global
```

### ⚠️ Center before RDMs / cosine distances
Mean-pooled patch features are **anisotropic** — cosine between *unrelated* images sits at
~0.8–0.9 (a large shared component), vs ~0.03 for DINOv3 CLS. Subtract the per-dimension mean over
the frame set before computing similarities, or the shared offset dominates. (CLS readouts don't
need this.)

### Run the pipeline
```bash
conda activate ccwm      # NOT babyview-pose (that env can't import zwm)

# babyview_877k, add grid4x4 at the recommended layer (base readouts already exist → skipped):
python create_image_embeddings.py \
  --model_name awwkl/zwm-babyview-1b \
  --input_image_dir /ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/ \
  --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview_877k/ \
  --image_id_file <out>/frame_ids_877802.txt \
  --layers 24 --grid4x4 --grid4x4_layers 24 \
  --gpus_per_worker 0.25 --num_processes 8

# eval set (RUN ON node14 — /data2 is node-local), one run per set, grid4x4 at all layers:
python create_image_embeddings.py \
  --model_name awwkl/zwm-babyview-1b \
  --image_manifest_csv /data2/mcfrank/vlm-headcam/eval_frames_mcfrank/eval_manifest.csv \
  --manifest_set konkle_test \
  --out_dir .../image_embeddings/eval_konkle_mcfrank/ \
  --grid4x4 --gpus_per_worker 0.25 --num_processes 8
```

Key flags: `--image_id_file` (pin the exact frame set; overrides `--max_images`),
`--image_manifest_csv` + `--manifest_set` (drive off explicit id/path rows), `--grid4x4`
(+`--grid4x4_layers` to restrict), `--layers` (which blocks; `-1`=last).

---

## 7. Caveats & gotchas

- **`facebook_dinov2-base` in babyview/ is 100,000 frames, a *different* random subset** (run under
  the old `--max_images 100000` default before the shuffle-then-truncate bug was fixed). It is **not
  paired** with the other models — intersect ids before any per-frame comparison.
- **Anisotropy**: center mean-pooled features before RDMs (see §6).
- **`/data2` is node-local**: `grid_baseline_train.parquet` and the eval frames exist only on
  **node14**. Reads from other nodes silently see an empty mount.
- **Atomic saves**: writes go to `<path>.tmp<pid>` → `fsync` → `os.replace`. Fixes a real bug where a
  killed process left a 0-byte/truncated `.npy` that the skip-check then treated as "done" forever.
  (5 such corpses were found and repaired across the 25M-file sweeps — 2 of them in the *original*
  Sept `facebook_dinov3-vitb16` dir.)
- **Verify by byte size, not just id/count.** A truncated file has the right name and passes an
  id-set check; only checking file sizes catches it.
- **Disk**: grid4x4 is 16× a mean-pool vector. On 877k, at 1 layer/model it's ~137 GB (fits
  `/ccn2a`); at all 7 layers it would be ~657 GB (would not fit) — hence 1 layer on 877k, all layers
  on the tiny eval set.

---

## 8. Code architecture

[`create_image_embeddings.py`](create_image_embeddings.py):

- **Backends** (`HFPooledBackend`, `VJEPA2Backend`, `ZWMBackend`) — one per architecture family.
  Each exposes `.tag`, `.variants` (list of dir suffixes), and `.encode(img) -> {suffix: tensor}`.
  All requested layers + grids come from **one forward pass**.
  - `HFPooledBackend`: `AutoModel`; CLS `pooler_output`, `_meanpatch`, optional `_grid4x4`.
  - `VJEPA2Backend`: encodes a still as a `tubelet_size`-frame clip (→ 1 spatial grid),
    `skip_predictor=True`; reads `hidden_states` at each layer.
  - `ZWMBackend`: loads raw `model.pt` via the ZWM repo (`ZWMPredictor`), all patches visible
    (zeros mask), reads block outputs; last block gets `ln_f`.
- **`grid4x4_pool(toks)`** — `[1,N,D]` layer-normed tokens → `[16,D]`, reshape to `√N×√N`,
  `adaptive_avg_pool2d(4)`, row-major.
- **`create_image_embedding(path, out_dir, backend, image_id=None)`** — skip if all variant files
  exist; else `encode` once and atomic-save each variant. `image_id` overrides the derived id
  (used by the CSV manifest mode).
- **Input modes**: (a) `--input_image_dir` walk + `image_id = <parent>_<stem>`, optionally pinned by
  `--image_id_file`; (b) `--image_manifest_csv` (+`--manifest_set`) with explicit `(image_id, path)`.
- **Parallelism**: Ray, `--num_processes` workers, `--gpus_per_worker` fraction; `RAY_TMPDIR` honored
  (point at `/dev/shm` — cluster root disks are often full).
