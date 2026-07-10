"""
Create image embeddings for images in a directory using a pre-trained DINOv3 model.

cd /ccn2/u/khaiaw/Code/babyview-pose/object-detection/image-embedding
conda activate babyview-pose     # facebook/* models only; BabyView models need `ccwm` (see below)


    --input_image_dir /ccn2a/dataset/babyview/2025.2/extracted_frames_1fps/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/babyview/ \

    --input_image_dir /ccn2/dataset/imagenet/test/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/imagenet_test/ \


    --input_image_dir /ccn2/dataset/kinetics400/Kinetics400/k400/train/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/kinetics400_train/ \


    --input_image_dir /ccn2/dataset/SAYCam/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/SAYCam/ \

    --input_image_dir /ccn2/dataset/ego4D/v1/chunked_resized/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/ego4D/ \

    --input_image_dir /ccn2/dataset/Moments/Moments_in_Time_Raw/training/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/Moments_in_Time_Raw_training/ \

    --input_image_dir /data2/klemenk/ssv2/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/ssv2/ \

    --input_image_dir /ccn2a/dataset/physion/ \

    --input_image_dir /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_all_cropped_by_class \
    --out_dir /data2/dataset/babyview/868_hours/outputs/yoloe_cdi_embeddings/ \
        
export CUDA_VISIBLE_DEVICES=6,7
python create_image_embeddings.py \
    --model_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --input_image_dir /ccn2/dataset/babyview/outputs_20250312/things_bv_overlapping_categories_corrected \
    --out_dir /ccn2/dataset/babyview/outputs_20250312/image_embeddings/things_bv_overlapping_categories_corrected \
    --data_type images \
    --max_images 987654321 \
    --debug

BabyView-trained models (run these from the `ccwm` env, NOT `babyview-pose`, which
cannot import `zwm` -- it is missing moviepy):

conda activate ccwm
export CUDA_VISIBLE_DEVICES=0,1,2,3
BV=/ccn2a/dataset/babyview/2025.2
python create_image_embeddings.py \
    --model_name awwkl/dinov3-vitl-babyview \
    --input_image_dir $BV/extracted_frames_1fps/ \
    --out_dir $BV/outputs/image_embeddings/babyview/ \
    --data_type images --max_images 987654321

# same, with --model_name awwkl/vjepa2-vitl-fpc16-256-babyview-bs3072-e140
# and    with --model_name awwkl/zwm-babyview-1b --gpus_per_worker 0.25
#   (ZWM-1b is 947M params; it needs a bigger GPU slice than the 0.125 default.
#    Fetch the checkpoint first: see ZWMBackend.)

Notes:
- Some DINOv3 checkpoints are gated on Hugging Face; request access if needed.
- Readouts differ by architecture, because only DINOv3 has a CLS token:
    DINOv3/DINOv2 -> `outputs.pooler_output`, i.e. the CLS token after the backbone's
                     own final LayerNorm. One vector, written to <model_tag>/.
    V-JEPA2, ZWM  -> no CLS and no pooler, so patch tokens are mean-pooled at several
                     depths, each written to a sibling <model_tag>_layer<L>/ directory.
                     Layers are 0-indexed block outputs, so a 24-block model runs 0..23 and
                     a 48-block model 0..47. -1 is accepted on the CLI as an alias for the
                     last block, but directories always carry the real number (_layer23,
                     _layer47) -- never _layer-1.
                     The LAST block is read out after the model's own final norm (ZWM's
                     ln_f; for V-JEPA2 hidden_states[n] is bit-identical to
                     last_hidden_state, so there is nothing extra to apply). NOTE this
                     differs from the ZWM spelke-seg evals, where `23` means the RAW final
                     block and `-1` means the normed one; here _layer23 is the normed one.
- Mean-pooled patch features are anisotropic (cosine similarity between unrelated images
  sits around 0.8-1.0), so center them before computing RDMs or distances.
- The multi-layer backends get all their layers from ONE forward pass, so extra layers
  cost disk, not GPU time.
"""

import os
import argparse
import sys
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import ray

ZWM_REPO_ROOT = '/ccn2/u/khaiaw/Code/ZWM'

# Models whose embedding is the CLS-pooled `pooler_output` (a single vector).
HF_POOLED_MODELS = [
    'facebook/dinov2-base',
    'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'awwkl/dinov3-vitl-babyview',
]
# Models with no pooler: mean-pool patch tokens at several depths instead.
VJEPA2_MODELS = ['awwkl/vjepa2-vitl-fpc16-256-babyview-bs3072-e140']
ZWM_MODELS = ['awwkl/zwm-babyview-170m', 'awwkl/zwm-babyview-1b']

# ------------------------- Args ----------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='facebook/dinov2-base',
                        choices=HF_POOLED_MODELS + VJEPA2_MODELS + ZWM_MODELS,
                        help='Hugging Face model id')
    parser.add_argument('--input_image_dir', type=str, default=None,
                        help='Directory of input images (jpg/png), searched recursively. '
                             'Not needed (and ignored) when --image_manifest_csv is given.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save output .npy embeddings')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of parallel workers (Ray tasks)')
    parser.add_argument('--debug', action='store_true',
                        help='If true, process only a few images')
    parser.add_argument('--save_dtype', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='dtype used when saving embeddings')
    parser.add_argument('--data_type', type=str, default='images', choices=['images', 'videos'],
                        help='Type of input data (images or videos)')
    parser.add_argument('--max_images', type=int, default=100000,
                        help='Maximum number of images to process (for testing)')
    parser.add_argument('--image_id_file', type=str, default=None,
                        help='Text file of image_ids (one per line, "<parent>_<stem>"). Restrict '
                             'this run to exactly those frames, so every model embeds the same set. '
                             'Needed because --max_images alone would subsample a DIFFERENT random '
                             'set each run (the shuffle is unseeded). See frame_ids_128395.txt.')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='For pooler-less models (V-JEPA2, ZWM): which transformer blocks to '
                             'read out. 0-indexed; -1 means the final block after the model\'s own '
                             'final LayerNorm. Matches the --feature_layer convention of the ZWM '
                             'spelke-seg evals. Each layer is written to its own sibling '
                             '<model_tag>_layer<L>/ directory. Defaults to blocks evenly spaced '
                             'through depth plus the last one. Ignored for pooled models.')
    parser.add_argument('--gpus_per_worker', type=float, default=0.125,
                        help='GPU fraction per Ray worker. ZWM-1b needs a bigger slice (0.25); '
                             'each worker torch.loads a 3.8GB checkpoint into CPU RAM too.')
    parser.add_argument('--grid4x4', action='store_true',
                        help='Also emit a 4x4 region grid: adaptive-avg-pool the patch tokens to '
                             '4x4 (16 vectors) instead of mean-pooling to 1. Enables the region-MIL '
                             'head (16 grid cells + the existing global mean = 17 regions). Each '
                             'file becomes [16, D] instead of [D]. For pooled models -> _grid4x4/; '
                             'for multi-layer models -> _layer<L>_grid4x4/.')
    parser.add_argument('--grid4x4_layers', type=int, nargs='+', default=None,
                        help='For multi-layer models, restrict --grid4x4 to these layers (a subset '
                             'of --layers; any not already requested are added). Default: every '
                             'layer in --layers. Ignored for pooled (DINO) models.')
    parser.add_argument('--image_manifest_csv', type=str, default=None,
                        help='CSV with columns image_id,path[,set]. Use image_id and path straight '
                             'from the CSV instead of deriving the id from the path. Needed when '
                             'ids do not follow <parent>_<stem> (e.g. the eval set, where files sit '
                             'flat in images/ but ids are airplane_00000). Overrides '
                             '--input_image_dir / --image_id_file.')
    parser.add_argument('--manifest_set', type=str, default=None,
                        help='With --image_manifest_csv, keep only rows whose "set" column equals '
                             'this. Needed because image_ids collide across sets (e.g. ball_00000 '
                             'in both konkle_test and levante_vocab) -- run once per set into its '
                             'own --out_dir so nothing overwrites.')
    return parser.parse_args()

def grid4x4_pool(toks):
    """toks: [1, N, D] (already layer-normed) row-major patch tokens, N a perfect square.
    Returns [16, D]: the patch grid adaptive-avg-pooled to 4x4, row-major (top row L->R first).
    Parallels the mean-pool readout (mean of the same layer-normed tokens); grid keeps location."""
    B, N, D = toks.shape
    g = int(round(N ** 0.5))
    if g * g != N:
        raise RuntimeError(f'patch count {N} is not a perfect square; cannot form a grid')
    x = toks.reshape(B, g, g, D).permute(0, 3, 1, 2)        # [1, D, g, g]
    x = F.adaptive_avg_pool2d(x, 4)                          # [1, D, 4, 4]
    return x.reshape(B, D, 16).permute(0, 2, 1).squeeze(0)   # [16, D]

def resolve_grid_layers(grid4x4, grid4x4_layers, layers, n_blocks):
    """Which layers additionally get a _grid4x4 readout. Returns (grid_set, layers_union).

    grid layers must be computed, so any grid layer not already in `layers` is unioned in.
    """
    if not grid4x4:
        return set(), layers
    if not grid4x4_layers:
        return set(layers), layers
    bad = [l for l in grid4x4_layers if not -1 <= l < n_blocks]
    if bad:
        raise ValueError(f'--grid4x4_layers {bad} out of range for a {n_blocks}-block model')
    gl = {n_blocks - 1 if l == -1 else l for l in grid4x4_layers}
    return gl, sorted(set(layers) | gl)

# Evenly spaced through depth, plus the final block. Mirrors the LAYERS sweep in the ZWM
# spelke-seg evals (0 4 8 12 16 20 23 for 24 blocks; 0 8 16 24 32 40 47 for 48).
DEFAULT_LAYER_STRIDE_FRAC = 1 / 6

def default_layers(n_blocks):
    """Evenly spaced 0-indexed blocks through the model, always including the last one.

    All layers come from a single forward pass, so extra layers cost disk, not GPU time.
    """
    step = max(1, round(n_blocks * DEFAULT_LAYER_STRIDE_FRAC))
    return sorted({*range(0, n_blocks, step), n_blocks - 1})

def resolve_layers(layers, n_blocks):
    """Validate --layers against model depth, or fall back to the defaults.

    Layers are 0-indexed block outputs. -1 is accepted as an alias for the last block and
    is normalized to n_blocks-1, so output dirs always carry the real layer number.

    An out-of-range layer is fatal: it would never be written, so the all-variants-exist
    skip check could never pass and every rerun would recompute every image.
    """
    if not layers:
        return default_layers(n_blocks)
    bad = [l for l in layers if not -1 <= l < n_blocks]
    if bad:
        raise ValueError(f'--layers {bad} out of range for a {n_blocks}-block model '
                         f'(expected -1 or 0..{n_blocks - 1})')
    layers = [n_blocks - 1 if l == -1 else l for l in layers]
    return sorted(dict.fromkeys(layers))

# -------------------- Backends -----------------------------------------------
# A backend turns one PIL image into {variant_suffix: 1-D embedding}. The suffix is
# appended to the model tag to form the output directory, so a pooled model writes
# <model_tag>/ and a multi-layer model writes <model_tag>_layer12/, _layer24/, ...

class HFPooledBackend:
    """DINOv2 / DINOv3: embedding is outputs.pooler_output (the CLS-pooled vector).

    Also saves a `_meanpatch` variant --- patch tokens mean-pooled the same way as the
    pooler-less backends --- so DINOv3 can be compared apples-to-apples against V-JEPA2
    and ZWM, which have no CLS token. Both come from the same forward pass.
    """

    def __init__(self, model_name, grid4x4=False):
        # AutoImageProcessor handles resize/normalize for DINOv3
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # Use bfloat16 by default per HF docs (works well on modern GPUs). Each Ray worker
        # sees exactly one GPU, so an explicit .to('cuda') beats device_map="auto" (which
        # would resolve to the same device but drags in an `accelerate` dependency).
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa"
        ).to('cuda')
        self.model.eval()
        self.patch_size = self.model.config.patch_size
        self.grid4x4 = grid4x4
        self.tag = model_name.replace('/', '_')
        self.variants = ['', '_meanpatch'] + (['_grid4x4'] if grid4x4 else [])

    def encode(self, img):
        inputs = self.processor(images=img, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)

        # Drop the prefix tokens (CLS, plus register tokens on some checkpoints: 4 on
        # facebook/dinov3-vitb16, 0 on dinov3-vitl-babyview) to leave only patch tokens.
        H, W = inputs['pixel_values'].shape[-2:]
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        toks = outputs.last_hidden_state
        prefix = toks.shape[1] - num_patches
        if not 0 <= prefix < 32:
            raise RuntimeError(f'token mismatch: got {toks.shape[1]}, expected {num_patches}')
        patches = toks[:, prefix:, :]
        # Same readout as the pooler-less backends: parameter-free LayerNorm, mean over patches.
        patches = F.layer_norm(patches.float(), (patches.shape[-1],))

        out = {
            # (shape: [1, hidden_size]) → [hidden_size]
            '': outputs.pooler_output.squeeze(0),
            '_meanpatch': patches.mean(dim=1).squeeze(0),
        }
        if self.grid4x4:
            out['_grid4x4'] = grid4x4_pool(patches)   # [16, D]
        return out


class VJEPA2Backend:
    """V-JEPA2 is a video encoder with no pooler.

    A still image is encoded as a `tubelet_size`-frame clip, so the single temporal tubelet
    collapses to one purely spatial patch grid --- the same trick as the paper's baseline in
    ZWM/scripts/eval/segments/baselines/eval_spelke_seg_vjepa2.py. (Tiling to the full
    frames_per_clip=16 would give 8 identical tubelets and cost 8x the compute for the same
    information.) Patch tokens are then mean-pooled at each requested block.
    """

    def __init__(self, model_name, layers=None, grid4x4=False, grid4x4_layers=None):
        from transformers import AutoVideoProcessor, VJEPA2Model
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = VJEPA2Model.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to('cuda').eval()
        cfg = self.model.config
        self.n_blocks = cfg.num_hidden_layers
        self.num_frames = cfg.tubelet_size          # -> t_feat = 1 (purely spatial)
        self.patch_size = cfg.patch_size
        self.layers = resolve_layers(layers, self.n_blocks)
        self.grid_layers, self.layers = resolve_grid_layers(
            grid4x4, grid4x4_layers, self.layers, self.n_blocks)
        self.tag = model_name.replace('/', '_')
        self.variants = [f'_layer{l}' for l in self.layers] + \
                        [f'_layer{l}_grid4x4' for l in self.layers if l in self.grid_layers]

    def encode(self, img):
        inputs = self.processor(img, return_tensors='pt')
        pv = inputs['pixel_values_videos']
        if pv.ndim == 4:
            pv = pv.unsqueeze(1)                    # [B, 1, C, H, W]
        pv = pv.to(self.model.device)

        B, T0, C, H, W = pv.shape
        if T0 != self.num_frames:
            reps = (self.num_frames + T0 - 1) // T0
            pv = pv.repeat(1, reps, 1, 1, 1)[:, :self.num_frames]
        expected = (H // self.patch_size) * (W // self.patch_size)   # t_feat == 1

        outputs = self.model(pixel_values_videos=pv, output_hidden_states=True,
                             skip_predictor=True, return_dict=True)

        out = {}
        for l in self.layers:
            # hidden_states[0] is the patch embedding, so 0-indexed block l is at index l+1.
            # For the last block that tensor IS last_hidden_state (verified bit-identical).
            toks = outputs.hidden_states[l + 1]
            if toks.shape[1] != expected:           # drop any prefix (CLS/register-like)
                prefix = toks.shape[1] - expected
                if not 0 < prefix < 32:
                    raise RuntimeError(f'token mismatch: got {toks.shape[1]}, expected {expected}')
                toks = toks[:, prefix:, :]
            # Parameter-free LayerNorm so depths are comparable (matches the ZWM evals).
            toks = F.layer_norm(toks.float(), (toks.shape[-1],))
            out[f'_layer{l}'] = toks.mean(dim=1).squeeze(0)          # mean over patch tokens
            if l in self.grid_layers:
                out[f'_layer{l}_grid4x4'] = grid4x4_pool(toks)      # [16, D]
        return out


class ZWMBackend:
    """ZWM is a pixel-reconstruction transformer: no CLS token, no pooler, and it loads
    from a raw model.pt via the ZWM repo rather than AutoModel.

    Each frame is encoded independently with all patches visible (mask all zeros), which
    is the frozen-backbone analogue used by the paper's own feature probe --- see
    ZWM/scripts/eval/segments/supplementary/eval_spelke_seg_feature_nn.py. Patch tokens
    are then mean-pooled at each requested block."""

    def __init__(self, model_name, layers=None, grid4x4=False, grid4x4_layers=None, device='cuda'):
        if ZWM_REPO_ROOT not in sys.path:
            sys.path.insert(0, ZWM_REPO_ROOT)
        from zwm.zwm_predictor import ZWMPredictor
        from zwm.data.image_processing import patchify_image
        from zwm.data.sequence_construction import get_pos_idxs

        self._patchify_image = patchify_image
        self._get_pos_idxs = get_pos_idxs

        ckpt = os.path.join(ZWM_REPO_ROOT, 'out', model_name, 'model.pt')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f'ZWM checkpoint not found at {ckpt}. Fetch it first with:\n'
                f'    cd {ZWM_REPO_ROOT} && python scripts/hf_model_download.py {model_name}'
            )
        self.predictor = ZWMPredictor(ckpt, device=device)
        cfg = self.predictor.model.config
        self.num_patches = (cfg.resolution // cfg.patch_size) ** 2
        self.n_blocks = len(self.predictor.model.transformer.h)
        self.layers = resolve_layers(layers, self.n_blocks)
        self.grid_layers, self.layers = resolve_grid_layers(
            grid4x4, grid4x4_layers, self.layers, self.n_blocks)
        self.tag = model_name.replace('/', '_')
        self.variants = [f'_layer{l}' for l in self.layers] + \
                        [f'_layer{l}_grid4x4' for l in self.layers if l in self.grid_layers]

    def _frame_to_patches(self, img):
        cfg = self.predictor.model.config
        pil = self.predictor.resize_crop_transform(img, cfg.resolution)
        arr = self.predictor.in_transform(pil).unsqueeze(0).permute(0, 2, 3, 1)
        return self._patchify_image(arr, cfg.patch_size)

    def encode(self, img):
        model = self.predictor.model
        device = self.predictor.device

        patches = self._frame_to_patches(img)
        seq = patches.reshape(1, self.num_patches, -1).to(device)
        pos = self._get_pos_idxs(patches, 0).reshape(1, -1).long().to(device)
        mask = torch.zeros((1, self.num_patches), device=device)

        blocks = set(self.layers)
        last_needed = max(blocks)

        out = {}
        with self.predictor.ctx:
            # mask is all zeros, so the mask-token term vanishes; embed + position.
            h = model.transformer.token_embedding(seq) + model.transformer.positional_embedding(pos)
            h = model.transformer.drop(h)
            for i, block in enumerate(model.transformer.h):
                h = block(h, mask=mask)
                if i in blocks:
                    # The last block is read out after the model's own final norm, matching
                    # V-JEPA2's last_hidden_state (and the ZWM eval's `-1`).
                    feats = model.transformer.ln_f(h) if i == self.n_blocks - 1 else h
                    toks = F.layer_norm(feats.float(), (feats.shape[-1],))
                    out[f'_layer{i}'] = toks.mean(dim=1).squeeze(0)   # mean over patch tokens
                    if i in self.grid_layers:
                        out[f'_layer{i}_grid4x4'] = grid4x4_pool(toks)  # [16, D]
                if i == last_needed:
                    break
        return out


def build_backend(model_name, layers=None, grid4x4=False, grid4x4_layers=None):
    if model_name in HF_POOLED_MODELS:
        return HFPooledBackend(model_name, grid4x4=grid4x4)
    if model_name in VJEPA2_MODELS:
        return VJEPA2Backend(model_name, layers, grid4x4=grid4x4, grid4x4_layers=grid4x4_layers)
    if model_name in ZWM_MODELS:
        return ZWMBackend(model_name, layers, grid4x4=grid4x4, grid4x4_layers=grid4x4_layers)
    raise ValueError(f'Unknown model: {model_name}')

# ----------------------- Core embedding fn -----------------------------------
def image_id_of(image_path):
    """Output filename stem: the last two "/" components of the path, joined by "_"."""
    parent = os.path.basename(os.path.dirname(image_path))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return f"{parent}_{stem}"

@torch.inference_mode()
def create_image_embedding(image_path, out_dir, backend, save_dtype='float16', image_id=None):
    if image_id is None:
        image_id = image_id_of(image_path)

    # One output dir per variant: <model_tag>/ for pooled models, and a sibling
    # <model_tag>_layer<L>/ per layer for the multi-layer ones.
    out_paths = {}
    for suffix in backend.variants:
        out_folder = os.path.join(out_dir, backend.tag + suffix)
        os.makedirs(out_folder, exist_ok=True)
        out_paths[suffix] = os.path.join(out_folder, f'{image_id}.npy')

    # change this so it is out_dir/parent/stem.npy
    # out_folder = os.path.join(out_dir, model_tag, parent)
    # os.makedirs(out_folder, exist_ok=True)
    # out_path = os.path.join(out_folder, f'{stem}.npy')

    if all(os.path.exists(p) for p in out_paths.values()):
        return

    # Load image robustly
    try:
        if image_path.lower().endswith('.mp4'):
            import cv2, random
            cap = cv2.VideoCapture(image_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise ValueError("No frames in video")
            # Seed off the path, not the clock: a resumed run that fills in a missing
            # variant must re-pick the SAME frame, or one video's layers would end up
            # describing different frames.
            idx = random.Random(image_path).randrange(total)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ValueError(f"Failed to read frame {idx}")
            # cv2 loads as BGR; convert to RGB and wrap as PIL.Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(image_path)

        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception as e:
        print(f'[WARN] Could not open {image_path}: {e}')
        return


    # Preprocess and run the model. One forward pass yields every requested variant.
    embeddings = backend.encode(img)

    # Save one .npy per variant, atomically: np.save straight to the final path leaves a
    # truncated (or 0-byte) file if the process dies mid-write, and the skip check above
    # would then treat that corpse as "done" forever. Write to a temp file in the same
    # directory, then os.replace (atomic within a filesystem).
    torch_dtype = torch.float16 if save_dtype == 'float16' else torch.float32
    for suffix, vec in embeddings.items():
        out_path = out_paths[suffix]
        if os.path.exists(out_path):
            continue
        arr = vec.detach().to(torch_dtype).cpu().numpy()
        tmp = f'{out_path}.tmp{os.getpid()}'
        try:
            with open(tmp, 'wb') as fh:
                np.save(fh, arr)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, out_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

# ------------------------- Ray worker ----------------------------------------
@ray.remote(max_retries=-1)
def process_list_of_images(model_name, items, out_dir, save_dtype, layers,
                           grid4x4=False, grid4x4_layers=None):
    # items: list of (path, image_id). image_id is None to derive it from the path.
    backend = build_backend(model_name, layers, grid4x4=grid4x4, grid4x4_layers=grid4x4_layers)
    for p, iid in items:
        try:
            create_image_embedding(p, out_dir, backend, save_dtype=save_dtype, image_id=iid)
        except Exception as e:
            print(f'[ERROR] {p}: {e}')

# --------------------------- Main --------------------------------------------
def collect_images(root):
    exts = {'.jpg', '.jpeg', '.png', '.mp4', '.JPEG', '.JPG', '.PNG', '.MP4'}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1] in exts:
                files.append(os.path.join(dirpath, fname))
    return files

def collect_videos(root):
    files = []
    for fname in os.listdir(root):
        if fname.lower().endswith('.mp4'):
            files.append(os.path.join(root, fname))
            
    if len(files) == 0:
        # do os.walk
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith('.mp4'):
                    files.append(os.path.join(dirpath, fname))
    return files

def load_manifest_items(csv_path, set_filter=None):
    """Read image_id,path[,set] rows -> list of (path, image_id). Ids come straight from
    the CSV, so files needn't follow the <parent>_<stem> naming rule. If set_filter is given,
    keep only rows whose 'set' column matches (ids collide across sets, so each set is a run)."""
    import csv
    items = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if set_filter is not None and row.get('set') != set_filter:
                continue
            items.append((row['path'], row['image_id']))
    if not items:
        raise SystemExit(f'[FATAL] no rows in {csv_path}' +
                         (f' with set={set_filter}' if set_filter else ''))
    # ids must be unique or files overwrite each other
    ids = [i for _, i in items]
    if len(set(ids)) != len(ids):
        import collections
        dup = [k for k, c in collections.Counter(ids).items() if c > 1]
        raise SystemExit(f'[FATAL] {len(dup)} duplicate image_ids (e.g. {dup[:3]}); '
                         f'pass --manifest_set to split by set')
    missing = [p for p, _ in items if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'[FATAL] {len(missing)} of {len(items)} manifest paths do not exist '
                         f'(first: {missing[0]})')
    return items

if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.image_manifest_csv:
        # Explicit (image_id, path) pairs; no id derivation, no --input_image_dir walk.
        items = load_manifest_items(args.image_manifest_csv, set_filter=args.manifest_set)
        print(f'Loaded {len(items)} (image_id, path) rows from {args.image_manifest_csv}' +
              (f' [set={args.manifest_set}]' if args.manifest_set else ''))
    else:
        if not args.input_image_dir:
            raise SystemExit('[FATAL] need --input_image_dir (or --image_manifest_csv)')
        if args.data_type == 'images':
            image_files = collect_images(args.input_image_dir)
        elif args.data_type == 'videos':
            image_files = collect_videos(args.input_image_dir)
        print(f'Found {len(image_files)} images/videos in {args.input_image_dir}')
        if len(image_files) == 0:
            sys.exit(0)

        if args.image_id_file:
            with open(args.image_id_file) as fh:
                want = {line.strip() for line in fh if line.strip()}
            before = len(image_files)
            image_files = [p for p in image_files if image_id_of(p) in want]
            print(f'Restricted {before} -> {len(image_files)} images '
                  f'matching {len(want)} ids from {args.image_id_file}')
            missing = len(want) - len(image_files)
            if missing:
                # Fail loudly: a short set would break paired cross-model comparisons.
                raise SystemExit(f'[FATAL] {missing} of {len(want)} ids have no matching source image')

        # id=None -> derived from path inside create_image_embedding
        items = [(p, None) for p in image_files]

    # Shuffle for load balance across Ray chunks (adjacent frames of one video are similar).
    np.random.shuffle(items)
    if args.image_id_file or args.image_manifest_csv:
        # The manifest IS the set. Truncating here would silently drop frames (the default
        # --max_images is smaller than the manifest), breaking paired cross-model comparison.
        if args.max_images < len(items):
            print(f'[INFO] ignoring --max_images {args.max_images}; manifest pins {len(items)} images')
    else:
        items = items[:args.max_images]

    if args.debug:
        # Single-process debug
        backend = build_backend(args.model_name, args.layers,
                                grid4x4=args.grid4x4, grid4x4_layers=args.grid4x4_layers)
        print(f'Backend {type(backend).__name__} tag={backend.tag} variants={backend.variants}')
        for p, iid in items[:5]:
            create_image_embedding(p, args.out_dir, backend, save_dtype=args.save_dtype, image_id=iid)
        print('Debug run complete.')
        sys.exit(0)

    # Parallel with Ray. Ray defaults its temp dir to /tmp/ray, and the cluster's root disk
    # is frequently 100% full -- which shows up as a silent mid-run hang, not a crash. Honor
    # RAY_TMPDIR (point it at /dev/shm) when set.
    ray_tmp = os.environ.get('RAY_TMPDIR')
    ray.init(ignore_reinit_error=True, _temp_dir=ray_tmp if ray_tmp else None)
    try:
        chunk_size = max(1, len(items) // args.num_processes + 1)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        worker = process_list_of_images.options(num_gpus=args.gpus_per_worker)
        tasks = [
            worker.remote(
                args.model_name, chunk, args.out_dir, args.save_dtype, args.layers,
                args.grid4x4, args.grid4x4_layers
            )
            for chunk in chunks
        ]
        ray.get(tasks)
    finally:
        ray.shutdown()

    print('Done.')
