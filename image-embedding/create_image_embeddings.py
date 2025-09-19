"""
Create image embeddings for images in a directory using a pre-trained DINOv3 model.

cd /ccn2/u/khaiaw/Code/babyview-pose/object-detection/image-embedding
conda activate babyview-pose


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

export CUDA_VISIBLE_DEVICES=1
python create_image_embeddings.py \
    --model_name facebook/dinov3-vitb16-pretrain-lvd1689m \
    --input_image_dir /ccn2a/dataset/physion/ \
    --out_dir /ccn2a/dataset/babyview/2025.2/outputs/image_embeddings/physion/ \
    --data_type videos \
    --debug

Notes:
- Some DINOv3 checkpoints are gated on Hugging Face; request access if needed.
- Embeddings are taken from `outputs.pooler_output` (the CLS-pooled vector).
"""

import os
import argparse
import sys
import glob
import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel
import ray

# ------------------------- Args ----------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='facebook/dinov2-base', choices=['facebook/dinov2-base', 'facebook/dinov3-vitb16-pretrain-lvd1689m'],
                        help='Hugging Face model id')
    parser.add_argument('--input_image_dir', type=str, required=True,
                        help='Directory of input images (jpg/png), searched recursively')
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
    return parser.parse_args()

# -------------------- Model / Processor --------------------------------------
def get_model_and_processor(model_name):
    # AutoImageProcessor handles resize/normalize for DINOv3
    processor = AutoImageProcessor.from_pretrained(model_name)
    # Use bfloat16 by default per HF docs (works well on modern GPUs)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    model.eval()
    return processor, model

# ----------------------- Core embedding fn -----------------------------------
@torch.inference_mode()
def create_image_embedding(image_path, out_dir, processor, model, save_dtype='float16'):
    # image_id should be the last two "/" components of the path
    parent = os.path.basename(os.path.dirname(image_path))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    image_id = f"{parent}_{stem}"
    model_tag = model.name_or_path.replace('/', '_')
    out_folder = os.path.join(out_dir, model_tag)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f'{image_id}.npy')

    if os.path.exists(out_path):
        return

    # Load image robustly
    try:
        if image_path.lower().endswith('.mp4'):
            import cv2, random
            cap = cv2.VideoCapture(image_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise ValueError("No frames in video")
            idx = random.randrange(total)
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


    # Preprocess and run model
    inputs = processor(images=img, return_tensors="pt").to(model.device)
    outputs = model(**inputs)

    # CLS-pooled embedding for the whole image
    # (shape: [1, hidden_size]) → [hidden_size]
    pooled = outputs.pooler_output.squeeze(0).to('cpu')

    # Save
    vec = pooled.detach().to(torch.float16).cpu().numpy()
    np.save(out_path, vec)

# ------------------------- Ray worker ----------------------------------------
@ray.remote(num_gpus=0.125)
def process_list_of_images(model_name, image_paths, out_dir, save_dtype):
    processor, model = get_model_and_processor(model_name)
    for p in image_paths:
        try:
            create_image_embedding(p, out_dir, processor, model, save_dtype=save_dtype)
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

if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.data_type == 'images':
        image_files = collect_images(args.input_image_dir)
    elif args.data_type == 'videos':
        image_files = collect_videos(args.input_image_dir)
    print(f'Found {len(image_files)} images/videos in {args.input_image_dir}')
    if len(image_files) == 0:
        sys.exit(0)

    np.random.shuffle(image_files)
    image_files = image_files[:args.max_images]  # limit to first 100k for now

    if args.debug:
        # Single-process debug
        processor, model = get_model_and_processor(args.model_name)
        for p in image_files[:5]:
            create_image_embedding(p, args.out_dir, processor, model, save_dtype=args.save_dtype)
        print('Debug run complete.')
        sys.exit(0)

    # Parallel with Ray
    ray.init(ignore_reinit_error=True)
    try:
        chunk_size = max(1, len(image_files) // args.num_processes + 1)
        chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

        tasks = [
            process_list_of_images.remote(
                args.model_name, chunk, args.out_dir, args.save_dtype
            )
            for chunk in chunks
        ]
        ray.get(tasks)
    finally:
        ray.shutdown()

    print('Done.')
