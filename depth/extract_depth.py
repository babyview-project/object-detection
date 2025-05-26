import glob
import os
import argparse
import random
from transformers import pipeline
from PIL import Image
from matplotlib import pyplot as plt
import ray
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

"""
Extract depth from images using a pre-trained model. Store it in numpy format and visualize the first set of results.
Note: Images are run and stored at 256 resolution (for the shorter side), due to lack of storage space.
So, the portrait images are (455,256)

python extract_depth.py \
    --input_dir /ccn2/dataset/babyview/outputs_20250312/sampled_frames/ \
    --output_dir /ccn2/dataset/babyview/outputs_20250312/depth/4M_frames/ \
    --debug
"""

def get_args():
    parser = argparse.ArgumentParser(description='Process videos to the desired fps, resolution, rotation.')
    parser.add_argument('--input_dir', type=str, help='Path to input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--num_processes', type=int, default=16, help='Number of processes to use')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for processing images')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()

resize_transform_256 = transforms.Resize(256)

def extract_depth(img_path_list, args):
    pipe = pipeline(task="depth-estimation", model="Intel/dpt-large", use_fast=True)

    # Process in batches of args.batch_size
    for global_idx in tqdm(range(0, len(img_path_list), args.batch_size), desc="Processing images"):
        batch_img_paths = img_path_list[global_idx: global_idx + args.batch_size]
        batch_images = [Image.open(img_path) for img_path in batch_img_paths]
        batch_images = [resize_transform_256(img) for img in batch_images]
        results = pipe(batch_images)
        for result, img_path in zip(results, batch_img_paths):
            dir_and_filename = "/".join(img_path.split('/')[-2:])
            out_vis_path = os.path.join(args.output_dir, 'viz', dir_and_filename.replace('/', '_'))
            out_np_path = os.path.join(args.output_dir, 'numpy', dir_and_filename.replace('.jpg', '.npy'))
            os.makedirs(os.path.dirname(out_vis_path), exist_ok=True)
            os.makedirs(os.path.dirname(out_np_path), exist_ok=True)

            # Extract depth prediction, clip, round, and cast to int8 for storage efficiency
            depth = result["predicted_depth"].detach().cpu().numpy()
            depth = np.clip(depth, -120, 120)
            depth = np.round(depth)
            depth = depth.astype(np.int8)
            min_depth, max_depth = depth.min(), depth.max()
            np.save(out_np_path, depth)

        # Save visualization for the first few sets of images
        if global_idx < 5:
            fig, ax = plt.subplots(1, 2, figsize=(6, 5))
            plt.rcParams.update({'font.size': 8})

            # Reload the image for visualization
            image = Image.open(img_path)
            image = resize_transform_256(image)
            ax[0].imshow(image)
            ax[0].set_title("Input Image")

            im = ax[1].imshow(depth, cmap="plasma", vmin=min_depth, vmax=max_depth)
            ax[1].set_title("Predicted Disparity (inverse-depth)")

            cbar = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)

            fig.savefig(out_vis_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        
@ray.remote(num_gpus=0.5)
def extract_depth_remote_babyview(img_path_list, args):
    return extract_depth(img_path_list, args)


if __name__ == "__main__":
    args = get_args()
    print(args)
    
    # Input images
    if args.input_dir:
        img_path_list = glob.glob(os.path.join(args.input_dir, '**/*.jpg'), recursive=True)
        
    random.shuffle(img_path_list)
    if args.debug:
        img_path_list = img_path_list[:64000]
    print('img_path_list:', len(img_path_list))
    
    ray.init(num_cpus=args.num_processes)
    num_chunks = args.num_processes
    chunks = np.array_split(img_path_list, num_chunks)
    
    futures = [extract_depth_remote_babyview.remote(chunk, args) for chunk in chunks]
    ray.get(futures)
    
    
