import glob
import os
import argparse
import random
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

"""
Visualize extracted depth from images stored in numpy format to verify it works okay.

python visualize_extracted_depth.py
"""

n_samples_to_visualize = 20
image_dir = '/ccn2/dataset/babyview/outputs_20250312/sampled_frames/'
depth_dir = '/ccn2/dataset/babyview/outputs_20250312/depth/4M_frames/'
viz_dir = '/ccn2/u/khaiaw/Code/babyview-pose/depth/viz_extracted_depth'

resize_transform_256 = transforms.Resize(256)

# === Begin visualizations ===
depth_files = glob.glob(os.path.join(depth_dir, '**/*.npy'), recursive=True)
print(f"Found {len(depth_files)} depth files in {depth_dir}")
random.shuffle(depth_files)
depth_files = depth_files[:n_samples_to_visualize]
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
    
for depth_path in tqdm(depth_files, desc="Visualizing depth files"):
    # Get the corresponding image file
    depth = np.load(depth_path)
    print(f"Visualizing depth: {depth.shape}")
    min_depth, max_depth = np.min(depth), np.max(depth)
    
    img_path = depth_path.replace(depth_dir, image_dir).replace('numpy', '').replace('.npy', '.jpg')
    out_vis_path = os.path.join(viz_dir, os.path.basename(depth_path).replace('.npy', '.jpg'))
    
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
