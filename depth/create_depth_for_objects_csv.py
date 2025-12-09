import glob
import os
import argparse
import random
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import ray
from tqdm import tqdm
import numpy as np
from torchvision import transforms

"""
TODO: This code is a work-in-progress
Detected objects are stored in a CSV (bounding_box_predictions.csv). For each detected object, get the depth of the image it belongs to.
Then create a CSV in the same directory as the bounding_box_predictions.csv and save detected_objects_depth.csv

cd /ccn2/u/khaiaw/Code/babyview-pose/object-detection/depth

python create_depth_for_objects_csv.py \
    --object_detections_dir /ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_allframes_1fps \
    --depth_dir /ccn2/dataset/babyview/outputs_20250312/depth/4M_frames/ \
    --viz_dir ./viz_detected_objects_depth/ \
    --debug
"""

resize_transform_256 = transforms.Resize(256)

def get_args():
    parser = argparse.ArgumentParser(description='Extract depth from images using a pre-trained model')
    parser.add_argument('--object_detections_dir', type=str, help='Path to input directory')
    parser.add_argument('--depth_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--viz_dir', type=str, default='./viz_detected_objects_depth/', help='Directory to save visualizations')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()

def add_visualize_dot(image, x, y, color='red'):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image

def create_detected_objects_depth_csv(detected_objects_csv, args):
    # TODO: Divide x, y values by 2, because depth is stored at smaller resolution
    # read the csv
    df = pd.read_csv(detected_objects_csv)
    for idx, row in df.iterrows():
        try:
            original_frame_path = row['original_frame_path']
            dir_and_filename = "/".join(original_frame_path.split('/')[-2:]) # get last two parts of the path
            depth_path = os.path.join(args.depth_dir, 'numpy', dir_and_filename.replace('.jpg', '.npy'))
            depth = np.load(depth_path)
            min_depth, max_depth = np.min(depth), np.max(depth)
            
            x_center = (row['xmin'] + row['xmax']) / 2
            y_center = (row['ymin'] + row['ymax']) / 2
            # divide by 2 again because depth is stored at smaller resolution
            x_center = int(x_center / 2)
            y_center = int(y_center / 2)
            if np.isnan(x_center) or np.isnan(y_center): 
                print('nan val detected'); continue
            depth_value = depth[y_center, x_center]
            
            # visualize
            if idx == 0:
                fig, ax = plt.subplots(1, 2, figsize=(6, 5))
                out_vis_path = os.path.join(args.viz_dir, dir_and_filename.replace('/', '_').replace('.jpg', '.png'))
                os.makedirs(os.path.dirname(out_vis_path), exist_ok=True)
                
                image = Image.open(original_frame_path)
                image = resize_transform_256(image)
                image = add_visualize_dot(image, x_center, y_center, color='red')
                ax[0].imshow(image)
                ax[0].set_title("Input Image")
                
                im = ax[1].imshow(depth, cmap="plasma", vmin=min_depth, vmax=max_depth)
                ax[1].set_title("Predicted Disparity (inverse-depth)")
                cbar = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
                
                fig.suptitle(f"{row['class_name']}: x_center={x_center}, y_center={y_center}, disparity={depth_value:.2f}")
                fig.savefig(out_vis_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            print(f"Error processing row {idx} in {detected_objects_csv}: {e}")
        return
            
@ray.remote(num_gpus=1.0)
def remote_create_detected_objects_depth_csv(object_detections_csv_list, args):
    for global_idx in tqdm(range(len(object_detections_csv_list)), desc="Processing CSVs"):
        object_detections_csv = object_detections_csv_list[global_idx]
        return create_detected_objects_depth_csv(object_detections_csv, args)

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    # Input images
    if args.object_detections_dir:
        object_detections_csv_list = glob.glob(os.path.join(args.object_detections_dir, '**/bounding_box_predictions.csv'), recursive=True)
        
    random.shuffle(object_detections_csv_list)
    print('object_detections_csv_list:', len(object_detections_csv_list))
    if args.debug:
        object_detections_csv_list = object_detections_csv_list[:100]
        for object_detections_csv in object_detections_csv_list:
            create_detected_objects_depth_csv(object_detections_csv, args)
        exit()
    
    ray.init(num_cpus=args.num_processes)
    num_chunks = args.num_processes
    chunks = np.array_split(object_detections_csv_list, num_chunks)
    
    futures = [remote_create_detected_objects_depth_csv.remote(chunk, args) for chunk in chunks]
    ray.get(futures)
    
    
