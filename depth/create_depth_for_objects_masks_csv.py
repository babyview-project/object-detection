import glob
import os
import shutil
import argparse
import random
import pandas as pd
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import ray
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from collections import defaultdict

"""
For each detected object, get its object mask, sample 100 random points, then get its depth. Store in a new CSV.
For 10k frames (80k+ detections), this takes ~5 minutes using only CPUs.

conda activate babyview-pose
cd /ccn2/u/khaiaw/Code/babyview-pose/object-detection/depth
python create_depth_for_objects_masks_csv.py \
    --object_masks_csv /ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_10k/bounding_box_predictions.csv \
    --depth_dir /ccn2/dataset/babyview/outputs_20250312/depth/4M_frames/ \
    --viz_dir ./viz_object_masks_depth/ \
    --output_csv_path ./yoloe_cdi_10k_object_masks_depth.csv \
    --debug
"""

resize_transform_256 = transforms.Resize(256)

def get_args():
    parser = argparse.ArgumentParser(description='Extract depth from images using a pre-trained model')
    parser.add_argument('--object_masks_csv', type=str, help='Path to input directory')
    parser.add_argument('--depth_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--viz_dir', type=str, default='./viz_detected_objects_depth/', help='Directory to save visualizations')
    parser.add_argument('--output_csv_path', type=str, default='./detected_objects_depth.csv', help='Path to save the output CSV')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()

def add_visualize_dot(image, x, y, color='red'):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image

def create_detected_objects_depth_csv(object_masks_df, args):
    # create a new DataFrame to store the results
    new_object_masks_depth_df = pd.DataFrame(columns=['superseded_gcp_name_feb25', 'time_in_extended_iso', 'class_name', 'depth_value'])
    
    # groupby saved_mask_path
    object_masks_df_grouped = object_masks_df.groupby('saved_mask_path', sort=False)
    
    # iterate through each group
    for mask_idx, (saved_mask_path, group) in enumerate(tqdm(object_masks_df_grouped, desc="Processing masks")):
        try:
            saved_mask = np.load(saved_mask_path, allow_pickle=True)
            original_frame_path = group.iloc[0]['original_frame_path']
            dir_and_filename = "/".join(original_frame_path.split('/')[-2:]) # get last two parts of the path
            depth_path = os.path.join(args.depth_dir, 'numpy', dir_and_filename.replace('.jpg', '.npy'))
            depth = np.load(depth_path)
            
            all_objects_random_location_list = []
            # all_objects_depth_value = defaultdict(list)
            all_objects_depth_value = {}
            
            # iterate through each row in the group
            for obj_idx, (_, row) in enumerate(group.iterrows()):
                class_name = row['class_name']
                depth_value = np.nan
                if saved_mask.shape == (): # no objects detected
                    pass
                else:
                    object_saved_mask = saved_mask[obj_idx][:-1, :-1] # for some reason the saved mask is larger than the original image, so we crop it slightly
                    mask_indices = np.argwhere(object_saved_mask)
                    if len(mask_indices) == 0:
                        pass
                    else:
                        random_indices = random.sample(range(len(mask_indices)), min(100, len(mask_indices))) # pick 100 random locations from the mask where it is True
                        random_locations = mask_indices[random_indices]
                        random_locations = random_locations // 2
                        all_objects_random_location_list.extend(random_locations.tolist())
                        
                        random_locations_depth_list = []
                        for loc in random_locations:
                            random_locations_depth_list.append(depth[loc[0], loc[1]])
                        depth_value = np.mean(random_locations_depth_list)
                        all_objects_depth_value[class_name] = depth_value
                
                new_row = pd.DataFrame([{
                    'superseded_gcp_name_feb25': row['superseded_gcp_name_feb25'],
                    'time_in_extended_iso': row['time_in_extended_iso'],
                    'class_name': class_name,
                    'depth_value': depth_value
                }])
                new_object_masks_depth_df = pd.concat([new_object_masks_depth_df, new_row], ignore_index=True)
            
            if mask_idx < 50:
                # visualize
                fig, ax = plt.subplots(1, 3, figsize=(10, 5))
                out_vis_path = os.path.join(args.viz_dir, dir_and_filename.replace('/', '_').replace('.jpg', '.png'))
                os.makedirs(os.path.dirname(out_vis_path), exist_ok=True)
                
                image = Image.open(original_frame_path)
                image = resize_transform_256(image)
                ax[0].imshow(image)
                ax[0].set_title("Input Image")
                
                im = ax[1].imshow(depth, cmap="plasma", vmin=np.min(depth), vmax=np.max(depth))
                ax[1].set_title("Predicted Disparity (inverse-depth)")
                cbar = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
                
                annotated_frame_path = row['saved_frame_path']
                annotated_image = Image.open(annotated_frame_path)
                annotated_image = resize_transform_256(annotated_image)
                ax[2].imshow(annotated_image)
                ax[2].set_title("Annotated Frame")
                
                all_objects_depth_value = sorted(all_objects_depth_value.items(), key=lambda x: x[1]) # sort all_objects_depth_value by depth values
                items = [f"{class_name}: {depth_value:.2f}" for class_name, depth_value in all_objects_depth_value]
                all_objects_depth_text = "\n".join([", ".join(items[i:i+5]) for i in range(0, len(items), 5)])
                fig.suptitle(all_objects_depth_text, y=1.05)
                fig.savefig(out_vis_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
            
        except Exception as e:
            print(f"Error processing saved_mask_path {saved_mask_path}: {e}")
            breakpoint()
            continue  
        
    new_object_masks_depth_df.to_csv(args.output_csv_path, index=False)
            
@ray.remote(num_gpus=1.0)
def remote_create_detected_objects_depth_csv(object_detections_csv_list, args):
    for global_idx in tqdm(range(len(object_detections_csv_list)), desc="Processing CSVs"):
        object_detections_csv = object_detections_csv_list[global_idx]
        return create_detected_objects_depth_csv(object_detections_csv, args)

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    # delete the viz directory if it exists, use shutil
    if os.path.exists(args.viz_dir):
        shutil.rmtree(args.viz_dir)
        
    object_masks_df = pd.read_csv(args.object_masks_csv)
    
    if args.debug:
        object_masks_df = object_masks_df[:200]  # for debugging, take only the first N rows
    
    create_detected_objects_depth_csv(object_masks_df, args)
    
    # ray.init(num_cpus=args.num_processes)
    # num_chunks = args.num_processes
    # chunks = np.array_split(object_detections_csv_list, num_chunks)
    
    # futures = [remote_create_detected_objects_depth_csv.remote(chunk, args) for chunk in chunks]
    # ray.get(futures)
    
    
