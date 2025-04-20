import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_mask_percentage(mask):
    """
    Calculate what percentage of the frame is covered by the mask
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_pixels = np.sum(mask)
    return (mask_pixels / total_pixels) * 100

def crop_image_by_mask(image, mask):
    """
    Crop the image using the binary mask with transparency
    Args:
        image: Original image (H,W,3)
        mask: Binary mask (H,W)
    Returns:
        Masked and cropped image with transparency
    """
    # Convert image to RGBA (4 channels)
    rgba = np.concatenate([image, np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    
    # Set alpha channel using mask
    rgba[..., 3] = mask * 255
    
    # Find non-zero region to crop
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop to non-zero region
    cropped_image = rgba[rmin:rmax, cmin:cmax]
    
    return cropped_image

def process_images(input_csv, output_dir, output_csv, size_threshold=None):
    """
    Process images and crop based on masks
    
    Args:
        input_csv (str): Path to input CSV file
        output_dir (str): Path to output directory
        output_csv (str): Path to output CSV file
        size_threshold (float, optional): Minimum percentage of frame that mask must cover
                                        (0 to 100). None means no threshold.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the input CSV
    df = pd.read_csv(input_csv)
    
    # Initialize list to store new rows
    new_rows = []
    
    # Process each row
    total_masks = 0
    filtered_masks = 0
    
    for idx, row in df.iterrows():
        try:
            # Load the original image
            original_frame_path = row['original_frame_path']
            original_frame = cv2.imread(original_frame_path)
            if original_frame is None:
                print(f"Failed to load image: {original_frame_path}")
                continue
                
            # Load the mask file
            masks = np.load(row['saved_mask_path'])
            
            # Create category subfolder
            category_dir = output_dir / row['class_name']
            category_dir.mkdir(exist_ok=True)
            
            # Process each mask in the file
            for mask_idx in range(masks.shape[0]):
                total_masks += 1
                mask = masks[mask_idx]
                
                # Check size threshold if specified
                if size_threshold is not None:
                    mask_percent = calculate_mask_percentage(mask)
                    if mask_percent < size_threshold:
                        continue
                
                filtered_masks += 1
                
                # Crop the image using the mask
                cropped_image = crop_image_by_mask(original_frame, mask)

                # Extract the parent folder name and the file name without extension
                parent_folder = os.path.basename(os.path.dirname(original_frame_path))
                file_name = os.path.splitext(os.path.basename(original_frame_path))[0]
                simplified_path = f"{parent_folder}_{file_name}"
                
                # Save as PNG to preserve transparency
                cropped_mask_filename = f"{row['class_name']}_mask_{mask_idx}_{simplified_path}.png"
                cropped_path = category_dir / cropped_mask_filename
                
                # Save the cropped image with transparency
                cv2.imwrite(str(cropped_path), cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
                # Add row to new_rows
                new_row = {
                    'class_name': row['class_name'],
                    'original_frame_path': row['original_frame_path'],
                    'saved_mask_path': row['saved_mask_path'],
                    'cropped_mask_path': str(cropped_path),
                    'mask_percentage': mask_percent if size_threshold is not None else None
                }
                new_rows.append(new_row)
            
            print(f"Processed frame {idx + 1}/{len(df)}")
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Create and save the output CSV
    output_df = pd.DataFrame(new_rows)
    output_df.to_csv(output_csv, index=False)
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"Total masks processed: {total_masks}")
    if size_threshold is not None:
        print(f"Masks meeting {size_threshold}% threshold: {filtered_masks}")
        print(f"Masks filtered out: {total_masks - filtered_masks}")
    print(f"Output CSV saved to {output_csv}")

if __name__ == "__main__":
    input_dir = "/ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_10k"
    output_dir = "/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class_mask"
    output_csv = os.path.join(output_dir, "cropped_mask_summary.csv")
    # Configuration
    input_csv = os.path.join(input_dir, "bounding_box_predictions.csv")

    size_threshold = None  # Set to a number between 0% and 100% to filter by size
    
    process_images(input_csv, output_dir, output_csv, size_threshold)