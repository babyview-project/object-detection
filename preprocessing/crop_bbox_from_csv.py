import os
import pandas as pd
from PIL import Image
from pathlib import Path

def crop_and_save_images(
    csv_path, 
    frame_folder, 
    output_folder, 
    output_csv_path
):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Prepare output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # List to collect output CSV rows
    output_rows = []
    
    for idx, row in df.iterrows():
        original_frame_path = row['original_frame_path']
        class_name = str(row['class_name'])
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        confidence = row['confidence']
        
        # Skip rows with NaN in any bbox or confidence column
        if pd.isna(xmin) or pd.isna(ymin) or pd.isna(xmax) or pd.isna(ymax) or pd.isna(confidence):
            print(f"Skipping row {idx} due to NaN values.")
            continue
        
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        confidence = float(confidence)
        
        # Find the image path
        image_path = os.path.join(frame_folder, original_frame_path)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Open image
        try:
            with Image.open(image_path) as img:
                # Get original image size
                orig_width, orig_height = img.size
                orig_size = orig_width * orig_height

                # Crop
                cropped = img.crop((xmin, ymin, xmax, ymax))
                
                # Get cropped image size
                crop_width, crop_height = cropped.size
                crop_size = crop_width * crop_height
                
                # Calculate proportion
                size_proportion = crop_size / orig_size

                # Prepare class folder
                class_folder = os.path.join(output_folder, class_name)
                Path(class_folder).mkdir(parents=True, exist_ok=True)
                
                # Prepare output filename
                # Extract the parent folder name and the file name without extension
                parent_folder = os.path.basename(os.path.dirname(original_frame_path))
                file_name = os.path.splitext(os.path.basename(original_frame_path))[0]
                simplified_path = f"{parent_folder}_{file_name}"
                cropped_filename = f"{class_name}_{confidence:.3f}_{simplified_path}.jpg"
                cropped_path = os.path.join(class_folder, cropped_filename)
                
                # Save cropped image
                cropped.save(cropped_path)
                
                # Add to output CSV
                output_rows.append({
                    'original_image_path': image_path,
                    'class_name': class_name,
                    'cropped_image_path': cropped_path,
                    'image_size_in_pixels': crop_size,
                    'image_size_prop': size_proportion
                })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Save output CSV
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Done! Cropped images and CSV saved to {output_folder} and {output_csv_path}")

if __name__ == "__main__":
    # Example usage
    csv_path = "/ccn2/dataset/babyview/outputs_20250312/yoloe/cdi_10k/bounding_box_predictions.csv"         # Path to your input CSV
    frame_folder = "/ccn2/dataset/babyview/outputs_20250312/sampled_frames/"                    # Path to your frame images
    output_folder = "/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class"         # Where to save cropped images
    output_csv_path = "/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class/cropped_images_summary.csv"  # Output CSV path

    crop_and_save_images(csv_path, frame_folder, output_folder, output_csv_path)