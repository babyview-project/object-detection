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
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        confidence = float(row['confidence'])
        
        # Find the image path
        image_path = os.path.join(frame_folder, original_frame_path)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Open image
        try:
            with Image.open(image_path) as img:
                # Crop
                cropped = img.crop((xmin, ymin, xmax, ymax))
                
                # Prepare class folder
                class_folder = os.path.join(output_folder, class_name)
                Path(class_folder).mkdir(parents=True, exist_ok=True)
                
                # Prepare output filename
                base_name = os.path.splitext(os.path.basename(original_frame_path))[0]
                cropped_filename = f"{base_name}_{class_name}_{confidence:.3f}.jpg"
                cropped_path = os.path.join(class_folder, cropped_filename)
                
                # Save cropped image
                cropped.save(cropped_path)
                
                # Add to output CSV
                output_rows.append({
                    'original_image_path': image_path,
                    'class_name': class_name,
                    'cropped_image_path': cropped_path
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
    csv_path = "input_annotations.csv"         # Path to your input CSV
    frame_folder = "frames"                    # Path to your frame images
    output_folder = "cropped_by_class"         # Where to save cropped images
    output_csv_path = "cropped_images_report.csv"  # Output CSV path

    crop_and_save_images(csv_path, frame_folder, output_folder, output_csv_path)