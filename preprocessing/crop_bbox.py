import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def load_class_names(names_file):
    """
    Load class names from obj.names file
    Returns a dictionary mapping class index to class name
    """
    class_dict = {}
    try:
        with open(names_file, 'r') as f:
            classes = f.read().strip().split('\n')
            for idx, class_name in enumerate(classes):
                class_dict[idx] = class_name.strip()
    except Exception as e:
        print(f"Error loading class names: {e}")
        return {}
    return class_dict

def yolo_to_bbox(image_width, image_height, center_x, center_y, width, height):
    """
    Convert YOLO format (center_x, center_y, width, height) to bbox coordinates (x_min, y_min, x_max, y_max)
    All values are normalized (0-1)
    """
    # Convert from center to top-left point
    x_min = (center_x - width/2) * image_width
    y_min = (center_y - height/2) * image_height
    
    # Convert from width/height to bottom-right point
    x_max = (center_x + width/2) * image_width
    y_max = (center_y + height/2) * image_height
    
    # Ensure coordinates are within image boundaries
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(image_width, int(x_max))
    y_max = min(image_height, int(y_max))
    
    return x_min, y_min, x_max, y_max

def create_class_folders(output_folder, class_dict):
    """
    Create subfolder for each class
    """
    for class_idx, class_name in class_dict.items():
        folder_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in class_name)
        class_folder = os.path.join(output_folder, folder_name)
        Path(class_folder).mkdir(parents=True, exist_ok=True)
    return {idx: "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name) 
            for idx, name in class_dict.items()}

def process_files(txt_folder, img_folder, output_folder, names_file):
    """
    Process all txt files and their corresponding images
    Returns a list of dictionaries containing file information for CSV
    """
    # Load class names
    class_dict = load_class_names(names_file)
    if not class_dict:
        print("Failed to load class names. Exiting.")
        return []

    # Create class subfolders and get sanitized folder names
    folder_dict = create_class_folders(output_folder, class_dict)
    
    # List to store file information for CSV
    file_info = []
    
    # Get all txt files
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        # Get corresponding image name (try both .jpg and .png)
        img_name_jpg = txt_file.replace('.txt', '.jpg')
        img_name_png = txt_file.replace('.txt', '.png')
        
        if os.path.exists(os.path.join(img_folder, img_name_jpg)):
            img_path = os.path.join(img_folder, img_name_jpg)
            original_img_name = img_name_jpg
        elif os.path.exists(os.path.join(img_folder, img_name_png)):
            img_path = os.path.join(img_folder, img_name_png)
            original_img_name = img_name_png
        else:
            print(f"No corresponding image found for {txt_file}")
            continue
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
            
        image_height, image_width = img.shape[:2]
        
        # Read annotations
        txt_path = os.path.join(txt_folder, txt_file)
        with open(txt_path, 'r') as f:
            annotations = f.readlines()
        
        # Process each bounding box
        for ann in annotations:
            try:
                # Parse YOLO format
                class_idx, center_x, center_y, width, height, confidence = map(float, ann.strip().split())
                class_idx = int(class_idx)
                
                if class_idx not in class_dict:
                    print(f"Unknown class index {class_idx} in {txt_file}")
                    continue
                
                # Get class name
                class_name = class_dict[class_idx]
                
                # Convert to pixel coordinates
                x_min, y_min, x_max, y_max = yolo_to_bbox(image_width, image_height, 
                                                         center_x, center_y, width, height)
                
                # Crop image
                cropped = img[y_min:y_max, x_min:x_max]
                
                # Create filename with confidence score
                base_name = os.path.splitext(original_img_name)[0]
                output_filename = f"{base_name}_{class_name}_{confidence:.3f}.jpg"
                
                # Get class folder name and create output path
                class_folder = folder_dict[class_idx]
                output_path = os.path.join(output_folder, class_folder, output_filename)
                
                # Save cropped image
                cv2.imwrite(output_path, cropped)
                
                # Store file information for CSV
                file_info.append({
                    'class_name': class_name,
                    'original_image_path': img_path,
                    'cropped_image_path': output_path
                })
                
            except Exception as e:
                print(f"Error processing annotation in {txt_file}: {e}")
                continue
    
    return file_info

def create_csv_report(file_info, output_folder):
    """
    Create CSV report from file information
    """
    df = pd.DataFrame(file_info)
    csv_path = os.path.join(output_folder, 'cropped_images_report.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV report saved to: {csv_path}")

def main():
    # Define your folders here
    txt_folder = "path/to/txt/folder"
    img_folder = "path/to/image/folder"
    output_folder = "path/to/output/folder"
    names_file = "path/to/obj.names"
    
    # Process files and get file information
    file_info = process_files(txt_folder, img_folder, output_folder, names_file)
    
    # Create CSV report
    create_csv_report(file_info, output_folder)

if __name__ == "__main__":
    main()