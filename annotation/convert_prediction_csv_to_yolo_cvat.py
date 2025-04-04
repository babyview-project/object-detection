import pandas as pd
import json
import argparse
from pathlib import Path
import zipfile
import os
import shutil
from PIL import Image

def get_image_dimensions(image_path):
    """
    Get the dimensions of an image using PIL
    """
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"Error reading image dimensions for {image_path}: {str(e)}")
        return None

def convert_to_yolo_format(row, img_width, img_height):
    """
    Convert bounding box coordinates to YOLO format (normalized coordinates)
    YOLO format requires:
    - x_center, y_center (normalized center coordinates)
    - width, height (normalized dimensions)
    - confidence (optional)
    All coordinate values should be between 0 and 1
    """
    try:
        # Get coordinates
        xmin = float(row['xmin'])
        ymin = float(row['ymin'])
        xmax = float(row['xmax'])
        ymax = float(row['ymax'])
        confidence = float(row['confidence'])  # Get confidence value
        
        # Calculate absolute width and height of the box
        box_width = xmax - xmin
        box_height = ymax - ymin
        
        # Calculate center points
        x_center = xmin + (box_width / 2)
        y_center = ymin + (box_height / 2)
        
        # Normalize all values by image dimensions
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = box_width / img_width
        height_norm = box_height / img_height
        
        return x_center_norm, y_center_norm, width_norm, height_norm, confidence
        
    except Exception as e:
        print(f"Error in coordinate conversion: {str(e)}")
        raise

def convert_csv_to_yolo(csv_path, frames_folder, output_path):
    """
    Convert CSV file to YOLO format and create a ZIP file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Create temporary directory for files
        temp_dir = Path('temp_yolo_files')
        temp_dir.mkdir(exist_ok=True)
        
        # Create obj.names file with class names
        class_names = sorted(df['class_name'].unique())
        with open(temp_dir / 'obj.names', 'w') as f:
            f.write('\n'.join(class_names))
        
        # Create obj.data file
        with open(temp_dir / 'obj.data', 'w') as f:
            f.write(f'classes = {len(class_names)}\n')
            f.write('train = train.txt\n')
            f.write('names = obj.names\n')
            f.write('backup = backup/\n')
        
        # Create class mapping
        class_mapping = {name: i for i, name in enumerate(class_names)}
        
        # Create images.txt file with image paths
        image_paths = []
        frames_folder = Path(frames_folder)
        
        # Process each image and its annotations
        for annotation_path, group in df.groupby('annotation_path'):
            image_name = Path(annotation_path).name
            image_paths.append(image_name)
            
            # Get image dimensions from the actual image file
            image_path = frames_folder / image_name
            dimensions = get_image_dimensions(image_path)
            
            if dimensions is None:
                print(f"Skipping {image_name} - couldn't read dimensions")
                continue
                
            img_width, img_height = dimensions
            print(f"Processing {image_name} - dimensions: {img_width}x{img_height}")
            
            # Create annotation file for this image
            anno_path = temp_dir / f"{Path(image_name).stem}.txt"
            
            with open(anno_path, 'w') as f:
                for _, row in group.iterrows():
                    # Convert coordinates to YOLO format using actual image dimensions
                    x_center, y_center, width, height, confidence = convert_to_yolo_format(row, img_width, img_height)
                    class_id = class_mapping[row['class_name']]
                    
                    # Write YOLO format line: <class_id> <x_center> <y_center> <width> <height> <confidence>
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
        
        # Create train.txt with image paths
        with open(temp_dir / 'train.txt', 'w') as f:
            f.write('\n'.join(image_paths))
        
        # Create ZIP file
        output_file = Path(output_path)
        with zipfile.ZipFile(output_file, 'w') as zipf:
            for file_path in temp_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(temp_dir))
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        print(f"\nSuccessfully converted CSV to YOLO format: {output_file}")
        print(f"Total images: {len(image_paths)}")
        print(f"Total classes: {len(class_names)}")
        
    except Exception as e:
        print(f"Error converting CSV to YOLO format: {str(e)}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert CSV annotations to YOLO format')
    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('frames_folder', help='Path to folder containing the frames')
    parser.add_argument('--output', '-o', 
                      help='Path to output ZIP file (default: input_filename_yolo.zip)',
                      default=None)
    
    args = parser.parse_args()
    
    # If no output path specified, create one based on input filename
    if args.output is None:
        input_path = Path(args.csv_path)
        args.output = input_path.parent / f"{input_path.stem}_yolo.zip"
    
    convert_csv_to_yolo(args.csv_path, args.frames_folder, args.output)

if __name__ == "__main__":
    main()