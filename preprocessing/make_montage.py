import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def create_montage(images, grid_size):
    """
    Create a montage from a list of images
    Args:
        images: List of numpy arrays (images)
        grid_size: Number of images per row/column
    Returns:
        Montage image as numpy array
    """
    # Resize images to same size
    cell_height = max(img.shape[0] for img in images)
    cell_width = max(img.shape[1] for img in images)
    
    # Create empty montage
    montage = np.zeros((cell_height * grid_size, cell_width * grid_size, 3), dtype=np.uint8)
    
    # Fill montage
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= len(images):
                break
            
            img = images[idx]
            # Resize if necessary
            if img.shape[:2] != (cell_height, cell_width):
                img = cv2.resize(img, (cell_width, cell_height))
                
            y_start = i * cell_height
            y_end = (i + 1) * cell_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width
            
            montage[y_start:y_end, x_start:x_end] = img
            idx += 1
    
    return montage

def visualize_image_sizes(input_dir: Path, output_dir: Path):
    """
    Create separate size visualizations for each category subfolder
    """
    # Create directory for size analysis
    size_dir = output_dir / 'size_analysis'
    size_dir.mkdir(exist_ok=True)
    print(f"Created size analysis directory: {size_dir}")
    
    # Process each category subfolder
    print(f"Scanning input directory: {input_dir}")
    for category_dir in input_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        category = category_dir.name
        print(f"Processing category: {category}")
        
        # Initialize list for this category's images
        category_images = []
        
        # Process all images in this category's subfolder
        for img_path in category_dir.glob("*.jpg"):  # or use "*.png" if needed
            img = cv2.imread(str(img_path))
            if img is not None:
                height, width = img.shape[:2]
                area = width * height
                category_images.append({
                    'width': width,
                    'height': height,
                    'area': area,
                    'filename': img_path.name
                })
        
        if not category_images:
            print(f"No images found in category: {category}")
            continue
            
        print(f"Found {len(category_images)} images in {category}")
        
        # Create summary file for this category
        summary_file = size_dir / f'{category}_size_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Size Analysis for Category: {category}\n")
            f.write("-" * 50 + "\n")
            
            # Extract dimensions
            widths = [img['width'] for img in category_images]
            heights = [img['height'] for img in category_images]
            areas = [img['area'] for img in category_images]
            
            # Calculate statistics
            stats = {
                'Count': len(category_images),
                'Width range': f"{min(widths)} - {max(widths)}",
                'Height range': f"{min(heights)} - {max(heights)}",
                'Mean width': f"{sum(widths)/len(widths):.1f}",
                'Mean height': f"{sum(heights)/len(heights):.1f}",
                'Mean area': f"{sum(areas)/len(areas):.1f}",
            }
            
            # Write statistics
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            
            # Find extreme cases
            f.write("\nExtreme cases:\n")
            # Largest area
            largest = max(category_images, key=lambda x: x['area'])
            f.write(f"Largest image: {largest['filename']} ({largest['width']}x{largest['height']})\n")
            # Smallest area
            smallest = min(category_images, key=lambda x: x['area'])
            f.write(f"Smallest image: {smallest['filename']} ({smallest['width']}x{largest['height']})\n")
        
        # Create scatter plot for this category
        plt.figure(figsize=(10, 8))
        plt.scatter(widths, heights, alpha=0.6)
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title(f'Image Dimensions - {category}\n{len(category_images)} images')
        
        # Add mean point
        mean_width = sum(widths) / len(widths)
        mean_height = sum(heights) / len(heights)
        plt.scatter([mean_width], [mean_height], color='red', s=100, 
                   label='Mean size', marker='*')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(size_dir / f'{category}_size_scatter.png')
        plt.close()
        
        # Create size distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(areas, bins=30, alpha=0.7)
        plt.xlabel('Image Area (pixels²)')
        plt.ylabel('Count')
        plt.title(f'Size Distribution - {category}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(size_dir / f'{category}_size_distribution.png')
        plt.close()
    
    print("Size analysis complete!")

def main():
    parser = argparse.ArgumentParser(description='Create montages and analyze image sizes')
    parser.add_argument("--input_dir", required=True, help="Input directory containing category subfolders")
    parser.add_argument("--output_dir", required=True, help="Output directory for montages")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    input_dir = Path(args.input_dir)
    print(f"Scanning input directory: {input_dir}")
    
    # First, analyze and visualize image sizes
    print("Analyzing image sizes...")
    visualize_image_sizes(input_dir, output_dir)
    
    # # Comment out montage creation temporarily
    # for category_dir in input_dir.iterdir():
    #     if not category_dir.is_dir():
    #         continue
            
    #     category = category_dir.name
    #     print(f"\nProcessing category: {category}")
        
    #     # Collect images for this category
    #     images = []
    #     for img_path in category_dir.glob("*.jpg"):
    #         img = cv2.imread(str(img_path))
    #         if img is not None:
    #             images.append(img)
    #             print(f"Loaded image: {img_path.name}")
    #         else:
    #             print(f"Warning: Could not read image {img_path}")

    #     if not images:
    #         print(f"No valid images found for category {category}")
    #         continue

    #     print(f"Creating montage for {category} with {len(images)} images")
        
    #     # Calculate grid size
    #     n_images = len(images)
    #     grid_size = int(np.ceil(np.sqrt(n_images)))
        
    #     # Create and save montage
    #     montage = create_montage(images, grid_size)
    #     montage_path = output_dir / f"{category}_montage.jpg"
    #     cv2.imwrite(str(montage_path), montage)
    #     print(f"Saved montage for {category} at {montage_path}")
        
    #     # Clear images list to free memory
    #     images.clear()

    print("Size analysis complete!")

if __name__ == "__main__":
    main()