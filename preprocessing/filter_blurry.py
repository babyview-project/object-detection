import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import json

def calculate_blurriness(image):
    """
    Calculate blurriness score using Laplacian variance.
    Lower values indicate more blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_category_blurriness(category_dir):
    """
    Analyze blurriness scores for all images in a category.
    Returns dict with image paths and their blur scores.
    """
    scores = {}
    for img_path in category_dir.glob('*.jpg'):  # Add '.png' if needed
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                blur_score = calculate_blurriness(img)
                scores[str(img_path)] = blur_score
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return scores

def plot_blurriness_distribution(all_scores, output_dir):
    """
    Plot histogram of blurriness scores and save statistics.
    """
    scores = list(all_scores.values())
    
    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=50, edgecolor='black')
    plt.title('Distribution of Blurriness Scores')
    plt.xlabel('Blurriness Score (higher = sharper)')
    plt.ylabel('Number of Images')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for different percentiles
    percentiles = [10, 25, 50, 75, 90]
    colors = ['r', 'g', 'b', 'g', 'r']
    for p, c in zip(percentiles, colors):
        value = np.percentile(scores, p)
        plt.axvline(x=value, color=c, linestyle='--', alpha=0.5,
                    label=f'{p}th percentile: {value:.2f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'blurriness_distribution.png')
    plt.close()
    
    # Save statistics
    stats = {
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'percentiles': {p: np.percentile(scores, p) for p in percentiles}
    }
    
    with open(output_dir / 'blur_statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    return stats

def filter_and_copy_images(all_scores, input_dir, output_dir, threshold):
    """
    Copy non-blurry images to output directory maintaining category structure.
    """
    total_images = len(all_scores)
    kept_images = 0
    
    for img_path, score in tqdm(all_scores.items(), desc="Copying sharp images"):
        if score >= threshold:
            # Convert string path back to Path object
            src_path = Path(img_path)
            # Get category name from parent directory
            category = src_path.parent.name
            # Create category directory in output
            dst_dir = output_dir / category
            dst_dir.mkdir(parents=True, exist_ok=True)
            # Copy image
            shutil.copy2(src_path, dst_dir / src_path.name)
            kept_images += 1
    
    print(f"\nKept {kept_images}/{total_images} images ({(kept_images/total_images)*100:.1f}%)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Filter blurry images from dataset')
    parser.add_argument('--input_dir', required=True, help='Input directory containing category subfolders')
    parser.add_argument('--output_dir', required=True, help='Output directory for filtered images')
    parser.add_argument('--threshold', type=float, help='Blurriness threshold (optional)')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all blurriness scores
    print("Analyzing image blurriness...")
    all_scores = {}
    for category_dir in tqdm(list(input_dir.iterdir()), desc="Processing categories"):
        if category_dir.is_dir():
            category_scores = analyze_category_blurriness(category_dir)
            all_scores.update(category_scores)
    
    if not all_scores:
        print("No images found!")
        return
    
    # Plot distribution and get statistics
    print("\nGenerating statistics and plots...")
    stats = plot_blurriness_distribution(all_scores, output_dir)
    
    # If threshold not provided, suggest one based on percentiles
    if args.threshold is None:
        suggested_threshold = stats['percentiles'][25]  # 25th percentile
        print(f"\nSuggested threshold (25th percentile): {suggested_threshold:.2f}")
        print("You can run the script again with --threshold value to filter images")
        return
    
    # Filter and copy images
    print(f"\nFiltering images with threshold {args.threshold}...")
    filter_and_copy_images(all_scores, input_dir, output_dir, args.threshold)

if __name__ == "__main__":
    main()