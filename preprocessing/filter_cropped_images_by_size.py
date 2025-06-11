import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_size_distributions(original_df, filtered_df, output_folder, size_threshold):
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(original_df['image_size_prop'], bins=50, alpha=0.5, label='All Images', color='blue')
    plt.hist(filtered_df['image_size_prop'], bins=50, alpha=0.5, label='Filtered Images', color='red')
    
    plt.xlabel('Image Size Proportion')
    plt.ylabel('Count')
    plt.title('Distribution of Image Size Proportions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at threshold
    plt.axvline(x=size_threshold, color='green', linestyle='--', label=f'Threshold ({size_threshold})')
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_folder, 'size_distribution.png')
    plt.savefig(plot_path)
    plt.close()

def filter_and_organize_images(
    csv_path,
    output_base_folder,
    size_threshold=0.05
):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Create output base folder
    Path(output_base_folder).mkdir(parents=True, exist_ok=True)
    
    # Filter for images with size_prop >= size_threshold
    filtered_df = df[df['image_size_prop'] >= size_threshold].copy()
    
    # Track statistics
    stats = {
        'total_images': len(df),
        'filtered_images': len(filtered_df),
        'processed_by_class': {}
    }
    
    # Add a column to track successful processing
    filtered_df['processed'] = False
    
    # Process each valid image
    for idx, row in filtered_df.iterrows():
        source_path = row['cropped_image_path']
        class_name = row['class_name']
        
        # Skip if source image doesn't exist
        if not os.path.exists(source_path):
            print(f"Image not found: {source_path}")
            continue
            
        # Create class subfolder
        class_folder = os.path.join(output_base_folder, class_name)
        Path(class_folder).mkdir(parents=True, exist_ok=True)
        
        # Prepare destination path
        dest_path = os.path.join(class_folder, os.path.basename(source_path))
        
        try:
            # Copy the image
            shutil.copy2(source_path, dest_path)
            
            # Mark as processed
            filtered_df.loc[idx, 'processed'] = True
            
            # Update statistics
            stats['processed_by_class'][class_name] = stats['processed_by_class'].get(class_name, 0) + 1
            
        except Exception as e:
            print(f"Error processing {source_path}: {e}")
            continue
    
    # Keep only successfully processed rows
    final_df = filtered_df[filtered_df['processed']].drop(columns=['processed'])
    
    # Save filtered CSV
    csv_output_path = os.path.join(output_base_folder, 'filtered_dataset.csv')
    final_df.to_csv(csv_output_path, index=False)
    
    # Create and save distribution plot
    plot_size_distributions(df, final_df, output_base_folder, size_threshold)
    
    # Print statistics
    print("\nProcessing Statistics:")
    print(f"Total images in CSV: {stats['total_images']}")
    print(f"Images meeting size criterion: {stats['filtered_images']}")
    print(f"Successfully processed images: {len(final_df)}")
    print("\nImages processed by class:")
    for class_name, count in stats['processed_by_class'].items():
        print(f"{class_name}: {count}")
    
    print(f"\nDone! Filtered images saved to {output_base_folder}")
    print(f"Filtered CSV saved to {csv_output_path}")
    print(f"Size distribution plot saved to {os.path.join(output_base_folder, 'size_distribution.png')}")

if __name__ == "__main__":
    # # Example usage
    # csv_path = "/path/to/your/input.csv"  # Path to your CSV file
    # output_folder = "/path/to/output/filtered_images"  # Where to save the filtered images
    
    # filter_and_organize_images(csv_path, output_folder)
    csv_path = "/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class/cropped_images_summary.csv"  # Path to your CSV file
    threshold = 0.05  # User-defined threshold
    output_folder = f"/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class_filtered-by-size-{threshold}"  # Where to save the filtered images
    
    filter_and_organize_images(
        csv_path=csv_path,
        output_base_folder=output_folder,
        size_threshold=threshold
    )