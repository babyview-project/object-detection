import os
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm  # for progress bar
import matplotlib.pyplot as plt
import seaborn as sns
from category_distribution import analyze_category_distribution

def merge_csv_files(root_dir, output_file, top_n, output_category_dir):
    """
    Recursively find and merge all 'bounding_box_predictions.csv' files.
    
    Args:
        root_dir (str): Path to the root directory to start searching
        output_file (str): Path to save the merged CSV file
    """
    # List to store all dataframes
    dfs = []
    
    # Counter for found files
    file_count = 0
    
    # First, collect all matching files
    csv_files = []
    for root, _, files in os.walk(root_dir):
        if 'bounding_box_predictions.csv' in files:
            csv_files.append(os.path.join(root, 'bounding_box_predictions.csv'))
    
    # Process files with progress bar
    for file_path in tqdm(csv_files, desc="Processing CSV files"):
        try:
            # Read the CSV file and add folder path information
            df = pd.read_csv(file_path)
            # Add the folder path as a column (optional)
            df['source_folder'] = os.path.relpath(os.path.dirname(file_path), root_dir)
            dfs.append(df)
            file_count += 1
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    
    if not dfs:
        raise ValueError("No 'bounding_box_predictions.csv' files found!")
    
    # Merge all dataframes
    print("\nMerging dataframes...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged dataframe
    print(f"Saving merged file to {output_file}")
    merged_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully merged {file_count} files into {output_file}")
    print(f"Total rows in merged file: {len(merged_df)}")

    # Plot the distribution
    # After merging, call the plotting function
    confidence_thresholds = [0.3, 0.5, 0.7]
    results = analyze_category_distribution(merged_df,
                                            confidence_thresholds=confidence_thresholds,
                                            output_dir=output_category_dir,
                                            top_n=int(top_n)
                                        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple bounding_box_predictions.csv files")
    parser.add_argument("input_dir", help="Root directory to search for CSV files")
    parser.add_argument("output_file", help="Output path for merged CSV file")
    parser.add_argument("top_n", type=int, help="Top N categories to plot")
    parser.add_argument("output_category_dir", help="Output path for category distribution CSV file")
    args = parser.parse_args()
    merge_csv_files(args.input_dir, args.output_file, args.top_n, args.output_category_dir)