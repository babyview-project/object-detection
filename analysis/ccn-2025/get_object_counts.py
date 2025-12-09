import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def count_objects_per_video(input_csv: str, output_dir: str):
    """
    Count number of detected objects per video and generate statistics,
    excluding rows with any NA values
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Print initial shape
    print(f"Initial number of rows: {len(df)}")
    
    # Remove rows with NAs
    df_clean = df.dropna()
    
    # Print number of rows removed
    rows_removed = len(df) - len(df_clean)
    print(f"Rows with NAs removed: {rows_removed} ({(rows_removed/len(df)*100):.2f}%)")
    print(f"Remaining rows: {len(df_clean)}")
    
    # Count objects per video
    video_counts = df_clean.groupby('superseded_gcp_name_feb25').size()
    
    # Basic statistics
    stats = {
        'Total videos': len(video_counts),
        'Total objects (excluding NAs)': video_counts.sum(),
        'Average objects per video': video_counts.mean(),
        'Median objects per video': video_counts.median(),
        'Min objects': video_counts.min(),
        'Max objects': video_counts.max(),
        'Std dev': video_counts.std(),
        'Rows with NAs removed': rows_removed,
        'Percentage rows removed': f"{(rows_removed/len(df)*100):.2f}%"
    }
    
    # Save statistics to text file
    stats_file = output_dir / 'object_count_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("Object Detection Statistics (Excluding NAs)\n")
        f.write("=======================================\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # Save detailed counts to CSV
    counts_df = pd.DataFrame(video_counts).reset_index()
    counts_df.columns = ['video_id', 'object_count']
    counts_df.to_csv(output_dir / 'objects_per_video.csv', index=False)
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(data=video_counts, bins=50)
    plt.title('Distribution of Objects per Video (Excluding NAs)')
    plt.xlabel('Number of Objects')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'object_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=video_counts)
    plt.title('Box Plot of Objects per Video (Excluding NAs)')
    plt.ylabel('Number of Objects')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'object_count_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\nSummary:")
    print(f"Total videos processed: {stats['Total videos']}")
    print(f"Total objects detected (excluding NAs): {stats['Total objects (excluding NAs)']}")
    print(f"Average objects per video: {stats['Average objects per video']:.2f}")
    print(f"\nDetailed results saved to: {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Count objects per video from detection CSV')
    parser.add_argument('--input_csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', default='../analysis/object_counts', 
                        help='Directory to save results (default: ../analysis/object_counts)')
    
    args = parser.parse_args()
    count_objects_per_video(args.input_csv, args.output_dir)

if __name__ == '__main__':
    main()