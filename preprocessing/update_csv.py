import pandas as pd

def rename_csv_column(csv_path, output_csv_path=None):
    # If no output path specified, will overwrite the input file
    if output_csv_path is None:
        output_csv_path = csv_path
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Rename the column
    df = df.rename(columns={'cropped_image_path': 'image_path'})
    
    # Save the modified CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Column renamed successfully! Saved to: {output_csv_path}")

if __name__ == "__main__":
    # Example usage
    threshold = 0.05
    csv_path = f"/ccn2/dataset/babyview/outputs_20250312/yoloe_cdi_10k_cropped_by_class_filtered-by-size-{threshold}/filtered_dataset.csv"
    # Uncomment and modify the line below if you want to save to a different file
    # output_csv_path = "/path/to/your/output.csv"
    
    rename_csv_column(csv_path)