import pandas as pd

def get_category_aoa(aoa_csv_path, category_csv_path):
    # Read both CSV files
    aoa_df = pd.read_csv(aoa_csv_path)
    category_df = pd.read_csv(category_csv_path)
    
    # Create a dictionary mapping uni_lemma to AoA
    aoa_dict = dict(zip(aoa_df['uni_lemma'], aoa_df['AoA']))
    
    # Add AoA column by mapping Category values to their corresponding AoA
    category_df['AoA'] = category_df['Category'].map(aoa_dict)
    
    # Print summary of missing values if any
    missing_count = category_df['AoA'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} categories have no matching AoA values")
        print("Categories with missing AoA:", category_df[category_df['AoA'].isna()]['Category'].unique())
    
    # Save the updated dataframe to a new CSV file
    output_path = category_csv_path.replace('.csv', '_with_aoa.csv')
    category_df.to_csv(output_path, index=False)
    print(f"\nSaved updated CSV to: {output_path}")
    
    return category_df

if __name__ == "__main__":
    # Replace with your CSV file paths
    aoa_csv_path = "../MCDI_items_with_AOA.csv"
    category_csv_path = "../category_distribution_conf_0.30.csv"
    updated_df = get_category_aoa(aoa_csv_path, category_csv_path)