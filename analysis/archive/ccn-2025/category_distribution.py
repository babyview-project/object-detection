import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
import os

def analyze_category_distribution(
    merged_df: pd.DataFrame,
    confidence_thresholds: Union[List[float], float],
    output_dir: str = 'category_distributions',
    top_n: int = 30
) -> dict:
    """
    Analyze category distribution for multiple confidence thresholds
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Input DataFrame containing 'class_name' and 'confidence' columns
    confidence_thresholds : List[float] or float
        List of confidence thresholds or single threshold value
    output_dir : str, optional
        Directory to save CSV files (default: 'category_distributions')
    top_n : int, optional
        Number of top categories to plot (default: 30)
        
    Returns:
    --------
    dict : Dictionary containing distribution DataFrames for each threshold
    """
    # Convert single threshold to list
    if isinstance(confidence_thresholds, (int, float)):
        confidence_thresholds = [confidence_thresholds]
    
    # Sort thresholds for consistent processing
    confidence_thresholds = sorted(confidence_thresholds)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results for each threshold
    results = {}
    
    # Create separate figures for counts and percentages
    fig_counts = plt.figure(figsize=(15, 8))
    ax_counts = fig_counts.add_subplot(111)
    
    fig_percentages = plt.figure(figsize=(15, 8))
    ax_percentages = fig_percentages.add_subplot(111)
    
    for threshold in confidence_thresholds:
        # Filter data based on confidence threshold
        filtered_df = merged_df[merged_df['confidence'] >= threshold]
        
        # Calculate distributions
        category_counts = filtered_df['class_name'].value_counts()
        percentage_distribution = filtered_df['class_name'].value_counts(normalize=True) * 100
        
        # Create distribution DataFrame
        distribution_df = pd.DataFrame({
            'Category': category_counts.index,
            'Count': category_counts.values,
            'Percentage': percentage_distribution.values,
            'Avg_Confidence': filtered_df.groupby('class_name')['confidence'].mean()
        })
        
        # Save to CSV
        output_path = os.path.join(output_dir, f'category_distribution_conf_{threshold:.2f}.csv')
        distribution_df.to_csv(output_path, index=False)
        print(f"\nConfidence threshold: {threshold:.2f}")
        print(f"Total categories: {len(distribution_df)}")
        print(f"Total instances: {filtered_df.shape[0]}")
        print(f"Distribution saved to: {output_path}")
        
        # Store results
        results[threshold] = distribution_df
        
        # Plot top N categories for this threshold
        print(distribution_df.shape)
        top_n_categories = distribution_df.head(top_n)
        
        # Plot counts
        ax_counts.plot(range(len(top_n_categories)), 
                      top_n_categories['Count'], 
                      marker='o', 
                      label=f'Conf ≥ {threshold:.2f}')
        
        # Plot percentages
        ax_percentages.plot(range(len(top_n_categories)), 
                          top_n_categories['Percentage'], 
                          marker='o', 
                          label=f'Conf ≥ {threshold:.2f}')
    
    # Customize count plot
    ax_counts.set_xticks(range(len(top_n_categories)))
    ax_counts.set_xticklabels(top_n_categories['Category'], rotation=45, ha='right')
    ax_counts.set_title(f'Distribution of Top {top_n} Categories - Counts')
    ax_counts.set_xlabel('Category')
    ax_counts.set_ylabel('Count')
    ax_counts.legend()
    ax_counts.grid(True, alpha=0.3)
    fig_counts.tight_layout()
    
    # Customize percentage plot
    ax_percentages.set_xticks(range(len(top_n_categories)))
    ax_percentages.set_xticklabels(top_n_categories['Category'], rotation=45, ha='right')
    ax_percentages.set_title(f'Distribution of Top {top_n} Categories - Percentages')
    ax_percentages.set_xlabel('Category')
    ax_percentages.set_ylabel('Percentage')
    ax_percentages.legend()
    ax_percentages.grid(True, alpha=0.3)
    fig_percentages.tight_layout()
    
    # Save the plots separately
    fig_counts.savefig(os.path.join(output_dir, 'threshold_comparison_counts.png'))
    fig_percentages.savefig(os.path.join(output_dir, 'threshold_comparison_percentages.png'))
    
    # Show the plots
    plt.show()
    
    # Create a summary comparison DataFrame
    summary_data = []
    for threshold, df in results.items():
        summary_data.append({
            'Confidence_Threshold': threshold,
            'Total_Categories': len(df),
            'Total_Instances': df['Count'].sum(),
            'Avg_Confidence': df['Avg_Confidence'].mean()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'threshold_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print("\nThreshold Comparison Summary:")
    print(summary_df)
    
    return results