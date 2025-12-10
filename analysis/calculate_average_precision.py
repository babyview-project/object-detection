#!/usr/bin/env python3
"""
Calculate average precision for categories in the inclusion file.
Averages precision across two annotators (Dora and Mira) for each category,
and calculates the global average across all included categories.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
project_root = Path(__file__).parent.parent
inclusion_file = project_root / "data" / "things_bv_overlap_categories_exclude_zero_precisions.txt"
precision_file = project_root / "data" / "combined_precision_metrics_with_prevalence.csv"
output_file = project_root / "data" / "average_precision_for_included_categories.csv"

# Read inclusion file
print(f"Reading inclusion file: {inclusion_file}")
with open(inclusion_file, 'r') as f:
    included_categories = [line.strip() for line in f if line.strip()]

print(f"Found {len(included_categories)} categories in inclusion file")

# Read precision metrics
print(f"Reading precision metrics: {precision_file}")
df = pd.read_csv(precision_file)

print(f"Total rows in precision file: {len(df)}")
print(f"Annotators: {df['annotator'].unique()}")

# Filter to only included categories
df_filtered = df[df['initial_object_name'].isin(included_categories)].copy()

print(f"Rows after filtering to included categories: {len(df_filtered)}")

# Check if all categories have data for both annotators
category_counts = df_filtered.groupby('initial_object_name')['annotator'].count()
missing_annotators = category_counts[category_counts < 2]
if len(missing_annotators) > 0:
    print(f"\nWarning: {len(missing_annotators)} categories are missing data for one or both annotators:")
    for cat, count in missing_annotators.items():
        annotators = df_filtered[df_filtered['initial_object_name'] == cat]['annotator'].unique()
        print(f"  {cat}: {count} annotator(s) - {list(annotators)}")

# Calculate average precision for each category
category_precisions = []
for category in included_categories:
    category_data = df_filtered[df_filtered['initial_object_name'] == category]
    
    if len(category_data) == 0:
        print(f"Warning: No data found for category '{category}'")
        continue
    
    # Get precision values for each annotator
    precisions = category_data['precision'].values
    
    if len(precisions) == 1:
        # Only one annotator has data
        avg_precision = precisions[0]
        annotators_present = category_data['annotator'].values[0]
        print(f"Warning: Category '{category}' only has data from {annotators_present}")
    else:
        # Average across annotators
        avg_precision = np.mean(precisions)
        annotators_present = list(category_data['annotator'].values)
    
    category_precisions.append({
        'category': category,
        'average_precision': avg_precision,
        'dora_precision': category_data[category_data['annotator'] == 'Dora']['precision'].values[0] if 'Dora' in category_data['annotator'].values else np.nan,
        'mira_precision': category_data[category_data['annotator'] == 'Mira']['precision'].values[0] if 'Mira' in category_data['annotator'].values else np.nan
    })

# Create results dataframe
results_df = pd.DataFrame(category_precisions)

# Calculate global average precision
global_avg_precision = results_df['average_precision'].mean()

print(f"\nCalculated average precision for {len(results_df)} categories")
print(f"Global average precision: {global_avg_precision:.6f}")

# Add global average as a summary row
summary_row = pd.DataFrame([{
    'category': 'GLOBAL_AVERAGE',
    'average_precision': global_avg_precision,
    'dora_precision': np.nan,
    'mira_precision': np.nan
}])

# Combine results with summary
final_df = pd.concat([results_df, summary_row], ignore_index=True)

# Save to CSV
print(f"\nSaving results to: {output_file}")
final_df.to_csv(output_file, index=False)

print(f"\nDone! Results saved to {output_file}")
print(f"\nSummary:")
print(f"  Number of categories: {len(results_df)}")
print(f"  Global average precision: {global_avg_precision:.6f}")
print(f"  Min category precision: {results_df['average_precision'].min():.6f}")
print(f"  Max category precision: {results_df['average_precision'].max():.6f}")
