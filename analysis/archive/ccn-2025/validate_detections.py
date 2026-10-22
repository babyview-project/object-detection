import os
from pathlib import Path
import argparse
from datetime import datetime

def read_label_mapping(obj_names_path):
    """
    Read obj.names file and create a mapping from index to label name
    """
    label_map = {}
    try:
        with open(obj_names_path, 'r') as f:
            for idx, line in enumerate(f):
                label = line.strip()
                if label:  # Skip empty lines
                    label_map[str(idx)] = label
    except Exception as e:
        print(f"Error reading obj.names file {obj_names_path}: {e}")
        return None
    return label_map

def read_first_column_with_labels(file_path, label_map, confidence_threshold=0.0):
    """
    Read the first column of a txt file and convert values to labels using label_map
    Optionally filter by confidence threshold
    """
    labels = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:  # Check if line is not empty
                    value = parts[0]
                    # Check confidence if available (last column)
                    confidence = float(parts[-1]) if len(parts) >= 6 else 1.0
                    
                    if confidence >= confidence_threshold and value in label_map:
                        labels.add(label_map[value])
                    elif value not in label_map:
                        print(f"Warning: Label index {value} not found in obj.names")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return labels

def calculate_metrics(ground_truth, predictions):
    """
    Calculate hits, misses, false alarms, precision, recall, and f-score
    """
    hits = len(ground_truth.intersection(predictions))
    misses = len(ground_truth.difference(predictions))
    false_alarms = len(predictions.difference(ground_truth))
    
    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    recall = hits / (hits + misses) if (hits + misses) > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'hits': hits,
        'misses': misses,
        'false_alarms': false_alarms,
        'precision': precision,
        'recall': recall,
        'f_score': f_score
    }

def calculate_overall_statistics(total_stats, label_stats):
    """
    Calculate overall statistics in two ways:
    1. Over all categories
    2. Over only categories with non-zero hits
    
    Parameters:
    total_stats: Dictionary with total counts
    label_stats: Dictionary with per-label statistics
    
    Returns:
    Dictionary containing both sets of statistics
    """
    # 1. Statistics over all categories
    all_precision = total_stats['total_hits'] / (total_stats['total_hits'] + total_stats['total_false_alarms']) if (total_stats['total_hits'] + total_stats['total_false_alarms']) > 0 else 0
    all_recall = total_stats['total_hits'] / (total_stats['total_hits'] + total_stats['total_misses']) if (total_stats['total_hits'] + total_stats['total_misses']) > 0 else 0
    all_f_score = 2 * (all_precision * all_recall) / (all_precision + all_recall) if (all_precision + all_recall) > 0 else 0

    # 2. Statistics over categories with non-zero hits
    active_categories = {label: stats for label, stats in label_stats.items() if stats['hits'] > 0}
    n_active = len(active_categories)
    
    if n_active > 0:
        active_hits = sum(stats['hits'] for stats in active_categories.values())
        active_misses = sum(stats['misses'] for stats in active_categories.values())
        active_false_alarms = sum(stats['false_alarms'] for stats in active_categories.values())
        
        active_precision = active_hits / (active_hits + active_false_alarms) if (active_hits + active_false_alarms) > 0 else 0
        active_recall = active_hits / (active_hits + active_misses) if (active_hits + active_misses) > 0 else 0
        active_f_score = 2 * (active_precision * active_recall) / (active_precision + active_recall) if (active_precision + active_recall) > 0 else 0
    else:
        active_precision = active_recall = active_f_score = 0
        active_hits = active_misses = active_false_alarms = 0

    return {
        'all_categories': {
            'n_categories': len(label_stats),
            'hits': total_stats['total_hits'],
            'misses': total_stats['total_misses'],
            'false_alarms': total_stats['total_false_alarms'],
            'precision': all_precision,
            'recall': all_recall,
            'f_score': all_f_score
        },
        'active_categories': {
            'n_categories': n_active,
            'hits': active_hits,
            'misses': active_misses,
            'false_alarms': active_false_alarms,
            'precision': active_precision,
            'recall': active_recall,
            'f_score': active_f_score
        }
    }

def save_statistics(stats_file, total_stats, label_stats, confidence_threshold=0.0):
    """
    Save detailed statistics to file
    """
    # Calculate both types of overall statistics
    overall_stats = calculate_overall_statistics(total_stats, label_stats)
    
    with open(stats_file, 'a') as f:  # Changed to append mode
        # Add threshold information to header
        f.write(f"\n\n{'='*50}\n")
        f.write(f"Statistics for confidence threshold ≥ {confidence_threshold}\n")
        f.write(f"{'='*50}\n\n")
        
        # Write overall statistics for all categories
        f.write("=== Overall Statistics (All Categories) ===\n")
        all_stats = overall_stats['all_categories']
        f.write(f"Total Categories: {all_stats['n_categories']}\n")
        f.write(f"Total Hits: {all_stats['hits']}\n")
        f.write(f"Total Misses: {all_stats['misses']}\n")
        f.write(f"Total False Alarms: {all_stats['false_alarms']}\n")
        f.write(f"Overall Precision: {all_stats['precision']:.3f}\n")
        f.write(f"Overall Recall: {all_stats['recall']:.3f}\n")
        f.write(f"Overall F-score: {all_stats['f_score']:.3f}\n\n")
        
        # Write statistics for categories with non-zero hits
        f.write("=== Overall Statistics (Categories with Non-zero Hits) ===\n")
        active_stats = overall_stats['active_categories']
        f.write(f"Active Categories: {active_stats['n_categories']}\n")
        f.write(f"Total Hits: {active_stats['hits']}\n")
        f.write(f"Total Misses: {active_stats['misses']}\n")
        f.write(f"Total False Alarms: {active_stats['false_alarms']}\n")
        f.write(f"Overall Precision: {active_stats['precision']:.3f}\n")
        f.write(f"Overall Recall: {active_stats['recall']:.3f}\n")
        f.write(f"Overall F-score: {active_stats['f_score']:.3f}\n\n")
        
        # Write per-label statistics
        f.write("=== Per-label Statistics ===\n")
        for label, stats in sorted(label_stats.items()):
            precision = stats['hits'] / (stats['hits'] + stats['false_alarms']) if (stats['hits'] + stats['false_alarms']) > 0 else 0
            recall = stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0
            f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f.write(f"\n{label}:\n")
            f.write(f"  Hits: {stats['hits']}\n")
            f.write(f"  Misses: {stats['misses']}\n")
            f.write(f"  False Alarms: {stats['false_alarms']}\n")
            f.write(f"  Precision: {precision:.3f}\n")
            f.write(f"  Recall: {recall:.3f}\n")
            f.write(f"  F-score: {f_score:.3f}\n")

def compare_folders(ground_truth_folder, prediction_folder, output_file, confidence_threshold=0.0):
    """
    Compare txt files in two folders using label mappings from obj.names
    """
    ground_truth_folder = Path(ground_truth_folder)
    prediction_folder = Path(prediction_folder)
    
    # Read label mappings from both folders
    gt_obj_names = ground_truth_folder / 'obj.names'
    pred_obj_names = prediction_folder / 'obj.names'
    
    if not gt_obj_names.exists() or not pred_obj_names.exists():
        print("Error: obj.names file not found in one or both folders")
        return
    
    gt_label_map = read_label_mapping(gt_obj_names)
    pred_label_map = read_label_mapping(pred_obj_names)
    
    if not gt_label_map or not pred_label_map:
        return
    
    # Track overall statistics
    total_stats = {
        'total_files': 0,
        'total_hits': 0,
        'total_misses': 0,
        'total_false_alarms': 0,
        'total_ground_truth': 0,
        'total_predictions': 0
    }
    
    # Store per-label statistics
    label_stats = {}
    
    # Process each file
    for gt_file in ground_truth_folder.glob('*.txt'):
        if gt_file.name == 'obj.names':  # Skip obj.names file
            continue
            
        pred_file = prediction_folder / gt_file.name
        
        if not pred_file.exists():
            print(f"Warning: No corresponding prediction file for {gt_file.name}")
            continue
        
        # Read labels from both files, applying confidence threshold only to predictions
        gt_labels = read_first_column_with_labels(gt_file, gt_label_map)
        pred_labels = read_first_column_with_labels(pred_file, pred_label_map, confidence_threshold)
        
        # Update statistics
        total_stats['total_files'] += 1
        total_stats['total_ground_truth'] += len(gt_labels)
        total_stats['total_predictions'] += len(pred_labels)
        
        # Update per-label statistics
        for label in gt_labels.union(pred_labels):
            if label not in label_stats:
                label_stats[label] = {'hits': 0, 'misses': 0, 'false_alarms': 0}
            
            if label in gt_labels and label in pred_labels:
                label_stats[label]['hits'] += 1
                total_stats['total_hits'] += 1
            elif label in gt_labels:
                label_stats[label]['misses'] += 1
                total_stats['total_misses'] += 1
            else:
                label_stats[label]['false_alarms'] += 1
                total_stats['total_false_alarms'] += 1
    
    # Save statistics to file, passing the confidence threshold
    save_statistics(output_file, total_stats, label_stats, confidence_threshold)

def main():
    parser = argparse.ArgumentParser(description='Compare ground truth and prediction txt files using label mappings')
    parser.add_argument('ground_truth_folder', help='Path to folder containing ground truth txt files and obj.names')
    parser.add_argument('prediction_folder', help='Path to folder containing prediction txt files and obj.names')
    parser.add_argument('--output', '-o', 
                      help='Path to output statistics file (default: statistics.txt)',
                      default='statistics.txt')
    parser.add_argument('--thresholds', '-t',
                      help='Comma-separated list of confidence thresholds (default: 0.3,0.5,0.7)',
                      default='0.3,0.5,0.7')
    
    args = parser.parse_args()
    
    # Initialize output file
    with open(args.output, 'w') as f:
        f.write(f"Statistics generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Process each threshold
    confidence_thresholds = [float(t) for t in args.thresholds.split(',')]
    for threshold in confidence_thresholds:
        compare_folders(args.ground_truth_folder, args.prediction_folder, args.output, threshold)

if __name__ == "__main__":
    main()
