
import argparse
import os
from PIL import Image
import supervision as sv
from ultralytics import YOLOE
from pathlib import Path
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image, directory, or text file with file paths"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yoloe-v8l-seg.pt",
        help="Path or ID of the model checkpoint"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["person"],
        help="List of class names to set for the model"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the annotated image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--save_frame_every",
        type=int,
        default=1,
        help="How often to save annotated frames"
    )
    parser.add_argument(
        "--save_with_mask",
        action='store_true',
        default=False,
        help="Whether to save annotated frames with mask or not"
    )
    return parser.parse_args()

def get_file_list(source):
    file_list = []
    if Path(source).suffix == ".txt":
        with open(source, "r") as f:
            file_list = [line.strip() for line in f.readlines()]  # Read file paths from the text file
    elif Path(source).suffix in [".jpg", ".png"]:
        file_list = [source]  # Single file case
    return file_list

# Given a file list run the model on each file and save outputs
def predict_images(model, file_list, args, custom_file_name=True, count=0, video_id=None, frame_id=None):
    args.output.mkdir(parents=True, exist_ok=True)
    csv_file = Path(args.output) / "bounding_box_predictions.csv"
    # If the CSV file already exists, load it into a DataFrame
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        existing_paths = set(df['original_frame_path'])  # Assuming 'input_path' is the column name in the CSV
    else:
        existing_paths = set()
    for input_path in file_list:
        if video_id is None:
            video_id = Path(input_path).parent.name
        if frame_id is None:
            frame_id = Path(input_path).stem
        file_name = f"{Path(input_path).stem}_annotated{Path(input_path).suffix}"
        if custom_file_name:
            file_name = f"{Path(input_path).parent.name}_{file_name}"
        output_path = Path(f'{args.output}/{file_name}')
        if Path(output_path).exists() or input_path in existing_paths:
            frame_id = None
            video_id = None
            continue
        image = Image.open(input_path).convert("RGB")
        results = model.predict(image, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        masked_pixel_counts = np.zeros(len(detections["class_name"]), dtype=int)
        if detections.mask is not None:    
            # detections.mask is an optional ndarray of format (6, 910, 512) if there are 6 labeled classes and the object is 910x512px
            # babyview frames appear to be 910x512 qualitatively
            for i, mask in enumerate(detections.mask):
                # Count the number of True values in the mask
                masked_count = np.count_nonzero(mask)
                masked_pixel_counts[i] = masked_count
        resolution_wh = image.size
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections["class_name"], detections.confidence)
        ]
        saved_path = ""
        if (count % args.save_frame_every == 0):
            annotated_image = image.copy()
            if args.save_with_mask:
                annotated_image = sv.MaskAnnotator(
                    color_lookup=sv.ColorLookup.INDEX,
                    opacity=0.4
                ).annotate(scene=annotated_image, detections=detections)
            annotated_image = sv.BoxAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                thickness=thickness
            ).annotate(scene=annotated_image, detections=detections)
            annotated_image = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                text_scale=text_scale,
                smart_position=True
            ).annotate(scene=annotated_image, detections=detections, labels=labels)
            annotated_image.save(output_path)
            saved_path = output_path
        count = count+1
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            # original frame path should be a uid for each frame, saved frame path is either empty if this frame isn't saved or the full path
            if f.tell() == 0:  # Write header only if the file is empty
                writer.writerow(["video_id", "frame_number", "xmin", "ymin", "xmax", "ymax", "confidence", "class_name", "masked_pixel_count", "original_frame_path", "saved_frame_path"])
            for bbox, confidence, class_name, masked_pixel_count in zip(detections.xyxy, detections.confidence, detections["class_name"], masked_pixel_counts):
                writer.writerow([video_id, frame_id, *bbox.tolist(), confidence, class_name, masked_pixel_count, input_path, saved_path])
        frame_id = None
        video_id = None
    return count

def main():
    args = parse_args()
    if not args.output:
        base = os.getcwd()
        args.output = Path(f"{base}/yoloe_outputs")
    main_output_folder = args.output
    model = YOLOE(args.checkpoint)
    model.to(args.device)
    model.set_classes(args.names, model.get_text_pe(args.names))
    file_list = get_file_list(source=args.source)
    count = 0
    if os.path.isdir(args.source):
        subdirs = [d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d))]
        # If there are subdirectories assume that this means we want to save csv files at a video level/too much data to save a single CSV
        # Also ignoring parent directory level files, in the future could switch to using os.walk 
        if subdirs:
            print("Processing videos")
            for subdir in tqdm(subdirs):
                subdir_path = Path(args.source) / subdir
                args.output = Path(f'{main_output_folder}/{subdir}')
                files_in_subdir = [str(file) for file in subdir_path.iterdir() 
                                if file.is_file() and (file.suffix in {'.jpg', '.png'})]
                new_count = predict_images(model, files_in_subdir, args, custom_file_name=False, count=count)               
                count = new_count
                if (count % 100000) == True:
                    print(f"Processed {count} images")
        else:
            file_list = [str(Path(f"{args.source}/{file}")) for file in os.listdir(args.source)] 
            predict_images(model, file_list, args)
    else:
        predict_images(model, get_file_list(args.source), args)
        

if __name__ == "__main__":
    main()
