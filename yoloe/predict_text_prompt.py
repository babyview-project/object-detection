
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
import time

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
        default=[],
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
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing saved data"
    )
    parser.add_argument("--rank_id", type=int, default=0, 
                        help="Rank ID for distributed running.")
    parser.add_argument("--num_parallel", type=int, default=1, 
                        help="Number of parallel processes.")
    parser.add_argument("--confidence", type=float, default=0.1, 
                        help="Confidence threshold for frame annotations.")
    return parser.parse_args()

def get_file_list(source):
    file_list = []
    if Path(source).suffix == ".txt":
        with open(source, "r") as f:
            file_list = [line.strip() for line in f.readlines()]  # Read file paths from the text file
    elif Path(source).suffix in [".jpg", ".png"]:
        file_list = [source]  # Single file case
    return file_list

def get_fourth_number(s):
    parts = s.split("_")
    return parts[3] if len(parts) > 3 else s

# Given a file list run the model on each file and save outputs
def predict_images(model, file_list, args, custom_file_name=True, count=0, video_id=None, frame_id=None):
    Path(args.output).mkdir(parents=True, exist_ok=True)
    csv_file = Path(args.output) / "bounding_box_predictions.csv"
    # If the CSV file already exists, load it into a DataFrame
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        existing_paths = set(df['original_frame_path'])  # Assuming 'original_frame_path' is the column name in the CSV
    else:
        existing_paths = set()
    print("Predicting frames")
    for input_path in tqdm(file_list):
        if video_id is None:
            video_id = Path(input_path).parent.name.removesuffix('_processed')
            # pulling unique hashed id
            # hashed_id = get_fourth_number(full_video_id)
        if frame_id is None:
            frame_id = Path(input_path).stem
        try:
            timestamp = "T"+time.strftime("%H:%M:%S", time.gmtime(int(frame_id)))
        except(Exception):
            timestamp = ""
        file_name = f"{Path(input_path).stem}_annotated{Path(input_path).suffix}"
        if custom_file_name:
            file_name = f"{Path(input_path).parent.name}_{file_name}"
        output_path = Path(f'{args.output}/{file_name}')
        if (input_path in existing_paths):
            if args.overwrite:
                df = pd.read_csv(csv_file)
                df = df[df['original_frame_path'] != input_path]
                # Save the updated DataFrame back to the CSV
                df.to_csv(csv_file, index=False)
            else:
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
        filtered_detections = detections[detections.confidence > args.confidence]
        # Check how many detections were filtered
        filtered_labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(filtered_detections["class_name"], filtered_detections.confidence)
        ]
        saved_path = ""
        if (count % args.save_frame_every == 0):
            annotated_image = image.copy()
            if args.save_with_mask:
                annotated_image = sv.MaskAnnotator(
                    color_lookup=sv.ColorLookup.INDEX,
                    opacity=0.4
                ).annotate(scene=annotated_image, detections=filtered_detections)
            annotated_image = sv.BoxAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                thickness=thickness
            ).annotate(scene=annotated_image, detections=filtered_detections)
            annotated_image = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                text_scale=text_scale,
                smart_position=True
            ).annotate(scene=annotated_image, detections=filtered_detections, labels=filtered_labels)
            annotated_image.save(output_path)
            saved_path = output_path
        count = count+1
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            # TODO: different types of csvs -- one for the main pipeline and one for test runs that doesn't include gcp name etc. and a new one for when we switch off of superseded gcp names
            # original frame path should be a uid for each frame, saved frame path is either empty if this frame isn't saved or the full path
            if f.tell() == 0:  # Write header only if the file is empty
                writer.writerow(["superseded_gcp_name_feb25", "time_in_extended_iso", "xmin", "ymin", "xmax", "ymax", "confidence", "class_name", "masked_pixel_count", "frame_number", "original_frame_path", "saved_frame_path"])
            wrote_to_csv = False
            for bbox, confidence, class_name, masked_pixel_count in zip(detections.xyxy, detections.confidence, detections["class_name"], masked_pixel_counts):
                writer.writerow([video_id, timestamp, *bbox.tolist(), confidence, class_name, masked_pixel_count, frame_id, input_path, saved_path])
                wrote_to_csv = True
            if not wrote_to_csv:
                writer.writerow([video_id, timestamp, "", "", "", "", "", "", "", frame_id, input_path, saved_path])
                wrote_to_csv = True
        frame_id = None
        video_id = None
    return count

def prompt_free_model(prompt_free_list='tools/ram_tag_list.txt'):
    unfused_model = YOLOE("yoloe-v8l.yaml")
    unfused_model.load("pretrain/yoloe-v8l-seg.pt")
    unfused_model.eval()
    unfused_model.cuda()
    with open(prompt_free_list, 'r') as f:
        names = [x.strip() for x in f.readlines()]
    vocab = unfused_model.get_vocab(names)
    model = YOLOE("pretrain/yoloe-11l-seg.pt")
    model.set_vocab(vocab, names=names)
    model.model.model[-1].is_fused = True
    model.model.model[-1].conf = 0.001
    model.model.model[-1].max_det = 1000
    return model

filename = "ultralytics/cfg/datasets/lvis.yaml"
def main():
    args = parse_args()
    if not args.output:
        base = os.getcwd()
        args.output = Path(f"{base}/yoloe_outputs")
    main_output_folder = args.output
    print(args.names)
    if len(args.names) <= 1:
        model = prompt_free_model()
    else:
        model = YOLOE(args.checkpoint)
        model.set_classes(args.names, model.get_text_pe(args.names))
    model.to(args.device)
    count = 0
    if os.path.isdir(args.source):
        subdirs = [d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d))]
        # If there are subdirectories assume that this means we want to save csv files at a video level/too much data to save a single CSV
        # Also ignoring parent directory level files, in the future could switch to using os.walk 
        if subdirs:
            number_of_subdirs = len(subdirs)
            group_size = number_of_subdirs // args.num_parallel
            start_idx = args.rank_id * group_size
            end_idx = start_idx + group_size
            if args.rank_id == args.num_parallel - 1:
                end_idx = number_of_subdirs
            current_group_frames = subdirs[start_idx:end_idx]
            print("Processing videos")
            for subdir in tqdm(current_group_frames):
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
            predict_images(model, file_list, args, custom_file_name=False)
    else:
        predict_images(model, get_file_list(args.source), args)
        

if __name__ == "__main__":
    main()
