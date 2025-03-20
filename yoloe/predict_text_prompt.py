
import argparse
import os
from PIL import Image
import supervision as sv
from ultralytics import YOLOE
from pathlib import Path
import csv

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
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.output:
        base = os.getcwd()
        args.output = Path(f"{base}/default_outputs")
    model = YOLOE(args.checkpoint)
    model.to(args.device)
    model.set_classes(args.names, model.get_text_pe(args.names))
    if Path(args.source).suffix == ".txt":
        with open(args.source, "r") as f:
            file_list = [line.strip() for line in f.readlines()]  # Read file paths from the text file
    elif os.path.isdir(args.source):
        file_list = [str(Path(f"{args.source}/{file}")) for file in os.listdir(args.source)]  # List all files in the directory
    elif Path(args.source).suffix in [".jpg", ".png"]:
        file_list = [args.source]  # Single file case
    else:
        file_list = []
    for input_path in file_list:
        file_name = f"{Path(input_path).parent.name}_{Path(input_path).name}"
        output_path = Path(f'{args.output}/{file_name}')
        image = Image.open(input_path).convert("RGB")
        results = model.predict(image, verbose=False)
        #print(results)
        detections = sv.Detections.from_ultralytics(results[0])
        resolution_wh = image.size
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections["class_name"], detections.confidence)
        ]
        annotated_image = image.copy()
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
        print(f"Annotated image saved to: {output_path}")
        # Save bounding boxes to CSV
        csv_file = Path(args.output) / "bounding_boxes.csv"
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Write header only if the file is empty
                writer.writerow(["filename", "xmin", "ymin", "xmax", "ymax", "confidence", "class_name"])
            for bbox, confidence, class_name in zip(detections.xyxy, detections.confidence, detections["class_name"]):
                writer.writerow([file_name, *bbox.tolist(), confidence, class_name])
    print(f"Bounding boxes saved to: {csv_file}")

if __name__ == "__main__":
    main()
