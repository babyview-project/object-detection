# post-processing frame csvs saved in the old format
from glob import glob
import os
import pandas as pd
import time
from pathlib import Path
import json
from tqdm import tqdm

config_path = Path(f"{os.getcwd()}/config.json")
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)
output_folder = config_data.get("save_folder")
output_1fps = f"{output_folder}/yoloe/cdi_allframes_1fps"

def fix_csv_formatting(file_path):
    df = pd.read_csv(file_path,dtype={'frame_number': str})
    # Create new column 'superseded_gcp_name_feb25' by removing suffix from 'video_id'
    df['superseded_gcp_name_feb25'] = df['video_id'].apply(lambda x: x.removesuffix("_processed"))
    df = df.drop(columns=['video_id'])
    df = df.rename(columns={'frame_number': 'frame_id'})
    # Create 'time_in_extended_iso' based on 'frame_number'
    df['time_in_extended_iso'] = df['frame_id'].apply(lambda x: "T" + time.strftime("%H:%M:%S", time.gmtime(int(x))))
    # Get parent directory for 'original_frame_path'
    original_frames = Path(df['original_frame_path'].iloc[0]).parent
    no_detection_rows = []
    # Loop through files in the original_frames directory and add new rows
    for frame in os.listdir(original_frames):
        frame_path = Path(f'{original_frames}/{frame}')
        if str(frame_path) not in df['original_frame_path'].values:
            frame_number = frame_path.stem
            no_detection_row = {
                'superseded_gcp_name_feb25': df['superseded_gcp_name_feb25'].iloc[0],  # Use the first 'superseded_gcp_name_feb25' - they should all be the same video
                'time_in_extended_iso': "T" + time.strftime("%H:%M:%S", time.gmtime(int(frame_number))),
                'frame_id': frame_number,
                'original_frame_path': str(frame_path),
                # Other columns are left blank 
            }
            no_detection_rows.append(no_detection_row)
    if no_detection_rows:
        no_detection_df = pd.DataFrame(no_detection_rows)
        # Concatenate the original DataFrame with the new rows
        df = pd.concat([df, no_detection_df], ignore_index=True)
    columns_order = ['superseded_gcp_name_feb25', 'time_in_extended_iso'] + [col for col in df.columns if col not in ['superseded_gcp_name_feb25', 'time_in_extended_iso']]
    df = df[columns_order]
    df = df.sort_values(by='time_in_extended_iso', ascending=True)
    df.to_csv(file_path, index=False)

def fix_all_csvs():
    print(f"{str(Path(output_1fps))}/**/*.csv")
    print("Fixing csv formatting")
    for file in tqdm(glob(str(Path(f"{output_1fps}/**/*.csv")), recursive=True)):
        fix_csv_formatting(file)

fix_all_csvs()