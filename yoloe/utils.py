# post-processing frame csvs saved in the old format
from glob import glob
import os
import shutil
import pandas as pd
import time
from pathlib import Path
import json
from tqdm import tqdm
import random 

config_path = Path(f"{os.getcwd()}/config.json")
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)
output_folder = config_data.get("save_folder")
output_1fps = str(Path(f"{output_folder}/yoloe/cdi_allframes_1fps"))
saved_frame_annotations_path = str(Path(f"{output_folder}/yoloe/cleaned_saved_frame_annotations.csv"))

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

def fix_all_csvs(csv_path=output_1fps):
    print("Fixing csv formatting")
    for file in tqdm(glob(str(Path(f"{csv_path}/**/*.csv")), recursive=True)):
        fix_csv_formatting(file)

def extract_saved_frames(csv_path=output_1fps):
    recordings_processed = pd.read_csv("tools/recordings_processed.csv")
    saved_rows = []
    for file in tqdm(glob(str(Path(f"{csv_path}/**/*.csv")), recursive=True)):
        # Randomly skip 80% of the files
        df = pd.read_csv(file,dtype={'frame_id': str})
        # even though we technically only need to look at iloc[0], adding a more robust filter in case we're doing joined csv files in the future
        filtered = df[df['superseded_gcp_name_feb25'].isin(recordings_processed['superseded_gcp_name_feb25'])]
        # Ensure blackout_region is NA
        valid_entries = recordings_processed.loc[
            recordings_processed['superseded_gcp_name_feb25'].isin(filtered['superseded_gcp_name_feb25']) &
            recordings_processed['blackout_region'].isna()
        ]
        if not valid_entries.empty:
            # Extract non-empty saved_frame_path
            valid_rows = df.dropna(subset=['saved_frame_path']).to_dict('records')  # Convert to list of dicts                
            saved_rows.extend(valid_rows)
    # Save all saved_rows to CSV
    pd.DataFrame(saved_rows).to_csv(saved_frame_annotations_path, index=False)

def pull_random_annotations(num=100):
    saved_rows = pd.read_csv(Path(f"{output_folder}/yoloe/cleaned_saved_frame_annotations.csv"))
    saved_rows_list = saved_rows.to_dict('records')
    # Get unique 'superseded_gcp_name_feb25' values
    unique_frames = saved_rows['saved_frame_path'].unique()
    # Randomly pick unique 'superseded_gcp_name_feb25' values
    random.seed(2)
    selected_frames = random.sample(list(unique_frames), min(num, len(unique_frames)))
    selected_frames = sorted(selected_frames)
    # Prepare list for annotation metadata
    annotation_data = []
    # Iterate through the selected 'superseded_gcp_name_feb25' values
    for index, saved_path in enumerate(selected_frames):
        # Get all rows for the current 'superseded_gcp_name_feb25'
        rows_for_gcp = [row for row in saved_rows_list if row['saved_frame_path'] == saved_path]
        # Process each selected row
        for row in rows_for_gcp:
            annotated_frame_path = row['saved_frame_path']
            original_frame_path = row['original_frame_path']
            # saving in annotations folder in current directory by default
            new_annotated_path = Path(f"annotated/{row['superseded_gcp_name_feb25']}_{str(row['frame_id']).zfill(5)}_annotated.jpg")
            new_original_path = new_annotated_path.name.replace('_annotated', '')
            full_annotation_path = Path(f"annotations/{new_annotated_path}")
            full_annotation_path.parent.mkdir(parents=True, exist_ok=True)
            if not os.path.exists(full_annotation_path):
                shutil.copy(annotated_frame_path, full_annotation_path)
                shutil.copy(original_frame_path, Path(f"annotations/{new_original_path}"))
            # Store metadata for annotations CSV
            row['current_annotated_frame_path'] = new_annotated_path
            row['current_frame_path'] = new_original_path
            row['annotation_frame_index'] = index
            annotation_data.append(row)
    # Convert list of dictionaries to DataFrame
    annotation_df = pd.DataFrame(annotation_data)
    # Sort the DataFrame by the 'superseded_gcp_name' column alphabetically
    annotation_df = annotation_df.sort_values(by='current_frame_path', ascending=True)
    # Save sorted annotation metadata to CSV
    annotation_df.to_csv("annotations/yoloe_predictions.csv", index=False)
    print(f"Copied {len(selected_frames)} frames for annotation.")

def cdi_words_used():
    df = pd.read_csv(Path(f"{os.getcwd()}/tools/MCDI_items_with_AoA.csv"))
    # Filter for nouns only or "person"
    df_nouns = df[(df["lexical_category"] == "nouns") | (df["uni_lemma"] == "person")]
    # Remove entries containing spaces in 'uni_lemma'
    df_nouns = df_nouns[~df_nouns["uni_lemma"].str.contains(" ", na=False)]
    # removing first column which is empty
    df_nouns = df_nouns.loc[:, ~df_nouns.columns.str.contains('^Unnamed')]
    df_nouns = df_nouns.drop(columns=["form_type"])
    df_nouns.to_csv("cdi_words.csv", index=False)

pull_random_annotations()
#cdi_words_used()
#fix_all_csvs()