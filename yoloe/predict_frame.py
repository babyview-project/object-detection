import os
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import supervision as sv
from ultralytics import YOLOE

cwd_path = os.getcwd()
config_path = Path(f"{cwd_path}/config.json")
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)
output_folder = config_data.get("save_folder")
cdi_output = f"{output_folder}/yoloe/cdi_allframes_1fps"
vedi_output = f"{output_folder}/yoloe/vedi_10k"
random_frames_path = f"{output_folder}/10000_random_frames.txt"
frames_path = f"{output_folder}/sampled_frames"
save_frame_every = 100
save_with_mask = False

# if word list is none, using VEDI by default
def predict_frames(frames_path, output_path, word_list=None):
    if word_list is None:
        word_list = " ".join(pd.read_csv(Path(f"{os.getcwd()}/tools/vedi_words.csv"))['object'].astype(str))
    command = (f"python predict_text_prompt.py  --source {frames_path} --output {output_path}  --checkpoint pretrain/yoloe-v8l-seg.pt   --names {word_list}  --save_frame_every {save_frame_every} {'--save_with_mask' if save_with_mask else ''} --device cuda")
    print(command)
    os.system(command)

def predict_video(video_path=f"{output_folder}/sampled_frames/00320003_2024-12-14_2_214a6cb812_processed", word_list=None):
    if word_list is None:
        word_list = " ".join(pd.read_csv(Path(f"{os.getcwd()}/tools/vedi_words.csv"))['object'].astype(str))
    command = (f"python predict_text_prompt.py  --source {video_path} --output {os.getcwd()}  --checkpoint pretrain/yoloe-v8l-seg.pt   --names {word_list}    --device cuda:0")
    os.system(command)

def predict_with_cdi_list():
    df = pd.read_csv(Path(f"{os.getcwd()}/tools/MCDI_items_with_AoA.csv"))
    # Filter for nouns only
    df_nouns = df[df["lexical_category"] == "nouns"]

    # Remove entries containing spaces
    df_nouns = df_nouns[~df_nouns["uni_lemma"].str.contains(" ", na=False)]

    # Create a space-separated string of uni_lemmas
    word_list = " ".join(df_nouns["uni_lemma"].astype(str)) + " " + "person"
    predict_frames(frames_path=frames_path, word_list=word_list,
                   output_path=cdi_output)

def predict_with_vedi_list():
    predict_frames(frames_path=frames_path,
                   word_list=None,
                   output_path=vedi_output)

predict_with_cdi_list()
#predict_with_vedi_list()