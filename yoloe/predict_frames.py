import os
import json
from pathlib import Path
import pandas as pd
import argparse

cwd_path = os.getcwd()
config_path = Path(f"{cwd_path}/config.json")
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)
output_folder = config_data.get("save_folder")
#cdi_output = f"{output_folder}/yoloe/cdi_allframes_1fps"
vedi_output = f"{output_folder}/yoloe/vedi_10k"
cdi_output = f"{output_folder}/yoloe/cdi_10k"
cdi_output_1k = f"{output_folder}/yoloe/cdi_1k"
all_output_10k = f"{output_folder}/yoloe/promptfree_10k"
frames_path_1k = f"{output_folder}/1000_random_frames.txt"
frames_path_10k = f"{output_folder}/10000_random_frames.txt"
frames_path = f"{output_folder}/sampled_frames"
save_frame_every = 100
save_with_mask = False

# if word list is none, using VEDI by default
def predict_frames_command(frames_path, output_path, rank_id, num_parallel, device_id, word_list=None):
    if word_list is None:
        word_list = []
    command = (f"source ~/miniconda3/bin/activate;conda activate yoloe;export CUDA_VISIBLE_DEVICES={device_id}; "
               f"python predict_text_prompt.py  --source {frames_path} --output {output_path} " 
               f"--checkpoint pretrain/yoloe-v8l-seg.pt   --names {word_list}  --save_frame_every {save_frame_every} "
               f"{'--save_with_mask' if save_with_mask else ''} --device cuda --rank_id {rank_id} --num_parallel {num_parallel}")
    return command

# TODO: extract frames from video, predict and reformat if required
def predict_video():
    return

def cdi_list():
    df = pd.read_csv(Path(f"{os.getcwd()}/tools/MCDI_items_with_AoA.csv"))
    # Filter for nouns only
    df_nouns = df[df["lexical_category"] == "nouns"]

    # Remove entries containing spaces
    df_nouns = df_nouns[~df_nouns["uni_lemma"].str.contains(" ", na=False)]

    # Create a space-separated string of uni_lemmas
    word_list = " ".join(df_nouns["uni_lemma"].astype(str)) + " " + "person"
    return word_list

def vedi_list():
    return " ".join(pd.read_csv(Path(f"{os.getcwd()}/tools/vedi_words.csv"))['object'].astype(str))
    
def main():
    parser = argparse.ArgumentParser(description="Process frames using YOLOE.")
    # TODO: supporting multiple sessions for text file inputs
    parser.add_argument("--device_ids", type=str, default="[0]", help="List of GPU device IDs to use.")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--text_prompts", type=str, default="cdi", help="Which list of possible detectable words to be used")
    parser.add_argument("--input_frames", type=str, default=frames_path, help="Text file or file path with the list of frames to be processed")
    parser.add_argument("--output_path", type=str, default=f"{output_folder}/yoloe", help="Path to store outputs at")
    # TODO: fix overwrite
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing saved data"
    )
    args = parser.parse_args()
    all_rank_ids = list(range(args.num_parallel))
    device_ids = [int(id) for id in args.device_ids.strip("[]").split(",")]

    num_devices = len(device_ids)
    rank_device_dict = {rank_id: device_ids[rank_id % num_devices] for rank_id in all_rank_ids}

    session_name_with_random_suffix = f"yoloe_predict_{os.urandom(4).hex()}"
    session_name = session_name_with_random_suffix
    # Create a new tmux session and split into the required number of panes
    os.system(f"tmux new-session -d -s {session_name}")
    for i in range(1, args.num_parallel):
        os.system(f"tmux split-window -t {session_name} -h")
        os.system(f"tmux select-layout -t {session_name} tiled")
    if args.text_prompts == "cdi":
        word_list = cdi_list()
        full_output_path = Path(f'{args.output_path}/cdi')
    elif args.text_prompts == "vedi":
        word_list = vedi_list()
        full_output_path = Path(f'{args.output_path}/vedi')
    # TODO: fix prompt free - using vedi list for now
    else:
        word_list = vedi_list()
        full_output_path = Path(f'{args.output_path}/vedi')
        #word_list = []
        #full_output_path = Path(f'{args.output_path}/promptfree')
    if args.input_frames == "10k":
        full_output_path = Path(f'{full_output_path}_10k')
        args.input_frames = frames_path_10k
    elif args.input_frames == "1k":
        full_output_path = Path(f'{full_output_path}_1k')
        args.input_frames = frames_path_1k
    # Send the command to each pane
    for rank_id in all_rank_ids:
        device_id = rank_device_dict[rank_id]
        command = predict_frames_command(frames_path=args.input_frames, word_list=word_list, 
                                         output_path=full_output_path, device_id=device_id, rank_id=rank_id, 
                                         num_parallel=args.num_parallel)
        os.system(f"tmux send-keys -t {session_name}.{rank_id} '{command}' Enter")
    print(f"Started {args.num_parallel} parallel processes in tmux session {session_name}")
    print(f"Use 'tmux attach -t {session_name}' to view progress")

if __name__ == "__main__":
    main()
    
