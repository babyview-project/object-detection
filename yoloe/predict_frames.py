import os
import json
from pathlib import Path
import pandas as pd
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from glob import glob
import time

cwd_path = os.getcwd()
config_path = Path(f"{cwd_path}/config.json")
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)
output_folder = config_data.get("save_folder")
#cdi_output = f"{output_folder}/yoloe/cdi_allframes_1fps"
cdi_output_1k = f"{output_folder}/yoloe/cdi_1k"
all_output_10k = f"{output_folder}/yoloe/promptfree_10k"
frames_path_1k = f"{output_folder}/1000_random_frames.txt"
frames_path_10k = f"{output_folder}/10000_random_frames.txt"
frames_path = f"{output_folder}/sampled_frames"
save_frame_every = 1
save_with_mask = False
session_name = "test"

# if word list is none, using VEDI by default
def predict_frames_command(frames_path, output_path, rank_id, num_parallel, device_id, word_list=[], overwrite=False):
    if word_list is None:
        word_list = []
    command = (f"source ~/miniconda3/bin/activate;conda activate yoloe;export CUDA_VISIBLE_DEVICES={device_id}; "
               f"python predict_text_prompt.py  --source {frames_path} --output {output_path} " 
               f"--checkpoint pretrain/yoloe-v8l-seg.pt   --names {word_list}  --save_frame_every {save_frame_every} "
               f"{'--save_with_mask' if save_with_mask else ''} --device cuda --rank_id {rank_id} --num_parallel {num_parallel} "
               f"{'--overwrite' if overwrite else ''}")
    return command

def is_tmux_session_active(session_name):
    """Check if the given tmux session is still active."""
    try:
        subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return True  # If no exception, the session exists
    except subprocess.CalledProcessError:
        return False  # Non-zero return code means session does not exist
    except FileNotFoundError:
        print("Tmux is not installed or not available in the system path.")
        return False

def run_parallel_predictions(word_list, full_output_path, session_name, args):
    all_rank_ids = list(range(args.num_parallel))
    device_ids = [int(id) for id in args.device_ids.strip("[]").split(",")]
    num_devices = len(device_ids)
    rank_device_dict = {rank_id: device_ids[rank_id % num_devices] for rank_id in all_rank_ids}
    # Create a new tmux session and split into the required number of panes
    os.system(f"tmux new-session -d -s {session_name}")
    for i in range(1, args.num_parallel):
        os.system(f"tmux split-window -t {session_name} -h")
        os.system(f"tmux select-layout -t {session_name} tiled")
    # Send the command to each pane
    for rank_id in all_rank_ids:
        device_id = rank_device_dict[rank_id]
        command = predict_frames_command(frames_path=args.input_frames, word_list=word_list, 
                                         output_path=full_output_path, device_id=device_id, rank_id=rank_id, 
                                         num_parallel=args.num_parallel, overwrite=args.overwrite)
        os.system(f"tmux send-keys -t {session_name}.{rank_id} '{command}' Enter")
    print(f"Started {args.num_parallel} parallel processes in tmux session {session_name}")
    print(f"Use 'tmux attach -t {session_name}' to view progress")

def extract_frames(video_path, output_folder):
    """Extracts frames from a video using ffmpeg."""
    command = [
        "ffmpeg", "-i", video_path,
        os.path.join(output_folder, "frame_%04d.png")
    ]
    subprocess.run(command, check=True)

def get_frame_rate(video_path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    fps_str = result.stdout.strip()
    
    # Convert fraction FPS (e.g., 30000/1001) to float
    if "/" in fps_str:
        num, denom = map(int, fps_str.split('/'))
        fps = num / denom
    else:
        fps = float(fps_str)
    
    return float(fps)

def create_video_from_frames(frame_folder, framerate, output_video):
    """Creates a video from frames using ffmpeg."""
    if os.path.exists(output_video):
        os.remove(output_video)
    framerate = str(round(framerate,2))
    output_video = str(output_video)
    command = [
        "ffmpeg", "-framerate", framerate, "-i",  
        os.path.join(frame_folder, "frame_%04d_annotated.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video
    ]
    subprocess.run(command, check=True)

def video_postprocessing(output_frames, framerate, output_video, session_name, overwrite=False):
    while is_tmux_session_active(session_name):
        print("Waiting for tmux session to finish...")
        time.sleep(20)  # Wait for 10 seconds before checking again
        # After tmux is done, proceed with cleanup and stitching
    if not os.path.exists(output_video) or overwrite:
        create_video_from_frames(output_frames, framerate, output_video)
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    # only keeping on average 1 annotated frame per second
    for i, frame in enumerate(glob(str(Path(f"{output_frames}/*.png")))):
        if i % 30 != 0:
            os.remove(frame)
    print("Complete!")

def predict_video(input_path, word_list, full_output_path, session_name, args):
    os.makedirs("temp", exist_ok=True)
    framerate = get_frame_rate(input_path)
    extract_frames(input_path, "temp")
    args.input_frames = str(Path(f"{os.getcwd()}/temp"))
    run_parallel_predictions(word_list, full_output_path, session_name, args)
    video_postprocessing(full_output_path, framerate, Path(f"{full_output_path}/output.mp4"), session_name, args.overwrite)

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
    session_name_with_random_suffix = f"yoloe_predict_{os.urandom(4).hex()}"
    session_name = session_name_with_random_suffix
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing saved data"
    )
    args = parser.parse_args()
    if args.text_prompts == "cdi":
        word_list = cdi_list()
        full_output_path = Path(f'{args.output_path}/cdi')
    elif args.text_prompts == "vedi":
        word_list = vedi_list()
        full_output_path = Path(f'{args.output_path}/vedi')
    # TODO: fix prompt free - using vedi list for now
    else:
        #word_list = vedi_list()
        #full_output_path = Path(f'{args.output_path}/vedi')
        word_list = []
        full_output_path = Path(f'{args.output_path}/{args.text_prompts or "promptfree"}')
    if args.input_frames == "10k":
        full_output_path = Path(f'{full_output_path}_10k')
        args.input_frames = frames_path_10k
    elif args.input_frames == "1k":
        full_output_path = Path(f'{full_output_path}_1k')
        args.input_frames = frames_path_1k
    if args.input_frames.endswith("mp4"):
        predict_video(args.input_frames, word_list, full_output_path, session_name, args)
    else:
        run_parallel_predictions(word_list, full_output_path, session_name, args)

if __name__ == "__main__":
    main()
    
