from datetime import datetime
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
import yaml

cwd_path = os.getcwd()
config_path = Path(f"{cwd_path}/config.yaml")
with open(config_path, 'r') as config_file:
    config_data = yaml.safe_load(config_file)
output_folder = config_data.get("save_folder")
#cdi_output = f"{output_folder}/yoloe/cdi_allframes_1fps"
cdi_output_1k = f"{output_folder}/yoloe/cdi_1k"
all_output_10k = f"{output_folder}/yoloe/promptfree_10k"
frames_path_1k = f"{output_folder}/1000_random_frames.txt"
frames_path_10k = f"{output_folder}/10000_random_frames.txt"
frames_path = f"{output_folder}/sampled_frames"
save_frame_every = config_data["frame_save_rate"]
frame_extraction_rate = config_data["video_processing"]["frames_per_second_extraction"]
session_name = "test"

# if word list is none, using VEDI by default
def predict_frames_command(frames_path, output_dir, rank_id, num_parallel, device_id, confidence, word_list=[], overwrite=False, device="cuda", use_conda=True):
    if word_list is None:
        word_list = []
    # if this does not work, make sure you are using the right conda environment. 
    # You might also have to add something like 'source ~/miniconda3/bin/activate;conda activate objects;' (without quotes) to the beginning of the command
    command = (f"{'conda activate objects;'if use_conda else ''}export CUDA_VISIBLE_DEVICES={device_id}; "
               f"python run_yoloe.py  --source {frames_path} --output {output_dir} " 
               f"--checkpoint pretrain/yoloe-v8l-seg.pt --names {word_list}  --save_frame_every {save_frame_every} "
               f"--device {device} --rank_id {rank_id} --num_parallel {num_parallel} "
               f"{'--overwrite' if overwrite else ''} --confidence {confidence}")
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

def run_parallel_predictions(word_list, full_output_dir, session_name, args):
    if args.tmux:
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
            command = predict_frames_command(frames_path=args.input_dir, word_list=word_list, 
                                            output_dir=full_output_dir, device_id=device_id, rank_id=rank_id, 
                                            confidence=args.confidence, num_parallel=args.num_parallel, device=args.device,
                                            overwrite=args.overwrite, use_conda=args.use_conda)
            send_keys_command = f"python tmux_send_long_command.py {session_name}.{rank_id} --command '{command}'"
            os.system(send_keys_command)
        print(f"Started {args.num_parallel} parallel processes in tmux session {session_name}")
        print(f"Use 'tmux attach -t {session_name}' to view progress")
        return True
    else:
        command = predict_frames_command(frames_path=args.input_dir, word_list=word_list, 
                                         output_dir=full_output_dir, device=args.device, device_id=0, rank_id=0,
                                         confidence=args.confidence, num_parallel=args.num_parallel, 
                                         overwrite=args.overwrite, use_conda=args.use_conda)
        os.system(command)
        return True

def extract_frames(video_path, output_folder, fps=1, use_conda=True):
    """Extracts frames from a video using ffmpeg."""
    if video_path.lower().endswith(".mp4"):
        video_path = Path(video_path).resolve().parent
    command = f"{'conda run -n objects'if use_conda else ''} python extract_frames.py --videos_dir {video_path} --output_dir {output_folder} --fps {round(fps)}"
    os.system(command)

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
        os.path.join(frame_folder, "%08d_annotated.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(frame_folder, output_video)
    ]
    subprocess.run(command, check=True)

def video_postprocessing(output_frames, framerate, output_video, session_name, extracted_frame_dir, overwrite=False):
    while is_tmux_session_active(session_name):
        print("Waiting for tmux session to finish...")
        time.sleep(20)  # Wait for 20 seconds before checking again
        # After tmux is done, proceed with cleanup and stitching
    if (not os.path.exists(output_video) or overwrite) and config_data["video_processing"]["create_output_video"]:
        for directory in os.listdir(output_frames):
            full_directory_path = os.path.join(output_frames, directory)
            if os.path.isdir(full_directory_path):
                create_video_from_frames(full_directory_path, framerate, output_video)
    if os.path.exists(extracted_frame_dir) and config_data["video_processing"]["delete_extracted_frames"]:
        shutil.rmtree(extracted_frame_dir)
    # only keeping on average 1 annotated frame per second if there are too many saved
    if len(glob(str(Path(f"{output_frames}/*.jpg")))) > 100:
        for i, frame in enumerate(glob(str(Path(f"{output_frames}/*.jpg")))):
            if i % round(framerate) != 0:
                os.remove(frame)
    print("Complete!")

def predict_videos(input_path, word_list, full_output_dir, session_name, args):
    global frame_extraction_rate
    if input_path.lower().endswith(".mp4") and frame_extraction_rate == 0:
        frame_extraction_rate = get_frame_rate(input_path)
    else:
        if frame_extraction_rate == 0:
            suffix = '*.[mM][pP][4]'
            video_files = glob.glob(os.path.join(args.videos_dir, f'**/{suffix}'), recursive=True)
            if video_files:
                frame_extraction_rate = get_frame_rate(video_files[0])
    frame_extraction_dir = Path(full_output_dir).resolve().parent.joinpath("extracted_video_frames")
    extract_frames(input_path, frame_extraction_dir, frame_extraction_rate, args.use_conda)
    args.input_dir = frame_extraction_dir
    _ = run_parallel_predictions(word_list, full_output_dir, session_name, args)
    video_postprocessing(full_output_dir, frame_extraction_rate, "output.mp4", session_name, frame_extraction_dir, args.overwrite)

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
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--text_prompts", nargs='+', default=["cdi"], help="Which list of possible detectable words to be used")
    parser.add_argument("--input_dir", type=str, default=frames_path, help="Text file or file path with the list of frames to be processed")
    parser.add_argument("--output_dir", type=str, default=f"{output_folder}/yoloe", help="Path to store outputs at")
    parser.add_argument("--confidence", type=float, default=0.1, help="Confidence threshold for frame annotations.")
    parser.add_argument("--no-tmux", action="store_false", dest="tmux")
    parser.add_argument("--no-conda", action="store_false", dest="use_conda")
    parser.add_argument("--videos", action="store_true")
    session_name_with_random_suffix = f"yoloe_predict_{os.urandom(4).hex()}"
    session_name = session_name_with_random_suffix
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Whether to overwrite existing saved data"
    )
    args = parser.parse_args()
    if args.text_prompts == ["cdi"]:
        word_list = cdi_list()
    elif args.text_prompts == ["vedi"]:
        word_list = vedi_list()
    # use yoloe promptfree mode although it has mixed results
    elif args.text_prompts == ["promptfree"]:
        word_list = []
    elif len(args.text_prompts) == 1 and args.text_prompts[0].endswith(".txt"):
        filename = args.text_prompts[0]
        with open(filename, "r") as file:
            word_list = [line.strip() for line in file if line.strip()]
        args.text_prompts[0] = Path(filename).basename
    else:
        word_list = " ".join(args.text_prompts)
        args.text_prompts[0] = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    full_output_dir = Path(f'{args.output_dir}/{args.text_prompts[0]}')
    if args.input_dir == "10k":
        full_output_dir = Path(f'{full_output_dir}_10k')
        args.input_dir = frames_path_10k
    elif args.input_dir == "1k":
        full_output_dir = Path(f'{full_output_dir}_1k')
        args.input_dir = frames_path_1k
    # if using the default input frames with all of the sampled frames
    elif args.input_dir == frames_path:
        full_output_dir = Path(f'{full_output_dir}_allframes_1fps')
    if args.videos:
        predict_videos(args.input_dir, word_list, full_output_dir, session_name, args)
    else:
        run_parallel_predictions(word_list, full_output_dir, session_name, args)
        

if __name__ == "__main__":
    main()
    
