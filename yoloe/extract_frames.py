'''
Script to create a dataset of image frames from videos


python scripts/create_frames_from_videos.py \
    --videos_dir "/ccn2/dataset/babyview/unzip_2025/" \
    --output_dir "/ccn2/dataset/babyview/sampled_frames_ms" \
        
'''

import os
import argparse
import subprocess
import ray
import glob

def get_args():
    parser = argparse.ArgumentParser(description='Process videos to the desired fps, resolution, rotation.')
    parser.add_argument('--videos_dir', type=str, required=True, help='Path to the directory with videos')
    parser.add_argument('--output_dir', type=str, default="extracted_frames", help='Path to the directory to save frames')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of parallel processes')
    parser.add_argument('--fps', type=int, default=1, help='FPS to extract frames at')
    return parser.parse_args()

def extract_frames_from_video(args, video_path):
    output_pattern = '%08d.jpg'
    video_name = os.path.basename(video_path).split('.')[0]
    output_dir = os.path.join(args.output_dir, video_name)
    output_path = os.path.join(output_dir, output_pattern)
    os.makedirs(output_dir, exist_ok=True)

    # I think I may have used this scale_filter to resize the images to 512x512
    # scale_filter = "scale='if(gte(iw,ih),-1,512):if(gte(iw,ih),512,-1)'"
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={args.fps}',
        '-qscale:v', '1',
        '-video_track_timescale', '1000',
        '-frame_pts', '1',  # This names files with the frame PTS value
        output_path
    ]
    print(cmd)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

@ray.remote
def extract_frames_from_video_remote(args, video_file):
    extract_frames_from_video(args, video_file)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    suffix = '*.[mM][pP][4]'
    video_files = glob.glob(os.path.join(args.videos_dir, f'**/{suffix}'), recursive=True)
    print(f"Processing {len(video_files)} videos") 

    # Parallelize the script
    ray.init(num_cpus=args.num_processes)
    futures = [extract_frames_from_video_remote.remote(args, video_file) for video_file in video_files]
    ray.get(futures)
    
if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
    