# YOLOe 
Much of the code is borrowed from the original [YOLOe repo](https://github.com/THU-MIG/yoloe) which includes instructions to [set up your environment](https://github.com/THU-MIG/yoloe?tab=readme-ov-file#installation). The instructions below includes an adapted version of those instructions. A [conda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) virtual environment is recommended.

## Installation

```
git clone https://github.com/babyview-project/object-detection
cd object-detection/yoloe
conda create -n objects python=3.10
conda activate objects
conda install ffmpeg
pip install ultralytics pandas tqdm pillow ray
```
It is also recommended to install [tmux](https://tmuxcheatsheet.com/how-to-install-tmux/) if you do not have tmux on your system.
While not recommended, it is also possible to also proceed without conda. Instructions to install ffmpeg without conda are [here](https://www.ffmpeg.org/download.html).

To extract frames from a video directory (the example video directory for example), you can run:
```
python extract_frames.py --videos_dir examples
```

To then run a demo on the extracted frames, you can run:
```
python predict_frames --input_dir extracted_frames --output_dir predicted_frames --overwrite
```

Or, you can directly use an example video with:
```
python predict_frames.py --input_dir examples --output_dir predict --confidence 0.5 --overwrite --videos
```

To run a demo on 1000 frames on the CCN server, you can run:
```
python predict_frames.py --input_dir 1k --output_dir yoloe
```

To run the full pipeline on all of the sampled frames, you can run:

```
python predict_frames.py --input_dir output_dir /ccn2/dataset/babyview/outputs_20250312 --device_ids [0,1,2,3,4,5,6,7] --num_parallel 8
```
