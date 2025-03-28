# YOLOe 
Much of the code is borrowed from the original [YOLOe repo](https://github.com/THU-MIG/yoloe) which includes instructions to [set up your environment](https://github.com/THU-MIG/yoloe?tab=readme-ov-file#installation).

To run a demo on 1000 frames locally, you can run:

```
python predict_frames.py --input_frames 1k --output_path yoloe
```

To run a demo on our example video, you can run:

```
python predict_frames.py --input_frames examples/example_clip.mp4 --output_path predict/example_clip --overwrite
```

To run the full pipeline on all of the sampled frames, you can run:

```
python predict_frames.py --device_ids [0,1,2,3,4,5,6,7] --num_parallel 8
```
