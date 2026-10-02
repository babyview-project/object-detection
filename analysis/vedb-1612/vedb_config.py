"""Paths for the VEDB volume-1612 world-camera subset.

Videos stay where they were downloaded. Frames, detections, and embeddings
are written beside them under ``DATA_ROOT`` so the git repo does not hold
JPEGs or shards. Analysis tables and figures live in this folder.
"""

from pathlib import Path

VEDB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VEDB_DIR.parents[1]

# 80-session worldPrivate subset (~178 GB). Do not move these mp4s.
VIDEOS_DIR = Path("/ccn2a/dataset/babyview/related_data/vedb_1612_world")
SUBSET_MANIFEST = VIDEOS_DIR / "world_subset_manifest.json"

# Preprocessing outputs, parallel to BabyView 2025.2.
DATA_ROOT = Path("/ccn2a/dataset/babyview/related_data/vedb_1612")
FRAMES_DIR = DATA_ROOT / "extracted_frames_1fps"
ANNOTATED_FRAMES_DIR = DATA_ROOT / "extracted_frames_1fps_annotated"
DETECTIONS_DIR = DATA_ROOT / "outputs" / "object_detections" / "cdi"
EMBEDDINGS_DIR = DATA_ROOT / "outputs" / "image_embeddings" / "vedb_full"

RESULTS_DIR = VEDB_DIR / "results"
FIGURES_DIR = VEDB_DIR / "figures"

# Same checkpoints and CLIP filter as the manuscript crop pipeline.
CLIP_MODEL = "openai/clip-vit-base-patch32"
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
CLIP_THRESHOLD = 0.27
FPS = 1
VOLUME_ID = 1612
N_SESSIONS = 80
