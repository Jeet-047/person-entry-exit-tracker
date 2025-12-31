import os
import torch
from datetime import datetime

# Paths & config
VIDEO_PATH = "person_entry_exit_test_video.mp4"
YOLO_MODEL = "yolov8n.pt"
LOG_JSON = "entry_exit_log.json"
SAVE_DIR = "logs"
os.makedirs(f"{SAVE_DIR}/entry", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/exit", exist_ok=True)

# Device and model settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.5    # For YOLO person detection
OVERLAP_THRESHOLD = 0.5 # Overlap threshold for door zone
START_TIME = datetime.strptime("09:00:00", "%H:%M:%S")
DEBUG_MODE = False
FRAME_LOOKAHEAD = 30

# Define door zone (x1, y1, x2, y2)
DOOR_BOX = (784, 55, 877, 343)  # Replace with your own

# Model configuration
MODEL_NAME = "resnet50"
EBD_SIMILARITY_THRESHOLD = 0.85 # Cosine similarity threshold for ReID embeddings