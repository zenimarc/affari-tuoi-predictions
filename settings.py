from pathlib import Path

# Define base directory relative to this settings file
BASE_DIR = Path(__file__).resolve().parent

# Define other paths relative to BASE_DIR
DATA_DIR = BASE_DIR / 'data'
VIDEO_DIR = DATA_DIR / 'videos'
DETECTION_MODEL_PATH = BASE_DIR / 'yolo-affarituoi.pt'