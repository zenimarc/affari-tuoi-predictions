from pathlib import Path

# Define base directory relative to this settings file
BASE_DIR = Path(__file__).resolve().parent

# Define other paths relative to BASE_DIR
DATA_DIR = BASE_DIR / 'data'
VIDEO_DIR = DATA_DIR / 'videos'
DETECTION_MODEL_PATH = BASE_DIR / 'yolo-affarituoi.pt'
YOLO_CLASSES = {'accepted_change': 0, 'accepted_offer': 1, 'available_prize': 2, 'change': 3, 'lucky_region_warning': 4, 'offer': 5, 'offer_warning': 6}
FRAME_EXTRACTION_INTERVAL_SECONDS = 4
class DetectionClass:
    ACCEPTED_CHANGE = "accepted_change"
    ACCEPTED_OFFER = "accepted_offer"
    AVAILABLE_PRIZE = "available_prize"
    CHANGE = "change"
    LUCKY_REGION_WARNING = "lucky_region_warning"
    OFFER = "offer"
    OFFER_WARNING = "offer_warning"
