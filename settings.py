from pathlib import Path

# Define base directory relative to this settings file
BASE_DIR = Path(__file__).resolve().parent

# Define other paths relative to BASE_DIR
DATA_DIR = BASE_DIR / 'data'
VIDEO_DIR = DATA_DIR / 'videos'
DETECTION_MODEL_PATH = BASE_DIR / 'yolo-affarituoi.pt'
YOLO_CLASSES_NAMES_TO_INT = {'accepted_change': 0, 'accepted_offer': 1, 'available_prize': 2, 'change': 3, 'lucky_region_warning': 4, 'offer': 5, 'offer_warning': 6}
YOLO_CLASSES_INT_TO_NAMES = {0: 'accepted_change', 1: 'accepted_offer', 2: 'available_prize', 3: 'change', 4: 'lucky_region_warning', 5: 'offer', 6: 'offer_warning'}
FRAME_EXTRACTION_INTERVAL_SECONDS = 4
class DetectionClass:
    ACCEPTED_CHANGE = "accepted_change"
    ACCEPTED_OFFER = "accepted_offer"
    AVAILABLE_PRIZE = "available_prize"
    CHANGE = "change"
    LUCKY_REGION_WARNING = "lucky_region_warning"
    OFFER = "offer"
    OFFER_WARNING = "offer_warning"

POSSIBLE_PRIZES = {0, 1, 5, 10, 20, 50, 75, 100, 200, 500, 5000, 10000, 15000, 20000, 30000, 50000, 75000, 100000,
                       200000, 300000}
