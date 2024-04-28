import cv2
from ultralytics import YOLO
from settings import VIDEO_DIR
from model import detection_model, recognize_digits

class GameAnalyzer:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.model = detection_model
