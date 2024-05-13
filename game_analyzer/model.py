from ultralytics import YOLO
from settings import DETECTION_MODEL_PATH
import numpy as np


# load the model that can be used by other packages
detection_model = YOLO(DETECTION_MODEL_PATH)

def detect_boxes_yolo(frames: np.ndarray):
    """
    Perform object detection on a frame using YOLOv8.

    Parameters:
        frame (np.ndarray): The frame on which object detection is to be performed.

    Returns:
        The frame with bounding boxes and labels drawn on it.
    """
    # Perform detection
    results = detection_model.predict(source=frames)

    return results









