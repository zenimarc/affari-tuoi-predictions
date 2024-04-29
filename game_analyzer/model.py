from ultralytics import YOLO
from settings import DETECTION_MODEL_PATH
import easyocr
import cv2
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



reader = easyocr.Reader(['en'], gpu=True)
def recognize_euros(roi: np.ndarray) -> str:
    """
    Perform OCR on a single image region (ROI) to recognize digits.

    Parameters:
        roi (np.ndarray): The image region of interest where digits are to be recognized.

    Returns:
        str: A string containing the recognized digits separated by spaces.
    """
    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(grayscale)
    # Perform OCR on the single ROI
    text = reader.recognize(hist, detail=0, allowlist='0123456789€.')


    # Join detected digits into a single string
    recognized_text = ' '.join(text)

    return recognized_text


