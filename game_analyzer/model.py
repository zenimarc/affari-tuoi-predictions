from ultralytics import YOLO
from settings import DETECTION_MODEL_PATH, DetectionClass, POSSIBLE_PRIZES
import easyocr
import cv2
import numpy as np
from game_analyzer.validators import ocr_validator
from game_analyzer.utils import levenshtein_distance

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
def recognize_euros(roi: np.ndarray, detection_class: DetectionClass, debug=False) -> str:
    """
    Perform OCR on a single image region (ROI) to recognize digits.

    Parameters:
        roi (np.ndarray): The image region of interest where digits are to be recognized.

    Returns:
        str: A string containing the recognized digits separated by spaces.
    """

    recognized_text = robust_ocr_extraction(roi, detection_class, debug)

    return recognized_text


def image_preprocess(img, debug=False):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image to isolate colors within the mask
    res = cv2.bitwise_and(img, img, mask=mask)

    # Convert the result to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image to get black text on a white background
    processed_img = cv2.bitwise_not(gray)

    # Display the processed image if debug mode is on
    if debug:
        cv2.imshow('Original Image', img)
        cv2.imshow('HSV Mask', mask)
        cv2.imshow('Resulting Image', res)
        cv2.imshow('Processed Image', processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_img

def image_preprocess_2(img, debug=False):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce initial noise
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)  # Smaller blur to preserve more details

    # Adjust the threshold value if needed
    threshold_value = 210
    _, processed_img = cv2.threshold(blur_img, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Keep the kernel size and morphological operations mild
    kernel_size = 1  # Keeping kernel small to avoid aggressive changes
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    #Apply morphological operations
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel, iterations=1)
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    if debug:
        cv2.imshow('Threshold Image', processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_img


def robust_ocr_extraction(image, detection_class: DetectionClass, debug=False):
    # Prepare different preprocessing methods
    preprocessors = [
        lambda img, _: img,
        image_preprocess,
        image_preprocess_2
    ]
    proposed_texts = []
    for preprocessor in preprocessors:
        processed_img = preprocessor(image, debug)
        text = reader.readtext(processed_img, detail=0, allowlist='0123456789€.')
        recognized_text = ' '.join(text)
        proposed_texts.append(recognized_text)

    chosen_text = choose_text_based_on_proposed_texts_and_detection_class(proposed_texts, detection_class)

    return chosen_text


def choose_text_based_on_proposed_texts_and_detection_class(proposed_texts, detection_class: DetectionClass):
    valid_texts = []
    invalid_texts = []
    for text in proposed_texts:
        if ocr_validator(text, detection_class):
            valid_texts.append(text)
        else:
            invalid_texts.append(text)

    proposals = get_counting_dict_from_proposals(valid_texts)
    max_votes_proposals = [proposal for proposal in proposals if proposals[proposal] == max(proposals.values())]

    chosen_text = None
    if len(max_votes_proposals) == 1:
        chosen_text = max_votes_proposals[0]
    elif len(max_votes_proposals) > 1:
        chosen_text = min(max_votes_proposals,
                          key=lambda x: sum(levenshtein_distance(x, other) for other in invalid_texts))

    return chosen_text
def get_counting_dict_from_proposals(proposals):
    counting_dict = {}
    for proposal in proposals:
        if proposal in counting_dict:
            counting_dict[proposal] += 1
        else:
            counting_dict[proposal] = 1

    return counting_dict




