import cv2
import pytest
from game_analyzer.model import detect_boxes_yolo
from game_analyzer.ocr import recognize_euros, choose_text_based_on_proposed_texts_and_detection_class
from game_analyzer.utils import amount_string_to_int
from settings import BASE_DIR, YOLO_CLASSES_NAMES_TO_INT, DetectionClass

def get_class_id_from_name(names, class_name):
    for class_id, name in names.items():
        if name == class_name:
            return class_id
    return None


def test_detect_correct_boxes():
    # Load an image
    image = cv2.imread(BASE_DIR / "tests/tests_data/frame1.jpg")

    # Perform detection
    results = detect_boxes_yolo(image)

    # Check if the number of detected boxes is correct
    assert len(results[0].boxes) == 5  # 5 boxes should be detected

    # Check if a box with class "offer" exists
    offer_class_id = get_class_id_from_name(results[0].names, 'offer')

    change_class_exists = sum(box.cls == offer_class_id for box in results[0].boxes)
    assert change_class_exists == 1, "No box with class 'change' was detected"

    # the number of occurrences of class "available_prize" should be 4:
    available_prize_class_id = get_class_id_from_name(results[0].names, 'available_prize')
    available_prize_count = sum(box.cls == available_prize_class_id for box in results[0].boxes)
    assert available_prize_count == 4, "The number of occurrences of class 'available_prize' is not 4"

def test_ocr_recognize_digits():
    # Load an image
    image = cv2.imread(BASE_DIR / "tests/tests_data/frame1.jpg")
    detections = detect_boxes_yolo(image)
    class_and_number = []
    for detection in detections:
        for box in detection.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers

            roi = image[y1:y2, x1:x2]  # Extract ROI from the image

            recognized_text = recognize_euros(roi, detection.names[int(box.cls)], debug=False)
            recognized_number = amount_string_to_int(recognized_text)
            class_and_number.append({detection.names[int(box.cls)]: recognized_number})

    prizes = [50, 100, 50000, 300000]
    for prize in prizes:
        assert {'available_prize': prize} in class_and_number

    # check that the classes "offer" has number 20000
    assert {'offer': 20000} in class_and_number


def test_ocr_recognize_digits_full():
    # Load an image
    image = cv2.imread(BASE_DIR / "tests/tests_data/frame2.jpg")
    detections = detect_boxes_yolo(image)
    class_and_number = []
    for detection in detections:
        for box in detection.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers

            expand_y = 0
            expand_x = 0
            # expand a bit the roi but check that the roi is not out of the image
            if expand_x != 0 or expand_y != 0:
                y1 = max(0, y1 - expand_y)
                y2 = min(image.shape[0], y2 + expand_y)
                x1 = max(0, x1 - expand_x)
                x2 = min(image.shape[1], x2 + expand_x)


            roi = image[y1:y2, x1:x2]  # Extract ROI from the image

            className = detection.names[int(box.cls)]
            recognized_text = recognize_euros(roi, className, debug=False)
            amount = amount_string_to_int(recognized_text)

            class_and_number.append({className: amount})

    prizes = [0, 1, 5, 10, 20, 50, 75, 100, 200, 500, 5000, 10000, 15000, 20000, 30000, 50000, 75000, 100000,
                       200000, 300000]
    for prize in prizes:
        assert {'available_prize': prize} in class_and_number

def test_ocr_single_image():
    image = cv2.imread(BASE_DIR / "tests/tests_data/prizes/prize_1.jpg")
    class_name = DetectionClass.AVAILABLE_PRIZE
    recognized_text = recognize_euros(image, class_name, debug=True)
    amount = amount_string_to_int(recognized_text)
    assert amount == 1






