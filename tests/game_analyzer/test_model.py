import cv2
import pytest
from game_analyzer.model import detect_boxes_yolo, recognize_digits
from settings import BASE_DIR

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
