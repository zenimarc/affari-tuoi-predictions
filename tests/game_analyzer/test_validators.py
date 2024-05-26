import cv2

from game_analyzer.game_analyzer import GameAnalyzer
from game_analyzer.validators import ocr_validator, is_valid_first_state
from settings import DetectionClass, POSSIBLE_PRIZES, BASE_DIR


def test_ocr_validator():
    """TEST validators for classes OFFER and AVAILABLE_PRIZE"""
    # Test AVAILABLE_PRIZE
    assert ocr_validator('€20.000', DetectionClass.AVAILABLE_PRIZE)
    assert ocr_validator('75', DetectionClass.AVAILABLE_PRIZE)
    assert ocr_validator('€20000', DetectionClass.AVAILABLE_PRIZE)
    assert ocr_validator('20000', DetectionClass.AVAILABLE_PRIZE)

    for prize in POSSIBLE_PRIZES:
        assert ocr_validator(f'€{prize}', DetectionClass.AVAILABLE_PRIZE)
        assert ocr_validator(str(prize), DetectionClass.AVAILABLE_PRIZE)

    assert not ocr_validator('750', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('€20.001', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('€20.00.0', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('20.00.0', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('€20.000.000', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('€20.000.001', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('€20.0.01', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('€400000', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('400000', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('20500', DetectionClass.AVAILABLE_PRIZE)

    # Test OFFER
    assert ocr_validator('€20.000', DetectionClass.OFFER)
    assert ocr_validator('20500', DetectionClass.OFFER)
    assert ocr_validator('20000', DetectionClass.OFFER)
    assert not ocr_validator('€400000', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('400000', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('€20.001', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('€20.00.0', DetectionClass.AVAILABLE_PRIZE)
    assert not ocr_validator('20.00.0', DetectionClass.AVAILABLE_PRIZE)


def test_first_state_validator():
    # Load an image
    image = cv2.imread(BASE_DIR / "tests/tests_data/frame_fullprizes.jpg")
    # Perform detection
    ga = GameAnalyzer(image)
    results = ga.detect_boxes_on_frames(image)
    state = ga.extract_single_game_state_from_yolo_result(results[0], idx=0)
    assert is_valid_first_state(state)







