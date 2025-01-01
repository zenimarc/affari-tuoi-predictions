import cv2

from game_analyzer.ocr import choose_text_based_on_proposed_texts_and_detection_class, recognize_euros, image_preprocess_4
from game_analyzer.utils import custom_similarity
from settings import DetectionClass, BASE_DIR


def test_choose_text_based_on_proposed_texts_and_detection_class():
    detectOffer = lambda x: choose_text_based_on_proposed_texts_and_detection_class(x, DetectionClass.OFFER)
    detectPrize = lambda x: choose_text_based_on_proposed_texts_and_detection_class(x, DetectionClass.AVAILABLE_PRIZE)

    assert detectOffer(['€20.000', '20.000', '€21.000']) == '20000'
    assert detectOffer(['€21.000', '20.000', '€20.000']) == '20000'
    assert detectOffer(['€20.001', '20.000', '€21.000']) == '20000'


    assert detectPrize(['€20.000', '20.000', '€21.000']) == '20000'
    assert detectPrize(['€21.000', '21.000', '€20.000']) == '20000'
    assert detectPrize(['€ 75.000', "'€ 75.000", '€ 75.000', '€ 75.000', '15008', '15008']) == '75000'
    assert detectPrize(['€ 75.000', '€ 75.000', '15000', '15000', '75000']) == '75000'



def test_recognize_simple_digits():
    roi = cv2.imread(str(BASE_DIR / "tests/tests_data/prizes/1/3.jpg"))
    detection_class = DetectionClass.AVAILABLE_PRIZE
    recognized_text = recognize_euros(roi, detection_class, debug=2)
    assert recognized_text == '1'

def test_image_preprocess_4():
    image = cv2.imread(str(BASE_DIR / "tests/tests_data/prizes/75000/1.jpg"))
    gray_img = image_preprocess_4(image, debug=2)
    assert gray_img is not None
