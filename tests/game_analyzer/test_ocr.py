import cv2

from game_analyzer.ocr import choose_text_based_on_proposed_texts_and_detection_class, recognize_euros, image_preprocess_4
from game_analyzer.utils import custom_similarity
from settings import DetectionClass, BASE_DIR


def test_choose_text_based_on_proposed_texts_and_detection_class():
    detectOffer = lambda x: choose_text_based_on_proposed_texts_and_detection_class(x, DetectionClass.OFFER)
    detectPrize = lambda x: choose_text_based_on_proposed_texts_and_detection_class(x, DetectionClass.AVAILABLE_PRIZE)

    assert detectOffer(['€20.000', '20.000', '€21.000']) in ['€20.000', '20.000']
    assert detectOffer(['€21.000', '20.000', '€20.000']) in ['€20.000', '20.000']
    assert detectOffer(['€20.001', '20.000', '€21.000']) in ['€20.000', '20.000']

    assert detectPrize(['€20.000', '20.000', '€21.000']) in ['€20.000', '20.000']
    assert detectPrize(['€21.000', '21.000', '€20.000']) in ['€21.000', '21.000']



def test_recognize_simple_digits():
    roi = cv2.imread(str(BASE_DIR / "tests/tests_data/prizes/1/2.jpg"))
    detection_class = DetectionClass.AVAILABLE_PRIZE
    recognized_text = recognize_euros(roi, detection_class, debug=True)
    assert recognized_text in ['€1', '1']

def test_image_preprocess_4():
    image = cv2.imread(str(BASE_DIR / "tests/tests_data/prizes/1/2.jpg"))
    gray_img = image_preprocess_4(image, debug=2)
    assert gray_img is not None
