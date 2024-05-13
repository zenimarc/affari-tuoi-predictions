
from game_analyzer.ocr import choose_text_based_on_proposed_texts_and_detection_class
from game_analyzer.utils import custom_similarity
from settings import DetectionClass
def test_choose_text_based_on_proposed_texts_and_detection_class():
    detectOffer = lambda x: choose_text_based_on_proposed_texts_and_detection_class(x, DetectionClass.OFFER)
    detectPrize = lambda x: choose_text_based_on_proposed_texts_and_detection_class(x, DetectionClass.AVAILABLE_PRIZE)

    assert detectOffer(['€20.000', '20.000', '€21.000']) in ['€20.000', '20.000']
    assert detectOffer(['€21.000', '20.000', '€20.000']) in ['€20.000', '20.000']
    assert detectOffer(['€20.001', '20.000', '€21.000']) in ['€20.000', '20.000']