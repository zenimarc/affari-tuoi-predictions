from settings import DetectionClass, POSSIBLE_PRIZES
from game_analyzer.utils import amount_string_to_int
import re

regex = r'^€?\d+(?:\.\d+)?$'

def rounded_num_validator(number: int):
    """normally offers are rounded to the nearest 10, so we'll check if the number is rounded to the nearest 10
    :returns True if the number is rounded to the nearest 10, False otherwise
    """
    if len(str(number)) >= 3:
        # check it ends with a 0
        return str(number)[-1] == '0'
def prizes_validator(text: str):
    if re.match(regex, text):
        amount = amount_string_to_int(text)
        if amount in POSSIBLE_PRIZES:
            return True
    return False

def offer_validator(text: str):
    if re.match(regex, text):
        amount = amount_string_to_int(text)
        validators = [rounded_num_validator, lambda x: x <= max(POSSIBLE_PRIZES)]
        return all(validator(amount) for validator in validators)


def ocr_validator(string, detection_class: DetectionClass):
    if detection_class == DetectionClass.OFFER:
        # if the detection class is OFFER, we'll check just the regex
        return offer_validator(string)
    else:
        return prizes_validator(string)
