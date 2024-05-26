from settings import DetectionClass, POSSIBLE_PRIZES
from game_analyzer.utils import amount_string_to_int
import re
from typing import List, Optional, TypedDict

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
    if detection_class == DetectionClass.OFFER or detection_class == DetectionClass.ACCEPTED_OFFER:
        # if the detection class is OFFER, we'll check just the regex
        return offer_validator(string)
    else:
        return prizes_validator(string)



class State(TypedDict):
    seq: int
    available_prizes: List[int]
    offer: Optional[int]
    accepted_offer: Optional[int]
    change: Optional[bool]
    accepted_change: Optional[bool]
    lucky_region_warning: Optional[bool]

def is_valid_state(state: State):
    """
    Check if the game state is valid.

    Parameters:
        state (dict): The game state to check.

    Returns:
        bool: True if the state is valid, False otherwise.
    """
    if state['seq'] is None:
        return False

    if len(state['available_prizes']) > 20:
        return False

    if state['offer'] is not None and state['offer'] > 300000:
        return False

    if state['accepted_offer'] is not None and state['accepted_offer'] > 300000:
        return False

    # only one of the following keys should be not None (offer, accepted_offer...)
    if sum([1 for key in state if (key not in ["available_prizes", "seq"] and state[key] is not None)]) > 1:
        return False

    # check that every prize is in the list of possible prizes
    for prize in state['available_prizes']:
        if prize not in POSSIBLE_PRIZES:
            return False

    # each prize should be unique
    if len(state['available_prizes']) != len(set(state['available_prizes'])):
        return False

    return True

def is_valid_first_state(state: State):
    """
    Check if the game state is valid as the first state.

    Parameters:
        state (dict): The game state to check.

    Returns:
        bool: True if the state is valid, False otherwise.
    """

    if not is_valid_state(state):
        return False

    # check that every prize in the list of possible prizes is in the list of available prizes
    for prize in POSSIBLE_PRIZES:
        if prize not in state['available_prizes']:
            return False

    # check that the available prizes are exactly the maximum number of possible prizes
    if len(state['available_prizes']) != len(POSSIBLE_PRIZES):
        return False

    # all the following keys should be None at the first state
    first_state_keys = [
        'offer',
        'accepted_offer',
        'change',
        'accepted_change',
        'lucky_region_warning'
    ]
    for key in first_state_keys:
        if state[key] is not None:
            return False

    return True

def is_valid_state_wrt_previous(state, previous_state):
    """
    Check if the game state is valid with respect to the previous state.

    Parameters:
        state (dict): The game state to check.
        previous_state (dict): The previous game state.

    Returns:
        bool: True if the state is valid, False otherwise.
    """
    if not is_valid_state(state):
        return False

    # if the number of available prizes has increased, return False
    if len(state.get("available_prizes")) > len(previous_state.get("available_prizes")):
        return False

    # if the number of available prizes has decreased by more than 1, return False
    if len(state.get("available_prizes")) < len(previous_state.get("available_prizes")) - 1:
        return False

    # if a prize was not in the previous state, it should not be in the current state because when a prize is gone it's gone
    for prize in state.get("available_prizes"):
        if prize not in previous_state.get("available_prizes"):
            return False

    # if previous state has an accepted offer, the current state should have the same accepted offer otherwise return False
    if previous_state.get(DetectionClass.ACCEPTED_OFFER) is not None and previous_state.get(DetectionClass.ACCEPTED_OFFER) != state.get(DetectionClass.ACCEPTED_OFFER):
        return False

    # if the current state has an accepted offer, and the previous state have an offer, the amount of the accepted offer should be the same as the offer in the previous state
    if state.get(DetectionClass.ACCEPTED_OFFER) is not None \
            and previous_state.get(DetectionClass.OFFER) is not None \
            and state.get(DetectionClass.ACCEPTED_OFFER) != previous_state.get(DetectionClass.OFFER):
        return False

    return True
