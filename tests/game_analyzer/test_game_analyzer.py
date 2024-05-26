from game_analyzer.game_analyzer import GameAnalyzer, fast_is_same_state_checksum, is_valid_state, is_valid_state_wrt_previous, amount_string_to_int, are_equivalent_states
from settings import BASE_DIR
import cv2
import pytest


def test_extract_game_state_from_yolo_results():
    # Load the video
    #video_path = BASE_DIR / "tests/tests_data/frame1.jpg"
    video_path = BASE_DIR / "data/videos/21017040_1800.mp4"
    game_analyzer = GameAnalyzer(video_path)

    frames = game_analyzer.extract_key_frames()
    #frames = [cv2.imread(video_path)]
    results = game_analyzer.detect_boxes_on_frames(frames)
    states = game_analyzer.extract_game_states_from_yolo_results(results)
    assert True


def test_preview_keyframes():
    # Load the video
    video_path = BASE_DIR / "data/videos/21008185_1800.mp4"
    game_analyzer = GameAnalyzer(video_path)

    game_analyzer.preview_key_frames()
    assert True

def test_extract_key_frames_and_detect_boxes():
    # Load the video
    video_path = BASE_DIR / "data/videos/21008185_1800.mp4"
    game_analyzer = GameAnalyzer(video_path)

    frames = game_analyzer.extract_key_frames()
    results = game_analyzer.detect_boxes_on_frames(frames[:20])
    assert True

def test_fast_is_same_state_checksum():
    state1 = {
        'seq': 1,
        'available_prizes': [20, 100, 50],
        'offer': 20000,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state2 = {
        'seq': 2,
        'available_prizes': [20, 100, 50],
        'offer': 20000,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state3 = {
        'seq': 3,
        'available_prizes': [20, 100, 50, 10000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state4 = {
        'seq': 4,
        'available_prizes': [1, 1, 1, 1],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state5 = {
        'seq': 5,
        'available_prizes': [1, 1, 2],
        'offer': True,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state6 = {
        'seq': 6,
        'available_prizes': [1, 1, 2],
        'offer': None,
        'accepted_offer': True,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }

    assert fast_is_same_state_checksum(state1, state1)
    assert fast_is_same_state_checksum(state1, state2)
    assert fast_is_same_state_checksum(state3, state4)
    assert fast_is_same_state_checksum(state5, state2)
    assert not fast_is_same_state_checksum(state5, state6)

def test_is_valid_state():
    state = {
        'seq': 1,
        'available_prizes': [1, 1, 2],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert not is_valid_state(state)
    state = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert is_valid_state(state)
    state = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': 300001,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert not is_valid_state(state)
    state = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': 200000,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert is_valid_state(state)
    state = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': 200000,
        'accepted_offer': None,
        'change': True,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert not is_valid_state(state)
    state = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': None,
        'accepted_offer': None,
        'change': True,
        'accepted_change': True,
        'lucky_region_warning': None,
    }
    assert not is_valid_state(state)


def test_is_valid_state_wrt_previous():
    state1 = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': 20000,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state2 = {
        'seq': 2,
        'available_prizes': [100, 200, 300000],
        'offer': None,
        'accepted_offer': 30000,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert not is_valid_state_wrt_previous(state2, state1)

    state1 = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state2 = {
        'seq': 2,
        'available_prizes': [100, 200, 300000, 20000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert not is_valid_state_wrt_previous(state2, state1)

    state1 = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': 20000,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state2 = {
        'seq': 2,
        'available_prizes': [100, 200],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert is_valid_state_wrt_previous(state2, state1)

    state1 = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state2 = {
        'seq': 2,
        'available_prizes': [100],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert not is_valid_state_wrt_previous(state2, state1)

def test_are_equivalent_states():
    state1 = {
        'seq': 1,
        'available_prizes': [100, 200, 300000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state2 = {
        'seq': 2,
        'available_prizes': [100, 200, 300000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert are_equivalent_states(state2, state1)

    state1 = {
        'seq': 1,
        'available_prizes': [200, 300000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    state2 = {
        'seq': 2,
        'available_prizes': [100, 200, 300000],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }
    assert not are_equivalent_states(state2, state1)

def test_amount_string_to_int():
    assert amount_string_to_int("€20.000") == 20000
    assert amount_string_to_int("€20") == 20
    assert amount_string_to_int("€200.000") == 200000
    assert amount_string_to_int("€0") == 0










