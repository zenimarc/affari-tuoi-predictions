from game_analyzer.game_analyzer import GameAnalyzer
from settings import BASE_DIR
import pytest

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


