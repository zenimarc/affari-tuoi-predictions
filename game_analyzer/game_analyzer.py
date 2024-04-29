import cv2
from ultralytics import YOLO
from settings import VIDEO_DIR, YOLO_CLASSES, FRAME_EXTRACTION_INTERVAL_SECONDS, DetectionClass
from game_analyzer.model import detect_boxes_yolo, recognize_euros
from concurrent.futures import ProcessPoolExecutor, as_completed


POSSIBLE_PRIZES = {0, 1, 5, 10, 20, 50, 75, 100, 200, 500, 5000, 10000, 15000, 20000, 30000, 50000, 75000, 100000,
                       200000, 300000}

class GameAnalyzer:
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.states = []

    def extract_frames_at_intervals(self, video_path, start, end, interval):
        cap = cv2.VideoCapture(video_path)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_id = start

        while frame_id <= end:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_id - start) % interval == 0:
                frames.append(frame)
            frame_id += 1

        cap.release()
        return frames

    def extract_key_frames(self, num_workers=4):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        interval = int(FRAME_EXTRACTION_INTERVAL_SECONDS * frame_rate)  # Interval to take a frame every N seconds

        # Divide the total number of frames into chunks for each worker
        frames_per_worker = total_frames // num_workers
        ranges = [(i * frames_per_worker, min((i + 1) * frames_per_worker - 1, total_frames)) for i in
                  range(num_workers)]

        frames = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(self.extract_frames_at_intervals, self.video_path, start, end, interval) for start, end in
                       ranges]

            # Collect and concatenate results in order
            for future in futures:
                frames.extend(future.result())

        return frames


    def detect_boxes_on_frames(self, frames):
        """
        Detect boxes on a list of frames using YOLOv8.

        Parameters:
            frames: A list of frames on which object detection is to be performed.

        Returns:
            list: A list of frames with bounding boxes and labels drawn on them.
        """
        # Perform detection on each frame
        results = detect_boxes_yolo(frames)

        return results

    def extract_game_state_from_yolo_results(self, results):
        """
        Extract the game state from the YOLOv8 detection results.

        Parameters:
            results: The detection results from YOLOv8.

        Returns:
            dict: A dictionary containing the game state.
        """

        for result in results:
            state = {
                'seq': None,
                'available_prizes': [],
                'offer': None,
                'accepted_offer': None,
                'change': None,
                'accepted_change': None,
                'lucky_region_warning': None,
            }

            if len(result.boxes) == 0:
                continue

            if len(self.states) > 1 and fast_is_same_state_checksum(state, self.states[-1]):
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers

                roi = result.orig_image[y1:y2, x1:x2]  # Extract ROI from the image

                recognized_text = recognize_euros(roi)

    def preview_key_frames(self):
        """
        Display the key frames extracted from the video.
        """
        frames = self.extract_key_frames(num_workers=8)
        for frame in frames:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()




def are_equivalent_states(state1, state2):
    """
    Compare two game states and return the differences.

    Parameters:
        state1 (dict): The first game state to compare.
        state2 (dict): The second game state to compare.

    Returns:
        bool: True if the states are the same, False otherwise.
    """
    state1_except_seq = {key: value for key, value in state1.items() if key != 'seq'}
    state2_except_seq = {key: value for key, value in state2.items() if key != 'seq'}

    return state1_except_seq == state2_except_seq


def fast_is_same_state_checksum(state1, state2):
    """
    Compare two game states and return if they are the same. But it's like a checksum, not a deep comparison.

    Parameters:
        state1 (dict): The first game state to compare.
        state2 (dict): The second game state to compare.

    Returns:
        bool: True if the states are the same, False otherwise.
    """
    for key in state1:
        if key == 'seq' or key == DetectionClass.AVAILABLE_PRIZES:
            continue
        elif bool(state1.get(key)) != bool(state2.get(key)):
            return False

    if len(state1.get(DetectionClass.AVAILABLE_PRIZES)) != len(state2.get(DetectionClass.AVAILABLE_PRIZES)):
        return False

    return True

def is_valid_state(state):
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

    # if multiple keys are thurthy return False
    if sum([1 for key in state if (key not in [DetectionClass.AVAILABLE_PRIZES, "seq"] and state[key] is not None)]) > 1:
        return False

    # check that every prize is in the list of possible prizes
    for prize in state['available_prizes']:
        if prize not in POSSIBLE_PRIZES:
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

    # if the number of available prizes has decreased, return False
    if len(state.get(DetectionClass.AVAILABLE_PRIZES)) > len(previous_state.get(DetectionClass.AVAILABLE_PRIZES)):
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


