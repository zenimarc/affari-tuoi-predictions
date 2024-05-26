import cv2
from ultralytics import YOLO
from settings import VIDEO_DIR, YOLO_CLASSES_NAMES_TO_INT, FRAME_EXTRACTION_INTERVAL_SECONDS, DetectionClass, POSSIBLE_PRIZES, YOLO_CLASSES_INT_TO_NAMES
from game_analyzer.model import detect_boxes_yolo
from game_analyzer.ocr import recognize_euros
from concurrent.futures import ProcessPoolExecutor, as_completed
from game_analyzer.utils import amount_string_to_int
from game_analyzer.validators import is_valid_state, is_valid_state_wrt_previous, is_valid_first_state, State



# define a InvalidFrameError exception
class InvalidFrameError(Exception):
    """If the frame is invalid, raise this exception."""
    def __init__(self, message):
        self.message = message
        super().__init__(message)


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
        #total_frames = 4000
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

    def extract_single_game_state_from_yolo_result(self, result, idx):
        """
        Extract the game state from the YOLOv8 detection results.

        Parameters:
            result: The detection results from YOLOv8.
            idx: The index of the frame.


        Returns:
            dict: A dictionary containing the game state.
        """

        try:
            # Skip if no boxes are detected
            if len(result.boxes) == 0:
                return None

            # Extract the meta state from the YOLOv8 result (which is a state but without doing OCR on all the boxes)
            meta_state = get_meta_state_from_yolo_result(result)
            meta_state['seq'] = idx

            # Skip if the state is the same as the previous state (fast comparing using meta state, so no check on the content of the boxes)
            if len(self.states) > 1 and fast_is_same_state_checksum(meta_state, self.states[-1]):
                return None

            # from now on, extract the full state by doing OCR on all the boxes
            state: State = {
                'seq': idx,
                'available_prizes': [],
                'offer': None,
                'accepted_offer': None,
                'change': None,
                'accepted_change': None,
                'lucky_region_warning': None,
            }
            for box in result.boxes:
                if box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.AVAILABLE_PRIZE]:
                    recognized_text = extract_string_from_box(box, result.orig_img)
                    if recognized_text == None:
                        raise InvalidFrameError(f"Invalid frame")
                    try: # Try to convert the recognized text to an integer
                        amount = amount_string_to_int(recognized_text)
                        if amount not in POSSIBLE_PRIZES:
                            print(f"Invalid frame with amount {amount} from recognized text {recognized_text}")
                            raise InvalidFrameError(f"Invalid frame with amount {amount} from recognized text {recognized_text}")
                        state["available_prizes"].append(amount)
                    except Exception as e:
                        # x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
                        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
                        #roi = result.orig_img[y1:y2, x1:x2]  # Extract ROI from the image
                        # roi = result.orig_img
                        # save image to disk
                        #cv2.imwrite(f"invalid_frame_{idx}.jpg", roi)
                        # cv2.imshow('frame', roi)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        raise InvalidFrameError(f"Invalid frame")

                elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.OFFER]:
                    recognized_text = extract_string_from_box(box, result.orig_img)
                    if recognized_text == None:
                        raise InvalidFrameError(f"Invalid frame")
                    offer = amount_string_to_int(recognized_text)
                    state['offer'] = offer

                elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.ACCEPTED_OFFER]:
                    recognized_text = extract_string_from_box(box, result.orig_img)
                    if recognized_text == None:
                        raise InvalidFrameError(f"Invalid frame")
                    accepted_offer = amount_string_to_int(recognized_text)
                    state['accepted_offer'] = accepted_offer

                elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.CHANGE]:
                    state['change'] = True

                elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.ACCEPTED_CHANGE]:
                    state['accepted_change'] = True

                elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.LUCKY_REGION_WARNING]:
                    state['lucky_region_warning'] = True

            # sort the available prizes
            state["available_prizes"] = sorted(state["available_prizes"])
        except InvalidFrameError:
            return None

        return state

    def extract_game_states_from_yolo_results(self, results: list):
        """
        Extract the game state from the YOLOv8 detection results.

        Parameters:
            results: The detection results from YOLOv8.

        Returns:
            dict: A dictionary containing the game state.
        """
        for idx, result in enumerate(results):
            state = self.extract_single_game_state_from_yolo_result(result, idx)
            if state is not None:
                # Append the state to the list of states if it is valid with respect to the previous state
                if len(self.states) > 0:
                    if not are_equivalent_states(state, self.states[-1]) and is_valid_state_wrt_previous(state, self.states[-1]):
                        self.states.append(state)
                # if it is the first state, just append it if it is a valid first state
                else:
                    if is_valid_first_state(state):
                        self.states.append(state)

        return self.states


    def preview_key_frames(self):
        """
        Display the key frames extracted from the video.
        """
        frames = self.extract_key_frames(num_workers=8)
        for frame in frames:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()




def extract_string_from_box(box, image):
    """
    Extract the string from a box in an image.

    Parameters:
        box: The box from which to extract the string.
        image: The image from which to extract the string.

    Returns:
        str: The string extracted from the box.
    """
    x1, y1, x2, y2 = box.xyxy[0]  # Extract bounding box coordinates
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
    roi = image[y1:y2, x1:x2]  # Extract ROI from the image
    className = YOLO_CLASSES_INT_TO_NAMES[int(box.cls)]
    recognized_text = recognize_euros(roi, className)
    return recognized_text



def are_equivalent_states(state1: State, state2: State):
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


def fast_is_same_state_checksum(state1: State, state2: State):
    """
    Compare two game states and return if they are the same. But it's like a checksum, not a deep comparison.

    Parameters:
        state1 (dict): The first game state to compare.
        state2 (dict): The second game state to compare.

    Returns:
        bool: True if the states are the same, False otherwise.
    """
    for key in state1:
        if key == 'seq' or key == "available_prizes":
            continue
        elif bool(state1.get(key)) != bool(state2.get(key)):
            return False

    if len(state1.get("available_prizes")) != len(state2.get("available_prizes")):
        return False

    return True



def get_meta_state_from_yolo_result(result):
    """
    Extract the meta state from the YOLOv8 detection result.

    Parameters:
        result: The detection result from YOLOv8.

    Returns:
        dict: A dictionary containing the meta state. which is a state but without doing ocr on all the boxes
    """
    meta_state = {
        'seq': None,
        'available_prizes': [],
        'offer': None,
        'accepted_offer': None,
        'change': None,
        'accepted_change': None,
        'lucky_region_warning': None,
    }

    for box in result.boxes:
        if box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.AVAILABLE_PRIZE]:
            meta_state['available_prizes'].append(1)
        elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.OFFER]:
            meta_state['offer'] = True
        elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.ACCEPTED_OFFER]:
            meta_state['accepted_offer'] = True
        elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.CHANGE]:
            meta_state['change'] = True
        elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.ACCEPTED_CHANGE]:
            meta_state['accepted_change'] = True
        elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.LUCKY_REGION_WARNING]:
            meta_state['lucky_region_warning'] = True

    return meta_state


