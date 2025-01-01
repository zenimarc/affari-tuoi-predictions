import json
import os

import cv2
from ultralytics import YOLO
from settings import VIDEO_DIR, YOLO_CLASSES_NAMES_TO_INT, FRAME_EXTRACTION_INTERVAL_SECONDS, DetectionClass, POSSIBLE_PRIZES, YOLO_CLASSES_INT_TO_NAMES
from game_analyzer.model import detect_boxes_yolo
from game_analyzer.ocr import recognize_euros
from concurrent.futures import ProcessPoolExecutor, as_completed
from game_analyzer.utils import amount_string_to_int
from game_analyzer.validators import is_valid_state, is_valid_state_wrt_previous, is_valid_first_state, State, \
    ocr_validator


# define a InvalidFrameError exception
class InvalidFrameError(Exception):
    """If the frame is invalid, raise this exception."""
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class GameAnalyzer:
    def __init__(self, video_path, debug=0):
        self.debug = debug
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

    def extract_single_game_state_from_yolo_result(self, result, idx, debug=0, threshold=0.80):
        """
        Extract the game state from the YOLOv8 detection results.

        Parameters:
            result: The detection results from YOLOv8.
            idx: The index of the frame.
            debug: Whether to display debug information.
            threshold: The confidence threshold for YOLO detections.

        Returns:
            dict: A dictionary containing the game state.
        """

        # debug could be set at self level or at the function level
        if debug == 0:
            debug = self.debug

        try:
            # Skip if no boxes are detected
            if len(result.boxes) == 0:
                return None

            # Filter boxes based on the confidence threshold
            boxes = [box for box in result.boxes if box.conf > threshold]
            if len(boxes) == 0:
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
            for box in boxes:
                try:
                    if box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.AVAILABLE_PRIZE]:
                        recognized_text = extract_string_from_box(box, result.orig_img, debug=debug)
                        if recognized_text is None:
                            raise ValueError("OCR failed for AVAILABLE_PRIZE")
                        amount = amount_string_to_int(recognized_text)
                        if amount not in POSSIBLE_PRIZES:
                            raise ValueError(f"Invalid prize amount: {amount}")
                        state["available_prizes"].append(amount)

                    elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.OFFER]:
                        recognized_text = extract_string_from_box(box, result.orig_img, debug=debug)
                        if recognized_text is None:
                            raise ValueError("OCR failed for OFFER")
                        if not ocr_validator(recognized_text, DetectionClass.OFFER):
                            raise ValueError(f"Invalid offer amount: {recognized_text}")
                        state['offer'] = amount_string_to_int(recognized_text)

                    elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.ACCEPTED_OFFER]:
                        recognized_text = extract_string_from_box(box, result.orig_img, debug=debug)
                        if recognized_text is None:
                            raise ValueError("OCR failed for ACCEPTED_OFFER")
                        if not ocr_validator(recognized_text, DetectionClass.ACCEPTED_OFFER):
                            raise ValueError(f"Invalid offer amount: {recognized_text}")
                        state['accepted_offer'] = amount_string_to_int(recognized_text)

                    elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.CHANGE]:
                        state['change'] = True

                    elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.ACCEPTED_CHANGE]:
                        state['accepted_change'] = True

                    elif box.cls == YOLO_CLASSES_NAMES_TO_INT[DetectionClass.LUCKY_REGION_WARNING]:
                        state['lucky_region_warning'] = True

                except Exception as e:
                    if debug > 0:
                        self._save_debug_info(idx, result, str(e), box)
                    raise InvalidFrameError(f"Invalid frame: {str(e)}")

            # Sort the available prizes
            state["available_prizes"] = sorted(state["available_prizes"])

        except InvalidFrameError as e:
            print(f"Invalid frame: {str(e)}")
            return None

        return state

    def _save_debug_info(self, idx, result, error_message, box=None):
        """
        Save debug information for a failed detection.

        Parameters:
            idx: The index of the frame.
            result: The detection result object.
            error_message: The error message describing the failure.
            box: (Optional) The specific box that caused the error.
        """
        # Create a debug folder for the specific frame
        debug_folder = f"debug/frame_{idx}"
        os.makedirs(debug_folder, exist_ok=True)

        # Save the full image
        cv2.imwrite(os.path.join(debug_folder, "full_image.jpg"), result.orig_img)

        # Annotate and save the image
        annotated_img = result.orig_img.copy()
        for b in result.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            label = f"{YOLO_CLASSES_INT_TO_NAMES[int(b.cls)]} ({float(b.conf):.2f})"
            color = (0, 255, 0) if b == box else (0, 0, 255)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(os.path.join(debug_folder, "annotated_image.jpg"), annotated_img)

        # Save a cropped ROI if a specific box caused the error
        if box:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = result.orig_img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(debug_folder, "roi.jpg"), roi)

        # Save the error message to a log file
        log_data = {
            "error_message": error_message,
            "boxes": [
                {
                    "cls": int(b.cls),
                    "conf": float(b.conf),
                    "coordinates": list(map(int, b.xyxy[0]))
                } for b in result.boxes
            ]
        }
        with open(os.path.join(debug_folder, "log.json"), "w") as log_file:
            json.dump(log_data, log_file, indent=4)

    def extract_game_states_from_yolo_results(self, results: list):
        """
        Extract the game state from the YOLOv8 detection results.

        Parameters:
            results: The detection results from YOLOv8.

        Returns:
            dict: A dictionary containing the game state.
        """
        for idx, result in enumerate(results):
            state = self.extract_single_game_state_from_yolo_result(result, idx, debug=self.debug)
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

    def save_states_to_json(self, states, output_file):
        """
        Save the game states to a JSON file.

        Parameters:
            states: The list of game states to save.
            output_file: The output JSON file to save the states.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write the states to the file
        with open(output_file, "w") as f:
            json.dump(states, f, indent=4)




def extract_string_from_box(box, image, debug=0):
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
    recognized_text = recognize_euros(roi, className, debug)
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


