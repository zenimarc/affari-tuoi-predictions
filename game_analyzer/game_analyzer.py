import cv2
from ultralytics import YOLO
from settings import VIDEO_DIR
from game_analyzer.model import detect_boxes_yolo, recognize_euros
from concurrent.futures import ProcessPoolExecutor, as_completed


class GameAnalyzer:
    def __init__(self, video_path):
        self.video_path = str(video_path)

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

        interval = int(5 * frame_rate)  # Interval to take a frame every 5 seconds

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

    def preview_key_frames(self):
        """
        Display the key frames extracted from the video.
        """
        frames = self.extract_key_frames(num_workers=8)
        for frame in frames:
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


