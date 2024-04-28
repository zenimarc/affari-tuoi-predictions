import cv2
from ultralytics import YOLO
from settings import DETECTION_MODEL_PATH, VIDEO_DIR

# Load the model
model = YOLO(DETECTION_MODEL_PATH)


def detect_and_draw(frame):
    # Perform the detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    return annotated_frame


# Open the video file
cap = cv2.VideoCapture(str(VIDEO_DIR / '21008185_1800.mp4'))

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection and draw bounding boxes and labels
    frame = detect_and_draw(frame)

    # Display the frame
    cv2.imshow('Live View', frame)

    # Wait for a key press to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and destroy all windows
cap.release()
cv2.destroyAllWindows()