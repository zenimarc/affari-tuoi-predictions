from collections import Counter

import easyocr
import cv2

from settings import DetectionClass
import numpy as np
from game_analyzer.validators import ocr_validator
from game_analyzer.utils import levenshtein_distance, custom_similarity, amount_string_to_int

reader = easyocr.Reader(['en'], gpu=True)
def recognize_euros(roi: np.ndarray, detection_class: DetectionClass, debug=0) -> str | None:
    """
    Perform OCR on a single image region (ROI) to recognize digits.

    Parameters:
        roi (np.ndarray): The image region of interest where digits are to be recognized.

    Returns:
        str: A string containing the recognized digits separated by spaces. If no digits are recognized, returns None.
        (No validation is performed here)
    """

    recognized_text = robust_ocr_extraction(roi, detection_class, debug)

    return recognized_text


def image_preprocess(img, debug=0):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image to isolate colors within the mask
    res = cv2.bitwise_and(img, img, mask=mask)

    # Convert the result to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image to get black text on a white background
    processed_img = cv2.bitwise_not(gray)

    scale_factor = 2
    upscaled = cv2.resize(processed_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    processed_img = cv2.blur(upscaled, (3, 3))

    # Display the processed image if debug mode is on
    if debug > 1:
        cv2.imshow('Processed Image', processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_img

def image_preprocess_2(img, debug=0):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce initial noise
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)  # Smaller blur to preserve more details

    # Adjust the threshold value if needed
    threshold_value = 210
    _, processed_img = cv2.threshold(blur_img, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Keep the kernel size and morphological operations mild
    kernel_size = 1  # Keeping kernel small to avoid aggressive changes
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    #Apply morphological operations
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel, iterations=1)
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    scale_factor = 2
    upscaled = cv2.resize(processed_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    processed_img = cv2.blur(upscaled, (3, 3))

    if debug > 1:
        cv2.imshow('Threshold Image', processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_img

def image_preprocess_3(img, debug=0):
    # crop the image even more by 5 pixels on each side
    img = img[5:-5, 5:-5]
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_factor = 2
    upscaled = cv2.resize(gray_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    gray_img = cv2.blur(upscaled, (5, 5))


    if debug > 1:
        cv2.imshow('Threshold Image', gray_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return gray_img



def image_preprocess_4(image: np.ndarray, debug=0) -> np.ndarray:
    """
    Preprocesses an image containing euro amounts to prepare it for EasyOCR.

    Parameters:
    - image (np.ndarray): The input image as a numpy array.
    - debug (int): Debug level (0: No debug, 1: Minimal debug, 2: Show processed steps).

    Returns:
    - np.ndarray: The preprocessed image ready for OCR.
    """
    # Step 1: Crop edges to remove potential border noise
    #image = image[6:-6, 6:-6]

    MIN_W_RELEVANT = 6
    MIN_H_RELEVANT = 15

    # Step 2: Convert to HSV and isolate white regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    isolated = cv2.bitwise_and(image, image, mask=mask)

    # Step 3: Convert to grayscale and invert
    gray = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)

    if debug > 1:
        cv2.imshow("Step 3: Inverted Image", inverted)
        cv2.waitKey(0)

    # Step 4: binarize the image using otsu thresholding
    binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if debug > 1:
        cv2.imshow("Step 4: Binarized Image", binary)
        cv2.waitKey(0)

    # morphological operations: opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # area open to remove small connected components
    binary = cv2.bitwise_not(binary)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            binary[labels == i] = 0
    binary = cv2.bitwise_not(binary)

    if debug > 1:
        cv2.imshow("Step 4: Binarized Image", binary)
        cv2.waitKey(0)

    # Step 5: Detect and remove the euro symbol (€)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by x-coordinate to process leftmost components first
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    image_height, image_width = binary.shape

    # Filter out noise by removing very small contours
    filtered_contours = [
        contour for contour in contours
        if cv2.boundingRect(contour)[2] > MIN_W_RELEVANT and cv2.boundingRect(contour)[3] > MIN_H_RELEVANT  # Width and height thresholds
    ]

    if debug > 1:
        debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(f"Filtered Contour {i + 1}", debug_image)
            cv2.waitKey(0)

    for idx, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Skip the large rectangle
        if w >= image_width * 0.9 and h >= image_height * 0.9:
            if debug > 0:
                print(f"Skipping large rectangle: Contour {idx + 1} with w={w}, h={h}")
            continue  # Skip the large rectangle

        # Process the first relevant contour (assume it's the euro symbol)
        if w > MIN_W_RELEVANT and h > MIN_H_RELEVANT:  # Ensure the contour has meaningful size
            cv2.rectangle(binary, (x, y), (x + w, y + h), 255, -1)  # Fill with white
            if debug > 0:
                print(f"Euro symbol removed: Contour {idx + 1} with w={w}, h={h}")
            break

    if debug > 1:
        cv2.imshow("Step 5: Euro Symbol Removed", binary)
        cv2.waitKey(0)

    # step 7: fill the black regions a bit more to better define digits
    # Adaptive kernel based on image size
    kernel_size = max(1, min(binary.shape[0] // 100, binary.shape[1] // 100))  # 1% of image size
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_closing, iterations=1)

    # step 8: exapnd the image a bit (10 pixels in each side) with white pixels
    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    # step 9: upscale by a scale factor of 2 if the image is too small
    scale_factor = 2
    upscaled = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    result = cv2.blur(upscaled, (5, 5))


    if debug > 1:
        cv2.imshow("Step 7: final Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if debug > 0:
        print("Preprocessing complete.")
    return result


def robust_ocr_extraction(image, detection_class: DetectionClass, debug=0):
    # Prepare different preprocessing methods along with their allowlists
    preprocessors = [
        {"func": lambda img, debug=0: img, "allowlist": None, "name": "Original Image"},
        {"func": image_preprocess, "allowlist": None, "name": "Preprocess 1"},
        {"func": image_preprocess_2, "allowlist": None, "name": "Preprocess 2"},
        {"func": image_preprocess_3, "allowlist": None, "name": "grayscale"},  # Default allowlist
        {"func": image_preprocess_4, "allowlist": '0123456789', "name": "rmEuroAndDots"},
    ]

    proposed_texts = []
    for preprocessor_entry in preprocessors:
        preprocessor = preprocessor_entry["func"]
        allowlist = preprocessor_entry.get("allowlist", '0123456789€.')
        processed_img = preprocessor(image, debug=debug)
        text = reader.readtext(processed_img, text_threshold=0.3, detail=0, allowlist=allowlist)
        recognized_text = ' '.join(text)
        proposed_texts.append(recognized_text)
        if preprocessor_entry.get("name") == "rmEuroAndDots":
            # adding 2 times the same text to increase the chances of being chosen (it's the best preprocessing)
            proposed_texts.append(recognized_text)

    chosen_text = choose_text_based_on_proposed_texts_and_detection_class(proposed_texts, detection_class)

    if debug > 0:
        print("proposed texts: ", proposed_texts)
        print("chosen text: ", chosen_text)

    return chosen_text


def choose_text_based_on_proposed_texts_and_detection_class(proposed_texts, detection_class: DetectionClass) -> str | None:
    # Normalize texts for consistent comparison
    normalized_texts = []
    for text in proposed_texts:
        try:
            normalized_texts.append(str(amount_string_to_int(text)))
        except ValueError:
            continue

    valid_texts = []
    invalid_texts = []
    for text in normalized_texts:
        if ocr_validator(text, detection_class):
            valid_texts.append(text)
        else:
            invalid_texts.append(text)

    # Count occurrences of valid texts
    proposals = Counter(valid_texts)
    max_votes_proposals = [proposal for proposal in proposals if proposals[proposal] == max(proposals.values())]

    chosen_text = None
    if len(max_votes_proposals) == 1:
        chosen_text = max_votes_proposals[0]
    elif len(max_votes_proposals) > 1:
        chosen_text = min(max_votes_proposals,
                          key=lambda x: sum(custom_similarity(x, str(other)) for other in normalized_texts))

    return str(chosen_text)

def get_counting_dict_from_proposals(proposals):
    counting_dict = {}
    for proposal in proposals:
        if proposal in counting_dict:
            counting_dict[proposal] += 1
        else:
            counting_dict[proposal] = 1

    return counting_dict