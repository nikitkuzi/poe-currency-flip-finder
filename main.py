import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mss import mss
import json
import mouse


class ScreenCapture:
    """Handles screen capturing of specified regions based on configuration.
    """

    def __init__(self, config):
        # load positions from config
        resolution = config.get_position("Resolution")
        market_right = config.get_position("Market_right")
        market_left = config.get_position("Market_left")
        items_top_left = config.get_position("Items_top_left")
        items_bottom_right = config.get_position("Items_bottom_right")
        # Define the region to capture (top, left, width, height)
        self.price = {"top": market_left[1], "left": market_left[0],
                      "width": market_right[0]-market_left[0], "height": items_bottom_right[1]-market_right[1]}

        self.items = {"top": items_top_left[1], "left": items_top_left[0],
                      "width": items_bottom_right[0]-items_top_left[0], "height": items_bottom_right[1]-items_top_left[1]}

    def take_screenshot(self, target: dict) -> np.ndarray:
        """Takse a screenshot of the specified region.

        Args:
            target (dict): Region to capture with keys 'top', 'left', 'width', 'height'.

        Returns:
            np.ndarray: Captured image in OpenCV format.
        """
        with mss() as sct:
            # Capture screen
            screenshot = sct.grab(target)

            # Convert to OpenCV format
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    def items_window(self) -> np.ndarray:
        """Calls take_screenshot for items window.

        Returns:
            np.ndarray: Captured image in OpenCV format.
        """
        return self.take_screenshot(self.items)

    def price_window(self) -> np.ndarray:
        """Call


        Returns:
            np.ndarray: Captured image in OpenCV format.
        """
        return self.take_screenshot(self.price)


class DetectTextLines:

    def merge_boxes_on_same_line(self, boxes, y_threshold=10, x_overlap_threshold=0.3):
        """
        Merge bounding boxes that are on the same line.

        Args:
            boxes: List of bounding boxes in (x, y, w, h) format
            y_threshold: Maximum vertical distance to consider boxes on same line
            x_overlap_threshold: Minimum overlap to consider boxes connected

        Returns:
            List of merged bounding boxes
        """
        if not boxes:
            return []

        # Sort boxes by y-coordinate (top to bottom)
        boxes.sort(key=lambda box: box[1])

        merged_boxes = []
        current_line = [boxes[0]]

        for box in boxes[1:]:
            x, y, w, h = box
            current_y = current_line[0][1]
            current_height = current_line[0][3]

            # Check if box is on the same line (similar y-coordinate)
            if abs(y - current_y) < y_threshold or abs(y + h - current_y - current_height) < y_threshold:
                current_line.append(box)
            else:
                # Merge boxes in current line
                merged_boxes.append(self.merge_line_boxes(current_line))
                current_line = [box]

        # Add the last line
        if current_line:
            merged_boxes.append(self.merge_line_boxes(current_line))

        return merged_boxes

    def merge_line_boxes(self, line_boxes):
        """
        Merge all boxes in a line into a single bounding box.
        """
        if not line_boxes:
            return None

        # Find the extreme coordinates
        min_x = min(box[0] for box in line_boxes)
        min_y = min(box[1] for box in line_boxes)
        max_x = max(box[0] + box[2] for box in line_boxes)
        max_y = max(box[1] + box[3] for box in line_boxes)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def detect_price(self, image, display_process=False):
        """
        Detect text lines in an image using computer vision techniques.

        Args:
            image_path (str): Path to the input image
            display_process (bool): Whether to display intermediate processing steps

        Returns:
            list: List of bounding rectangles for detected text lines
        """

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply adaptive threshold to get binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 3, 4)

        # if display_process:
        #     plt.figure(figsize=(15, 10))
        #     plt.subplot(2, 3, 1), plt.imshow(
        #         cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     plt.title('Original Image'), plt.axis('off')
        #     plt.subplot(2, 3, 2), plt.imshow(gray, cmap='gray')
        #     plt.title('Grayscale'), plt.axis('off')
        #     plt.subplot(2, 3, 3), plt.imshow(binary, cmap='gray')
        #     plt.title('Binary Image'), plt.axis('off')

        # Define a rectangular kernel that is wider than it is tall
        # This helps in connecting characters into text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

        # Apply dilation to connect text components
        dilated = cv2.dilate(binary, kernel, iterations=3)

        # if display_process:
        #     plt.subplot(2, 3, 4), plt.imshow(dilated, cmap='gray')
        #     plt.title('After Dilation'), plt.axis('off')

        # Find contours in the dilated image
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and aspect ratio to find text lines
        text_lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # d = calculate_white_pixel_density(binary[y:y+h, x:x+w])

            # Filter based on aspect ratio and area
            # aspect_ratio = w / h
            # area = w * h

            # Adjust these thresholds based on your specific needs
            # if aspect_ratio > 20 and area > 100:# and aspect_ratio < 40 and h > 5:
            # if 100 < area:  # < 5000:# and 1< aspect_ratio < 5:# and d > 0.01:
            text_lines.append((x, y, w, h))

        # Sort text lines by their y-coordinate (top to bottom)
        text_lines.sort(key=lambda rect: rect[1])

        # result_image = image.copy()
        # for (x, y, w, h) in text_lines:
        #     cv2.rectangle(result_image, (x, y),
        #                       (x + w, y + h), (0, 255, 0), 1)

        # if display_process:
        #     plt.subplot(2, 3, 5), plt.imshow(
        #         cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        #     plt.title('Detected Text Lines'), plt.axis('off')

        text_lines = self.merge_boxes_on_same_line(
            text_lines, y_threshold=8, x_overlap_threshold=0.3)

        # Draw bounding boxes on the original image
        # result_image = image.copy()
        # for (x, y, w, h) in text_lines:
        #     if w/h < 15:
        #         cv2.rectangle(result_image, (x, y),
        #                       (x + w, y + h), (0, 255, 0), 1)

        # if display_process:
        #     plt.subplot(2, 3, 6), plt.imshow(
        #         cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        #     plt.title('Detected Text Lines Merged'), plt.axis('off')
        #     plt.tight_layout()
        #     plt.show()

        return text_lines

    def detect_items(self, image, display_process=False):
        """
        Detect text lines in an image using computer vision techniques.

        Args:
            image_path (str): Path to the input image
            display_process (bool): Whether to display intermediate processing steps

        Returns:
            list: List of bounding rectangles for detected text lines
        """

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply adaptive threshold to get binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 3, 5)

        # if display_process:
        #     plt.figure(figsize=(15, 10))
        #     plt.subplot(2, 3, 1), plt.imshow(
        #         cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     plt.title('Original Image'), plt.axis('off')
        #     plt.subplot(2, 3, 2), plt.imshow(gray, cmap='gray')
        #     plt.title('Grayscale'), plt.axis('off')
        #     plt.subplot(2, 3, 3), plt.imshow(binary, cmap='gray')
        #     plt.title('Binary Image'), plt.axis('off')

        # Define a rectangular kernel that is wider than it is tall
        # This helps in connecting characters into text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

        # Apply dilation to connect text components
        dilated = cv2.dilate(binary, kernel, iterations=4)

        # if display_process:
        #     plt.subplot(2, 3, 4), plt.imshow(dilated, cmap='gray')
        #     plt.title('After Dilation'), plt.axis('off')

        # Find contours in the dilated image
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and aspect ratio to find text lines
        text_lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # d = calculate_white_pixel_density(binary[y:y+h, x:x+w])

            # Filter based on aspect ratio and area
            # aspect_ratio = w / h
            # area = w * h

            # Adjust these thresholds based on your specific needs
            # if aspect_ratio > 20 and area > 100:# and aspect_ratio < 40 and h > 5:
            # if 100 < area:  # < 5000:# and 1< aspect_ratio < 5:# and d > 0.01:
            text_lines.append((x, y, w, h))

        # Sort text lines by their y-coordinate (top to bottom)
        text_lines.sort(key=lambda rect: rect[1])

        # result_image = image.copy()
        # for (x, y, w, h) in text_lines:
        #     cv2.rectangle(result_image, (x, y),
        #                       (x + w, y + h), (0, 255, 0), 1)

        # if display_process:
        #     plt.subplot(2, 3, 5), plt.imshow(
        #         cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        #     plt.title('Detected Text Lines'), plt.axis('off')
        #     plt.tight_layout()
        #     plt.show()

        return text_lines


class ConfigPositions:
    """
    Handles loading, saving, and updating screen position configurations
    from a JSON file for use in screen capture and automation tasks.
    """

    def __init__(self, config_file: str = "currency-exchange-config.json") -> None:
        """
        Initialize the ConfigPositions object.

        Args:
            config_file (str): Path to the configuration JSON file.
        """
        self.config_file: str = config_file
        self.config: dict[str, tuple[int, int]] = {}
        self.load_config()

    def load_config(self) -> None:
        """
        Load configuration from the JSON file.
        If the file does not exist, initializes with an empty config.
        """
        try:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}

    def __save_config(self) -> None:
        """
        Save the current configuration to the JSON file.
        """
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def set_position(self) -> None:
        """
        Interactively set screen positions for each key in the config.
        Prompts the user to right-click at the desired position for each key,
        then saves the coordinates.
        """
        for key, _ in self.config.items():
            print(f"right click at the position of {key}")
            mouse.wait(button='right', target_types=('up'))
            x, y = mouse.get_position()
            self.config[key] = (x, y)
        self.__save_config()

    def get_position(self, key: str) -> tuple[int, int] | None:
        """
        Retrieve the screen position for a given key.

        Args:
            key (str): The configuration key.

        Returns:
            tuple or None: (x, y) coordinates if found, else None.
        """
        return self.config.get(key, None)


# Example usage
if __name__ == "__main__":

    # conf = ConfigPositions()
    # screenshot_capture = ScreenCapture(conf)
    # time.sleep(2)
    # img = screenshot_capture.items_window()
    # cv2.imwrite("screenshot8.png", img)  # Save the screenshot for reference
    # exit(0)

    line_detector = DetectTextLines()

    image = cv2.imread("screenshot8.png")
    text_lines= line_detector.detect_items(
        image, display_process=True)

    # for i in range(7):
    #     print(f"Processing image {i+1}")
    #     image = cv2.imread(f"screenshot{i+1}.png")
    #     # Detect text lines
    #     text_lines, result_image = line_detector.detect_price(image, display_process=True)
