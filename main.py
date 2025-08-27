import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mss import mss
import json
import mouse
import pytesseract

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
    """
    Provides methods to detect and merge text lines in images using computer vision techniques.
    """

    def merge_boxes_on_same_line(
        self,
        boxes: list[tuple[int, int, int, int]],
        y_threshold: int = 10,
        x_overlap_threshold: float = 0.3
    ) -> list[tuple[int, int, int, int]]:
        """
        Merge bounding boxes that are on the same horizontal line.

        Args:
            boxes (list): List of bounding boxes in (x, y, w, h) format.
            y_threshold (int): Maximum vertical distance to consider boxes on the same line.
            x_overlap_threshold (float): Minimum horizontal overlap ratio to consider boxes connected.

        Returns:
            list: List of merged bounding boxes in (x, y, w, h) format.
        """
        if not boxes:
            return []

        # Sort boxes by y-coordinate (top to bottom)
        boxes.sort(key=lambda box: box[1])

        merged_boxes: list[tuple[int, int, int, int]] = []
        current_line: list[tuple[int, int, int, int]] = [boxes[0]]

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

    def merge_line_boxes(
        self,
        line_boxes: list[tuple[int, int, int, int]]
    ) -> tuple[int, int, int, int] | None:
        """
        Merge all bounding boxes in a line into a single bounding box.

        Args:
            line_boxes (list): List of bounding boxes in (x, y, w, h) format.

        Returns:
            tuple or None: Merged bounding box in (x, y, w, h) format, or None if input is empty.
        """
        if not line_boxes:
            return None

        # Find the extreme coordinates
        min_x = min(box[0] for box in line_boxes)
        min_y = min(box[1] for box in line_boxes)
        max_x = max(box[0] + box[2] for box in line_boxes)
        max_y = max(box[1] + box[3] for box in line_boxes)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def detect_price(
        self,
        image: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Detect text lines in an image (for price window) using computer vision techniques.

        Args:
            image (np.ndarray): Input image in OpenCV format.
            display_process (bool): Whether to display intermediate processing steps.

        Returns:
            list: List of bounding rectangles for detected text lines in (x, y, w, h) format.
        """

        # Convert to grayscale
        gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred: np.ndarray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply adaptive threshold to get binary image
        binary: np.ndarray = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 3, 4
        )

        # Define a rectangular kernel that is wider than it is tall
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

        # Apply dilation to connect text components
        dilated: np.ndarray = cv2.dilate(binary, kernel, iterations=3)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours based on area and aspect ratio to find text lines
        text_lines: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            text_lines.append((x, y, w, h))

        # Sort text lines by their y-coordinate (top to bottom)
        text_lines.sort(key=lambda rect: rect[1])

        # Merge boxes on the same line
        text_lines = self.merge_boxes_on_same_line(
            text_lines, y_threshold=8, x_overlap_threshold=0.3
        )
        for idx, (x, y, w, h) in enumerate(text_lines):
            cropped = image[y:y+h, x:x+w]
            cv2.imwrite(f"text_line_{idx+1}.png", cropped)
        return text_lines

    def detect_items(
        self,
        image: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Detect text lines in an image (for items window) using computer vision techniques.

        Args:
            image (np.ndarray): Input image in OpenCV format.
            display_process (bool): Whether to display intermediate processing steps.

        Returns:
            list: List of bounding rectangles for detected text lines in (x, y, w, h) format.
        """

        # Convert to grayscale
        gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred: np.ndarray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply adaptive threshold to get binary image
        binary: np.ndarray = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 3, 5
        )

        # Define a rectangular kernel that is wider than it is tall
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

        # Apply dilation to connect text components
        dilated: np.ndarray = cv2.dilate(binary, kernel, iterations=4)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours based on area and aspect ratio to find text lines
        text_lines: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            text_lines.append((x, y, w, h))

        # Sort text lines by their y-coordinate (top to bottom)
        text_lines.sort(key=lambda rect: rect[1])
        # for idx, (x, y, w, h) in enumerate(text_lines):
        #     cropped = image[y:y+h, x:x+w]
        #     cv2.imwrite(f"text_line_{idx+1}.png", cropped)
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




# def preprocess_and_extract_text(image: np.ndarray) -> str:
#     """
#     Preprocess image to improve OCR accuracy and extract text
#     """
#     # Read image using OpenCV
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to get binary image
#     _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Optional: Remove noise
#     kernel = np.ones((1, 1), np.uint8)
#     processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     processed = cv2.medianBlur(processed, 3)

#     # Show images using matplotlib
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 3, 1)
#     plt.title("Grayscale")
#     plt.imshow(gray, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.title("Thresholded")
#     plt.imshow(thresh, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.title("Processed")
#     plt.imshow(processed, cmap='gray')
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()
    
#     # Convert back to PIL image for Tesseract
#     custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789:,.'
#     # Extract text
#     # text = pytesseract.image_to_string(thresh)
#     text = pytesseract.image_to_string(processed, config=custom_config)
    
#     return text.strip()





def preprocess_line_image(image_path):
    """
    Preprocess image containing a line with numbers and separators
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                #   cv2.THRESH_BINARY, 11, 1)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    
    # Apply morphological operations to clean up
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Invert back to black text on white background
    # kernel = np.ones((2, 2), np.uint8)
    # result = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    
    
    # kernel = np.ones((1, 1), np.uint8)
    # result = cv2.dilate(cleaned, kernel, iterations=1)
    
    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)


    # Find contours of white regions (hollow areas)
    # Find contours of white regions (hollow areas)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # result = thresh.copy()
    # # Fill each contour that represents a hollow area
    # for contour in contours:
    #     area = cv2.contourArea(contour)
        
    #     # Only fill areas larger than minimum (to avoid noise)
    #     if area > 1:
    #         # Create mask for this contour
    #         mask = np.zeros_like(gray)
    #         cv2.drawContours(result, [contour], -1, 255, -1)
            
    #         # Fill the hollow area with black
    #         # result[mask == 255] = 0

    
    # result = cv2.bitwise_not(result)

    # Plot all images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Thresholded")
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Processed")
    plt.imshow(cleaned, cmap='gray')
    plt.axis('off')




    plt.tight_layout()
    plt.show()
    
    return cleaned, image

def extract_formatted_line(image_path):
    """
    Extract formatted line with numbers, colons, dots, and commas
    """
    # Preprocess image
    processed_img, original_img = preprocess_line_image(image_path)
    
    # Configure Tesseract for your specific character set
    # Allow digits, colon, period, and comma
    custom_config = r'--oem 3 --psm 12 -c tessedit_char_whitelist=0123456789:., '
    
    # Extract text
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    
    
    return text, processed_img, original_img







# Example usage
if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # conf = ConfigPositions()
    # screenshot_capture = ScreenCapture(conf)
    # time.sleep(2)
    # img = screenshot_capture.items_window()
    # cv2.imwrite("screenshot8.png", img)  # Save the screenshot for reference
    # exit(0)

    line_detector = DetectTextLines()

    image = cv2.imread("screenshot7.png")
    text_lines = line_detector.detect_price(
        image)

    # for i in range(7):
    #     print(f"Processing image {i+1}")
    #     image = cv2.imread(f"screenshot{i+1}.png")
    #     # Detect text lines
    #     text_lines, result_image = line_detector.detect_price(image)

    # Usage
    # extractor = NumberExtractor(whitelist="0123456789.,:", psm_mode=13)

    # Simple extraction
    # numbers = extractor.extract("numbers_image.png")
    for i in range(16, 12, -1):
        extracted_text, processed_img, original_img = extract_formatted_line(f"text_line_{i}.png")
        print(f"Extracted text: '{extracted_text}'")
        break

    
