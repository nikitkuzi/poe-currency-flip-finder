import logging
import time
import cv2
import keyboard
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
                merged_boxes.append(self.__merge_line_boxes(current_line))
                current_line = [box]

        # Add the last line
        if current_line:
            merged_boxes.append(self.__merge_line_boxes(current_line))

        return merged_boxes

    def __merge_line_boxes(
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
        image: np.ndarray,
        merge: bool = True
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
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
        text_boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            text_boxes.append((x, y, w, h))

        # Sort text lines by their y-coordinate (top to bottom)
        text_boxes.sort(key=lambda rect: rect[1])

        # for (x, y, w, h) in text_boxes:
        #     cv2.rectangle(result_image, (x, y),
        #                   (x + w, y + h), (0, 255, 0), 1)

        # plt.subplot(2, 3, 1)
        # plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        # plt.title('Detected Text Lines')
        # plt.axis('off')
        # plt.show()

        # Merge boxes on the same line
        if merge:
            text_boxes = self.merge_boxes_on_same_line(
                text_boxes, y_threshold=10, x_overlap_threshold=0.3
            )

        # for idx, (x, y, w, h) in enumerate(text_boxes):
        #     cropped = image[y:, x:x+w]
        #     cv2.imwrite(f"img/text_number_{idx+1}.png", cropped)
        tmp = [(x, y, w, h) for (x, y, w, h) in text_boxes if h > 10]
        text_boxes = tmp

        if merge:
            text_imgs = [image[y:y+h+5, x:x+w]
                         for (x, y, w, h) in text_boxes]
        else:
            text_boxes.sort(key=lambda rect: rect[0])
            text_imgs = [image[0:, x:x+w]
                         for (x, y, w, h) in text_boxes]
        # for img in text_imgs:
        #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')
        #     plt.show()
        return text_imgs, text_boxes

    def detect_items(
        self,
        image: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        """
        Detect text lines in an image (for items window) using computer vision techniques.

        Args:
            image (np.ndarray): Input image in OpenCV format.
            display_process (bool): Whether to display intermediate processing steps.

        Returns:
            list: List of bounding rectangles for detected text lines in (x, y, w, h) format.
        """

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Convert to grayscale
        gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        image = cv2.resize(image, None, fx=2, fy=2)

        # Apply Gaussian blur to reduce noise
        blurred: np.ndarray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply adaptive threshold to get binary image
        binary: np.ndarray = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 5, 11
        )

        # Define a rectangular kernel that is wider than it is tall
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 8))

        # Apply dilation to connect text components
        dilated: np.ndarray = cv2.dilate(binary, kernel, iterations=4)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours based on area and aspect ratio to find text lines
        text_boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            text_boxes.append((x, y, w, h))

        # Sort text lines by their y-coordinate (top to bottom)
        text_boxes.sort(key=lambda rect: rect[1])
        # for idx, (x, y, w, h) in enumerate(text_boxes):
        #     cropped = image[y:y+h, x:x+w]
        #     cv2.imwrite(f"text_line_{idx+1}.png", cropped)

        tmp = [(x, y, w, h)
               for (x, y, w, h) in text_boxes if h > 10 and w*h > 3000]
        text_boxes = tmp

        text_imgs = []
        boxes = []

        for (x, y, w, h) in text_boxes:
            text_imgs.append(image[y:y+h+5, x:x+w])
            boxes.append((x//2, y//2, w//2, h//2))

        text_imgs = [image[y:y+h+5, x:x+w]
                     for (x, y, w, h) in text_boxes]

        result_image = image.copy()
        # for (x, y, w, h) in text_boxes:
        #     cv2.rectangle(result_image, (x, y),
        #                   (x + w, y + h), (0, 255, 0), 1)
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 5, 1)
        # plt.title("Original")
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.subplot(1, 5, 2)
        # plt.title("Grayscale")
        # plt.imshow(gray, cmap='gray')
        # plt.axis('off')

        # plt.subplot(1, 5, 3)
        # plt.title("Thresholded")
        # plt.imshow(binary, cmap='gray')
        # plt.axis('off')

        # plt.subplot(1, 5, 4)
        # plt.title("Processed")
        # plt.imshow(dilated, cmap='gray')
        # plt.axis('off')

        # plt.subplot(1, 5, 5)
        # plt.title("Processed")
        # plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()

        # return text_imgs, text_boxes
        return text_imgs, boxes


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

    def get_position(self, key: str) -> tuple[int, int]:
        """
        Retrieve the screen position for a given key.

        Args:
            key (str): The configuration key.

        Returns:
            tuple or None: (x, y) coordinates if found, else None.
        """
        try:
            return self.config[key]
        except KeyError:
            logger.error(f"Key '{key}' not found in configuration.")
            raise KeyError(f"Key '{key}' not found in configuration.")


class TextExtractor:
    """
    Extracts formatted text lines containing numbers, colons, dots, and commas
    from images using OCR (Optical Character Recognition) with Tesseract.
    """

    def __init__(self, tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe") -> None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def __preprocess_number_image(self, image):
        """
        Preprocess image containing a line with numbers and separators
        """
        # Read image
        if image is None:
            raise ValueError(f"Could not load image")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # Apply adaptive thresholding
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #   cv2.THRESH_BINARY, 55, 21)

        # kernel = np.ones((1, 1), np.uint8)
        # gray1 = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        # gray1 = cv2.morphologyEx(gray1, cv2.MORPH_DILATE, kernel, iterations=2)
        # plt.imshow(gray1, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # exit(0)
        # _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        # thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        # thresh = cv2.resize(thresh, None, fx=2, fy=2)
        # thresh = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        # cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        _, thresh = cv2.threshold(cleaned, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 1), np.uint8)
        cleaned = cv2.dilate(thresh, kernel, iterations=2)
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

        cleaned = cv2.bitwise_not(cleaned)
        # cleaned = cv2.bitwise_not(thresh)

        # Plot all images
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 4, 1)
        # plt.title("Original")
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.subplot(1, 4, 2)
        # plt.title("Grayscale")
        # plt.imshow(gray, cmap='gray')
        # plt.axis('off')

        # plt.subplot(1, 4, 3)
        # plt.title("Thresholded")
        # plt.imshow(thresh, cmap='gray')
        # plt.axis('off')

        # plt.subplot(1, 4, 4)
        # plt.title("Processed")
        # plt.imshow(cleaned, cmap='gray')
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()

        return cleaned

    def extract_number(self, image):
        """
        Extract formatted line with numbers, colons, dots, and commas
        """
        # Preprocess image
        processed_img = self.__preprocess_number_image(image)
        # Configure Tesseract for your specific character set
        # Allow digits, colon, period, and comma
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:., '

        # Extract text
        text = pytesseract.image_to_string(
            processed_img, config=custom_config).strip()

        return text

    def extract_text(self, image, preprocess: bool = False) -> str:
        """
        Extract all text from an image using Tesseract OCR.

        Args:
            image_path (str): Path to the input image.

        Returns:
            str: Extracted text.
        """
        # Read image
        if image is None:
            raise ValueError(f"Could not load image")

        # Configure Tesseract
        custom_config = r'--oem 3 --psm 6 -l eng'
        # Convert to grayscale and resize for better OCR accuracy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # gray = cv2.resize(gray, None, fx=2, fy=2)
        if preprocess:
            gray = self.__preprocess_number_image(image)

        # Extract text
        text = pytesseract.image_to_string(
            gray, config=custom_config).strip().replace('\n', ' ')

        return text

    # def get_price_and_stock(self, images: list[np.ndarray]):


class PriceAndStockExtractor:

    def __init__(self, line_detector: DetectTextLines, text_extractor: TextExtractor):
        self.line_detector = line_detector
        self.text_extractor = text_extractor

    def extract_price_and_stock(self, text_images: list[np.ndarray], limit: int = 2) -> list[tuple[str, str]]:
        prices_and_stocks = []
        processed_windows = 0
        look_for_ratio_stock = True
        seen_prices = 0
        seen_ratios = 0
        for img in text_images:

            if look_for_ratio_stock:
                extracted_text = self.text_extractor.extract_text(
                    img, preprocess=False)
                print(f"Extracted text: '{extracted_text}'")
                if "stock" in extracted_text.lower():
                    seen_ratios += 1
                    prices_and_stocks.append([])
                    processed_windows += 1
                    look_for_ratio_stock = False
            elif seen_ratios > 0 and not look_for_ratio_stock:
                price_and_stock_img, _ = self.line_detector.detect_price(
                    img, merge=False)
                price = self.text_extractor.extract_number(
                    price_and_stock_img[0])
                stock = self.text_extractor.extract_number(
                    price_and_stock_img[1])
                price = price.replace("::", ":")
                price = price.replace("..", ".")
                price = price.replace(":.", ":")
                price = price.replace(".:", ":")
                stock = stock.replace(".", ",")
                print(f"Extracted price: '{price}', stock: '{stock}'")

                if price == "" or stock == "":
                    processed_windows += 1
                    look_for_ratio_stock = True
                    continue

                prices_and_stocks[-1].append((price, stock))
                seen_prices += 1
                if seen_prices >= limit or seen_prices >= 6:
                    look_for_ratio_stock = True
                    seen_prices = 0
                    if processed_windows >= 2:
                        break

        return prices_and_stocks


class MovementHandler:

    def __init__(self, price_and_stock_extractor: PriceAndStockExtractor, config: ConfigPositions) -> None:
        self.config = config
        self.pse = price_and_stock_extractor

    def __show_market_window(self, delay=0.01) -> None:

        mouse.move(*self.config.get_position("Market_mid"))
        time.sleep(delay)
        keyboard.press("alt")
        time.sleep(delay)

    def __hide_market_window(self) -> None:
        keyboard.release("alt")
        time.sleep(0.01)

    def __find_item(self, item_name: str) -> None:

        keyboard.press("ctrl")
        keyboard.press("f")
        time.sleep(0.1)
        keyboard.release("f")
        keyboard.release("ctrl")

        keyboard.write(item_name)

        # find first position of first matching item

    def __change_currency(self, currency_name: str) -> None:
        # load from conf position of have box
        pos = self.config.get_position("I_have_name")
        # click on have box
        mouse.move(pos[0], pos[1])
        mouse.click(button='left')
        time.sleep(0.01)
        self.__find_item(currency_name)

    def change_to_chaos(self) -> None:
        self.__change_currency("Chaos Orb")

    def change_to_divine(self) -> None:
        self.__change_currency("Divine Orb")

    # todo
    def change_wanted_item(self, item_name: str) -> None:
        # load from conf position of want box
        pos = (0, 0)
        # click on want box
        mouse.move(pos[0], pos[1])
        mouse.click(button='left')
        time.sleep(0.01)
        self.__find_item(item_name)


def extract(img_path: str = "img/screenshot8.png"):

    # conf = ConfigPositions()
    # screenshot_capture = ScreenCapture(conf)
    # time.sleep(2)
    # image = screenshot_capture.price_window()
    # cv2.imwrite("screenshot9.png", image)  # Save the screenshot for reference
    # exit(0)

    line_detector = DetectTextLines()
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    image = cv2.imread(img_path)

    text_imgs, _ = line_detector.detect_price(image)

    text_extractor = TextExtractor()

    # price_and_stock_extractor = PriceAndStockExtractor(line_detector, text_extractor)

    # prices_and_stocks = price_and_stock_extractor.extract_price_and_stock(text_imgs)
    # for price, stock in prices_and_stocks[0]:
    #     print(f"Price: '{price}', Stock: '{stock}'")
    # for price, stock in prices_and_stocks[1]:
    #     print(f"Price: '{price}', Stock: '{stock}'")
    # exit(0)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # items, boxes = line_detector.detect_items(image)

    # # exit(0)
    # for img, box in zip(items, boxes):
    #     text = text_extractor.extract_text(img)
    #     print(f"Extracted item text: '{text}' at box {box}")
    #     # break

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # usage
    # input
    # items = ["Scarab of Risk", "Fine delirium Orb"]
    # return
    # imtes_with_prices = {item_name: {currency: (price, stock)}}

    # TODO:
    # steps
    # find_wanted_item
    # change_currency
    # show_market_window
    # extract price and stock
    # hide_market_window
    # change_currency
    # show_market_window
    # extract price and stock
    # hide_market_window

    # return items_with_prices


# Example usage
if __name__ == "__main__":
    logger = logging
    logger.basicConfig(filename='log.log', level=logging.DEBUG,
                       format='%(levelname)s: %(name)s - %(asctime)s - %(message)s', datefmt='%d/%b/%y %H:%M:%S')
    extract()
