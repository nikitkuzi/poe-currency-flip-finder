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


class ScreenCapture:
    """
    Handles screen capturing of specified regions based on configuration.
    """

    def __init__(self, config: ConfigPositions) -> None:
        """
        Initialize ScreenCapture with positions from config.

        Args:
            config (ConfigPositions): Configuration object with screen positions.
        """
        resolution = config.get_position("Resolution")
        market_right = config.get_position("Market_right")
        market_left = config.get_position("Market_left")
        market_bottom_right = config.get_position("Market_bottom_right")
        items_top_left = config.get_position("Items_top_left")
        items_bottom_right = config.get_position("Items_bottom_right")
        # Define the region to capture (top, left, width, height)
        self.price = {"top": market_left[1], "left": market_left[0],
                      "width": market_right[0]-market_left[0], "height": market_bottom_right[1]-market_right[1]}

        self.items = {"top": items_top_left[1], "left": items_top_left[0],
                      "width": items_bottom_right[0]-items_top_left[0], "height": items_bottom_right[1]-items_top_left[1]}

    def __take_screenshot(self, target: dict) -> np.ndarray:
        """
        Take a screenshot of the specified region.

        Args:
            target (dict): Region to capture with keys 'top', 'left', 'width', 'height'.

        Returns:
            np.ndarray: Captured image in OpenCV format.
        """
        with mss() as sct:
            screenshot = sct.grab(target)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def items_window(self) -> np.ndarray:
        """
        Capture the items window region.

        Returns:
            np.ndarray: Captured image in OpenCV format.
        """
        return self.__take_screenshot(self.items)

    def price_window(self) -> np.ndarray:
        """
        Capture the price window region.

        Returns:
            np.ndarray: Captured image in OpenCV format.
        """
        return self.__take_screenshot(self.price)


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
            merge (bool): Whether to merge boxes on the same line.

        Returns:
            tuple: List of cropped images for detected text lines,
                   List of bounding rectangles for detected text lines in (x, y, w, h) format.
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
        # kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))

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

        # Merge boxes on the same line
        if merge:
            text_boxes = self.merge_boxes_on_same_line(
                text_boxes, y_threshold=10, x_overlap_threshold=0.3
            )

        # Filter out boxes that are too small
        text_boxes = [(x, y, w, h) for (x, y, w, h) in text_boxes if h > 10]

        if merge:
            text_imgs = [image[y:y+h+5, x:x+w]
                         for (x, y, w, h) in text_boxes]
        else:
            text_boxes.sort(key=lambda rect: rect[0])
            text_imgs = [image[0:, x:x+w]
                         for (x, y, w, h) in text_boxes]
        return text_imgs, text_boxes

    def detect_items(
        self,
        image: np.ndarray
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        """
        Detect text lines in an image (for items window) using computer vision techniques.

        Args:
            image (np.ndarray): Input image in OpenCV format.

        Returns:
            tuple: List of cropped images for detected text lines,
                   List of bounding rectangles for detected text lines in (x, y, w, h) format.
        """

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

        # Filter out boxes that are too small or have too little area
        text_boxes = [(x, y, w, h)
                      for (x, y, w, h) in text_boxes if h > 10 and w*h > 3000]

        text_imgs = []
        boxes = []

        for (x, y, w, h) in text_boxes:
            text_imgs.append(image[y:y+h+5, x:x+w])
            boxes.append((x//2, y//2, w//2, h//2))

        text_imgs = [image[y:y+h+5, x:x+w]
                     for (x, y, w, h) in text_boxes]

        return text_imgs, boxes


class TextExtractor:
    """
    Extracts formatted text lines containing numbers, colons, dots, and commas
    from images using OCR (Optical Character Recognition) with Tesseract.
    """

    def __init__(self, tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe") -> None:
        """
        Initialize TextExtractor and set the Tesseract executable path.

        Args:
            tesseract_path (str): Path to the Tesseract executable.
        """
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def __preprocess_number_image(self, image):
        """
        Preprocess image containing a line with numbers and separators.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image suitable for OCR.
        """
        if image is None:
            raise ValueError(f"Could not load image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        _, thresh = cv2.threshold(cleaned, 120, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 1), np.uint8)
        cleaned = cv2.dilate(thresh, kernel, iterations=2)
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.bitwise_not(cleaned)
        return cleaned

    def extract_number(self, image):
        """
        Extract formatted line with numbers, colons, dots, and commas.

        Args:
            image (np.ndarray): Input image.

        Returns:
            str: Extracted text.
        """
        processed_img = self.__preprocess_number_image(image)
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:., '
        text = pytesseract.image_to_string(
            processed_img, config=custom_config).strip()
        return text

    def extract_text(self, image, preprocess: bool = False) -> str:
        """
        Extract all text from an image using Tesseract OCR.

        Args:
            image (np.ndarray): Input image.
            preprocess (bool): Whether to preprocess the image for numbers.

        Returns:
            str: Extracted text.
        """
        if image is None:
            raise ValueError(f"Could not load image")

        custom_config = r'--oem 3 --psm 6 -l eng'
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if preprocess:
            gray = self.__preprocess_number_image(image)
        text = pytesseract.image_to_string(
            gray, config=custom_config).strip().replace('\n', ' ')
        return text


class PriceAndStockExtractor:
    """
    Extracts price and stock information from a list of images containing text lines.
    """

    def __init__(self, line_detector: DetectTextLines, text_extractor: TextExtractor):
        """
        Initialize PriceAndStockExtractor.

        Args:
            line_detector (DetectTextLines): Object for detecting text lines.
            text_extractor (TextExtractor): Object for extracting text from images.
        """
        self.line_detector = line_detector
        self.text_extractor = text_extractor

    def extract_price_and_stock(self, text_images: list[np.ndarray], limit: int = 2) -> list[list[tuple[str, str]]]:
        """
        Extract price and stock pairs from a list of images.

        Args:
            text_images (list): List of images containing text lines.
            limit (int): Maximum number of price/stock pairs to extract per window.

        Returns:
            list: List of tuples (price, stock).
        """
        prices_and_stocks = []
        processed_windows = 0
        look_for_ratio_stock = True
        seen_prices = 0
        seen_ratios = 0
        for img in text_images:
            if look_for_ratio_stock:
                extracted_text = self.text_extractor.extract_text(
                    img, preprocess=False)
                # print(f"Extracted text: '{extracted_text}'")
                if "stock" in extracted_text.lower():
                    seen_ratios += 1
                    prices_and_stocks.append([])
                    processed_windows += 1
                    look_for_ratio_stock = False
            elif seen_ratios > 0 and not look_for_ratio_stock:
                price_and_stock_img, _ = self.line_detector.detect_price(
                    img, merge=False)
                # plt.imshow(cv2.cvtColor(price_and_stock_img[0], cv2.COLOR_BGR2RGB))
                # plt.axis('off')
                # plt.show()
                price = self.text_extractor.extract_number(
                    price_and_stock_img[0])
                stock = self.text_extractor.extract_number(
                    price_and_stock_img[1])
                price = price.replace("::", ":")
                price = price.replace("..", ".")
                price = price.replace(":.", ":")
                price = price.replace(".:", ":")
                stock = stock.replace(".", ",")
                # print(f"Extracted price: '{price}', stock: '{stock}'")

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
    """
    Handles mouse and keyboard automation for interacting with the currency exchange UI.
    """

    def __init__(self, config: ConfigPositions) -> None:
        """
        Initialize MovementHandler.

        Args:
            config (ConfigPositions): Configuration object with screen positions.
        """
        self.config = config
        self.detector = DetectTextLines()
        self.text_extractor = TextExtractor()
        self.screen_capture = ScreenCapture(config)
        self.pse = PriceAndStockExtractor(self.detector, self.text_extractor)

    def __show_market_window(self, delay=0.01) -> None:
        """
        Show the market window by moving the mouse and pressing 'alt'.
        """
        mouse.move(10, 10)
        time.sleep(0.3)
        mouse.move(*self.config.get_position("Market_mid"))
        time.sleep(delay)
        keyboard.press("alt")
        time.sleep(delay)

    def __hide_market_window(self, delay=0.01) -> None:
        """
        Hide the market window by releasing 'alt'.
        """
        keyboard.release("alt")
        time.sleep(delay)

    def __find_item(self, item_name: str) -> None:
        """
        Finds item in the items window and clicks on it.

        Args:
            item_name (str): Name of the item to find.
        """
        keyboard.press("ctrl")
        keyboard.press("f")
        time.sleep(0.01)
        keyboard.release("f")
        keyboard.release("ctrl")

        keyboard.write(item_name)
        time.sleep(0.3)

        item_window = self.screen_capture.items_window()
        time.sleep(0.01)
        item_imgs, boxes = self.detector.detect_items(item_window)
        time.sleep(0.01)
        for img, box in zip(item_imgs, boxes):
            text = self.text_extractor.extract_text(img)
            # print(f"Extracted item text: '{text}' at box {box}")
            if item_name.lower() == text.lower():
                pos = self.config.get_position("Items_top_left")
                x = pos[0] + box[0] + box[2] // 2
                y = pos[1] + box[1] + box[3] // 2
                mouse.move(x, y)
                time.sleep(0.01)
                mouse.click(button='left')
                time.sleep(0.1)
                break

    def change_item(self, item_name: str, wanted=True) -> None:
        """
        Change the selected item in the UI.

        Args:
            item_name (str): Name of the item to select.
            wanted (bool): If True, select "I want" box, else "I have" box.
        """
        if wanted:
            pos = self.config.get_position("I_want_name")
        else:
            pos = self.config.get_position("I_have_name")
        mouse.move(pos[0], pos[1])
        time.sleep(0.01)
        mouse.click(button='left')
        time.sleep(0.2)
        self.__find_item(item_name)

    def __change_to_chaos(self) -> None:
        """
        Change selection to "Chaos Orb" in the UI.
        """
        self.change_item("Chaos Orb", wanted=False)

    def __change_to_divine(self) -> None:
        """
        Change selection to "Divine Orb" in the UI.
        """
        self.change_item("Divine Orb", wanted=False)

    def show(self) -> None:
        """
        Show the market window.
        """
        self.__show_market_window()
        time.sleep(0.5)
        self.__hide_market_window()

    def __extract(self, item_name: str) -> list[list[tuple[str, str]]]:
        """
        Extract price and stock for a given item.

        Args:
            item_name (str): Name of the item.

        Returns:
            list: List of tuples (price, stock).
        """
        if item_name == "Chaos Orb":
            self.__change_to_chaos()
        else:
            self.__change_to_divine()
        time.sleep(0.01)
        self.__show_market_window()
        time.sleep(0.01)
        market_window = self.screen_capture.price_window()
        price_imgs, _ = self.detector.detect_price(market_window)
        prices_and_stock = self.pse.extract_price_and_stock(price_imgs)
        self.__hide_market_window()
        return prices_and_stock

    def extract_price(self) -> dict[str, list[list[tuple[str, str]]]]:
        """
        Extract prices and stock for Chaos Orb and Divine Orb.

        Returns:
            dict: Mapping from item name to list of (price, stock) tuples.
        """
        prices_with_stock = {}
        prices_with_stock["Chaos Orb"] = self.__extract("Chaos Orb")
        prices_with_stock["Divine Orb"] = self.__extract("Divine Orb")
        return prices_with_stock

    def extract_chaos_to_divine_ratio(self) -> list[list[tuple[str, str]]]:
        """
        Extract the ratio of Chaos Orb to Divine Orb.

        Returns:
            list: List of tuples (price, stock) for Chaos Orb when Divine Orb is selected.
        """
        self.change_item("Chaos Orb", wanted=True)
        time.sleep(0.1)
        return self.__extract("Divine Orb")


class ProfitCalculator:

    """
    Calculates the most profitable item to flip or provide liquidity.
    Supports 4 strategies:
      1. Buy with Divine → sell for Divine (liquidity)
      2. Buy with Chaos → sell for Chaos (liquidity)
      3. Buy with Divine → sell for Chaos (flip)
      4. Buy with Chaos → sell for Divine (flip)
    Uses top prices, ignores stock, and applies 1% undercuts
    """

    def __init__(self, items_with_prices: dict[str, dict[str, list[list[tuple[str, str]]]]], chaos_to_divine: list[list[tuple[str, str]]]) -> None:
        self.items_with_prices = items_with_prices
        self.chaos_to_divine_rate = self.__get_chaos_to_divine_rate(
            chaos_to_divine)

    def _parse_ratio(self, ratio_str: str) -> float:
        """
        Parses a ratio string and returns its float value.

        The ratio string can be in the format "left:right" or a single numeric value.
        Commas in the string are removed before parsing.
        If the format is "left:right", both sides are converted to floats and the ratio (left/right) is returned.

        Args:
            ratio_str (str): The ratio string to parse.

        Returns:
            float: The parsed ratio as a float, or 0.0 on failure.
        """
        ratio_str = ratio_str.replace(",", "")
        if ":" in ratio_str:
            left, right = ratio_str.split(":")
            try:
                left = float(left)
                right = float(right)
                return left / right if right != 0 else 0.0
            except Exception:
                return 0.0
        try:
            return float(ratio_str)
        except Exception:
            return 0.0

    def _average_price(self, orders: list[tuple[str, str]], undercut: int = 1, limit: int = 2) -> float:
        """
        Calculates the average price from a list of order tuples, applying an undercut percentage.

        Args:
            orders (list[tuple[str, str]]): A list of tuples where each tuple contains price information as strings.
            undercut (int, optional): The percentage to change the average price by. Defaults to 1.
            limit (int, optional): The maximum number of orders to consider for averaging. Defaults to 2.

        Returns:
            float: The undercut average price calculated from the provided orders. Returns 0.0 if no orders are given.
        """
        if not orders:
            return 0.0
        prices = [self._parse_ratio(price) for price, _ in orders[:limit]]
        return (sum(prices) / len(prices)) * (1 - undercut * 0.01)

    def __get_chaos_to_divine_rate(self, chaos_to_divine: list[list[tuple[str, str]]]) -> float:
        """
        Returns the current chaos to divine exchange rate.
        """
        rate_str = chaos_to_divine[0][0][0]
        self.chaos_to_divine_rate = self._parse_ratio(rate_str)
        return self.chaos_to_divine_rate

    def calculate_best_profit(self) -> tuple[dict[str, float | str | None], dict[str, list[dict[str, float | None]]]]:
        """
        Calculates the most profitable item to flip or provide liquidity.
        Supports 4 strategies:
          1. Buy with Divine → sell for Divine (liquidity)
          2. Buy with Chaos → sell for Chaos (liquidity)
          3. Buy with Divine → sell for Chaos (flip)
          4. Buy with Chaos → sell for Divine (flip)
        Uses average prices, ignores stock, and applies undercuts:
          - Divine: -0.05
          - Chaos: -0.5
        Returns:
            tuple: (best_result, results)
              best_result: dict with keys 'item', 'profit', 'method'
              results: dict mapping item to list of profit/method dicts
        """
        best_result = {"item": None, "profit": 0.0, "method": None}
        results = {item: [] for item in self.items_with_prices.keys()}
        for item, prices in self.items_with_prices.items():
            chaos_orders = prices.get("Chaos Orb", [])
            divine_orders = prices.get("Divine Orb", [])

            # Average top two orders with undercuts
            # avg_chaos_buy = self._average_price(chaos_orders[1])
            # avg_chaos_sell = self._average_price(chaos_orders[0])
            # avg_divine_buy = self._average_price(divine_orders[1])
            # avg_divine_sell = self._average_price(divine_orders[0])
            avg_chaos_buy = self._parse_ratio(chaos_orders[1][0][0]) * 1.01
            avg_chaos_sell = self._parse_ratio(chaos_orders[0][0][0]) * 0.99
            avg_divine_buy = self._parse_ratio(divine_orders[1][0][0]) * 1.01
            avg_divine_sell = self._parse_ratio(divine_orders[0][0][0]) * 0.99
            print(f"Item: {item}, avg_chaos_buy: {avg_chaos_buy}, avg_chaos_sell: {avg_chaos_sell}, avg_divine_buy: {avg_divine_buy}, avg_divine_sell: {avg_divine_sell}")

            # 1. Provide liquidity: Buy & Sell Divine (Divine Liquidity)
            profit = (avg_divine_buy - avg_divine_sell) * 100 // avg_divine_buy
            results[item].append(
                {"profit": profit, "method": "Divine Liquidity"})
            if profit > best_result["profit"]:
                best_result = {"item": item, "profit": profit,
                               "method": "Divine Liquidity"}

            # 2. Provide liquidity: Buy & Sell Chaos (Chaos Liquidity)
            profit = (avg_chaos_buy - avg_chaos_sell) * 100 // avg_chaos_buy
            results[item].append(
                {"profit": profit, "method": "Chaos Liquidity"})
            if profit > best_result["profit"]:
                best_result = {"item": item, "profit": profit,
                               "method": "Chaos Liquidity"}

            # 3. Flip: Buy Divine → Sell Chaos
            sell_for_chaos = avg_divine_buy / avg_chaos_sell
            profit = (sell_for_chaos / self.chaos_to_divine_rate - 1) * 100 // 1
            results[item].append(
                {"profit": profit, "method": "Divine→Chaos Flip"})
            if profit > best_result["profit"]:
                best_result = {"item": item, "profit": profit,
                               "method": "Divine→Chaos Flip"}

            # 4. Flip: Buy Chaos → Sell Divine
            sell_for_divine = avg_chaos_buy / avg_divine_sell
            profit = (sell_for_divine *
                      self.chaos_to_divine_rate - 1) * 100 // 1
            results[item].append(
                {"profit": profit, "method": "Chaos→Divine Flip"})
            if profit > best_result["profit"]:
                best_result = {"item": item, "profit": profit,
                               "method": "Chaos→Divine Flip"}
        for k, v in results.items():
            v.sort(key=lambda x: x['profit'], reverse=True)
        return best_result, results

    def get_top_profit(self, profit: dict[str, list[dict[str, float | None]]], top_n: int = 3) -> list[dict[str, float | str | None]]:
        """
        Get the top N unique profitable items.

        Args:
            profit (dict): Profit results from calculate_best_profit.
            top_n (int): Number of top items to return.

        Returns:
            list: List of top N profit results.
        """
        all_profits = []
        seen_items = set()
        for item, methods in profit.items():
            for method in methods:
                all_profits.append({'item': item, **method})
        all_profits.sort(key=lambda x: x['profit'], reverse=True)
        res = []
        while len(res) < top_n:
            for p in all_profits:
                if p['item'] not in seen_items:
                    res.append(p)
                    seen_items.add(p['item'])
        return res


def extract(img_path: str = "img/screenshot2.png"):

    # usage
    # input
    mv = MovementHandler(ConfigPositions())
    items = ["Cartography Scarab of Risk",
             "Fine delirium Orb", "Delirium Scarab of Paranoia"]
    # items = ["Cartography Scarab of Risk"]
    items_with_prices: dict[str,
                            dict[str, list[list[tuple[str, str]]]]] = dict()

    # for item in items:
    #     mv.change_item(item, wanted=True)
    #     prices = mv.extract_price()
    #     items_with_prices[item] = prices
    # chaos_to_divine = mv.extract_chaos_to_divine_ratio()

    # print(items_with_prices)
    # print(chaos_to_divine)
    # exit(0)

    items_with_prices = {'Cartography Scarab of Risk': {'Chaos Orb': [[('1:64', '12'), ('1:65', '28')], [('1:61', '5,612'), ('1:60', '4,380')]], 'Divine Orb': [[('2.12:1', '256'), ('2.10:1', '197')], [('2.45:1', '80'), ('2.50:1', '218')]]}, 'Fine delirium Orb': {'Chaos Orb': [[('1:46.67', '69'), ('1:47', '353')], [(
        '1:36', '2,628'), ('1:35', '9,485')]], 'Divine Orb': [[('2.90:1', '29'), ('2.81:1', '211')], [('3:1', '869'), ('3.12:1', '16')]]}, 'Delirium Scarab of Paranoia': {'Chaos Orb': [[('1:9.50', '124'), ('1:10', '113')], [('1:8', '800'), ('1:7', '826')]], 'Divine Orb': [[('11:1', '143'), ('10:1', '170')], [('13:1', '135'), ('13.50:1', '52')]]}}
    chaos_to_divine = [[('136:1', '3,128'), ('135.50:1', '1,897')], [
        ('138:', '100'), ('139:1', '113')]]

    # chaos_to_divine = [[('144:1', '120'), ('143:1', '329')], [('147:1', '555'), ('148:1', '20')]]
    # items_with_prices = {'Cartography Scarab of Risk': {'Chaos Orb': [[('1:53.90', '50'), ('1:55', '101')], [('1:53.20', '3,320'), ('1:53', '3,710')]], 'Divine Orb': [[('2.35:1', '120'), ('2.35:1', '329')], [('2.80:1', '555'), ('2.90:1', '20')]]}, 'Fine delirium Orb': {'Chaos Orb': [[('1:39', '2'), ('1:39.49', '51')], [('1:33', '2,310'), ('1:32', '1,120')]], 'Divine Orb': [[('2.80:1', '2,240'), ('2.70:1', '667')], [('3:1', '742'), ('3.30:1', '947')]]}}

    # items_with_prices = {'Cartography Scarab of Risk': {'Chaos Orb': [[('1:67', '2,078'), ('1:67.50', '6')]], 'Divine Orb': [[('2.15:1', '129'), ('2.12:1', '256')], [('2.30:1', '10'), ('2.33:1', '6')]]}}
    # chaos_to_divine = [[('139:1', '11,815'), ('138:', '5,658')], [('141:1', '13'), ('142:1', '165')]]

    pf = ProfitCalculator(items_with_prices, chaos_to_divine)
    best_flip, res = pf.calculate_best_profit()
    print(f"Best flip: {best_flip}")
    print(pf.get_top_profit(res, top_n=3))


# Example usage
if __name__ == "__main__":
    time.sleep(2)
    logger = logging
    logger.basicConfig(filename='log.log', level=logging.DEBUG,
                       format='%(levelname)s: %(name)s - %(asctime)s - %(message)s', datefmt='%d/%b/%y %H:%M:%S')
    extract()
