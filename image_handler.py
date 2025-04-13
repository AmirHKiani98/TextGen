import cv2
import numpy as np
import os

IMAGE_DEBUG = False
class ImageHandler:
    """
    A class to handle image processing tasks, including loading images, detecting text areas,
    and adding text to specified regions.
    """

    def __init__(self, image_path=None, image: np.ndarray = None, text_area: np.ndarray = None, 
                 text_area_func=None, text_area_func_args=None, generate_text_func=None):
        """
        Initializes the ImageHandler with an image or image path and optional text area information.
        """
        self._initialize_image(image_path, image)
        self.text_area_func_args = text_area_func_args or {}
        self._initialize_text_area(text_area, text_area_func, generate_text_func)
        self.word_boxes = []
        self.words = []
        if IMAGE_DEBUG:
            print("Image loaded successfully.")

    def _initialize_image(self, image_path, image):
        if image_path:
            self.image = self._load_image(image_path)
        elif isinstance(image, np.ndarray):
            self.image = image
        else:
            raise ValueError("Either image_path or image must be provided.")

    def _initialize_text_area(self, text_area, text_area_func, generate_text_func):
        if text_area is not None:
            self.text_area = text_area
        elif text_area_func is not None:
            if not callable(text_area_func):
                raise ValueError("text_area_func must be a callable function.")
            self.text_area = text_area_func(self.image, **self.text_area_func_args)
            self._validate_text_area_func(generate_text_func)
        else:
            self.text_area = None

    def _validate_text_area_func(self, generate_text_func):
        if generate_text_func is None or not callable(generate_text_func):
            raise ValueError("generate_text_func must be a callable function.")
        self.generate_text_func = generate_text_func
        if not isinstance(self.text_area, np.ndarray) or self.text_area.shape[1] != 4:
            raise ValueError("text_area_func must return a valid np.array with 4 columns.")

    def get_num_boxes(self):
        """Returns the number of detected text areas."""
        return self.text_area.shape[0]

    def set_texts(self, texts: list):
        """
        Sets the texts to be added to the detected text areas.
        """
        if len(texts) != self.get_num_boxes():
            raise ValueError("Number of texts must match the number of detected text areas.")
        self.texts = np.array(texts)

    def _load_image(self, path):
        """Loads an image from the specified path."""
        return cv2.imread(path)

    def _to_grayscale(self, img):
        """Converts an image to grayscale."""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _remove_noise(self, img):
        """Removes noise from an image using median blur."""
        return cv2.medianBlur(img, 3)

    def _threshold(self, img):
        """Applies adaptive thresholding to an image."""
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def set_text_area_manually(self, text_area: list):
        """Manually sets the text area coordinates."""
        self.text_area = text_area

    def set_text_area_function(self, func):
        """Sets a function to automatically detect text areas."""
        self.text_area_func = func

    def find_intersected_areas(self):
        """
        Finds intersected areas of the text boxes.
        """
        if not hasattr(self, 'text_area'):
            raise ValueError("Text area not set. Please set it before finding intersected areas.")

        intersected_areas = []
        for i, box1 in enumerate(self.text_area):
            for box2 in self.text_area[i + 1:]:
                x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
                x2, y2 = min(box1[0] + box1[2], box2[0] + box2[2]), min(box1[1] + box1[3], box2[1] + box2[3])
                if x1 < x2 and y1 < y2:
                    intersected_areas.append((box1, box2))
        return intersected_areas

    def keep_larger_intersected_areas(self):
        """
        Keeps the largest intersected area from the text boxes.
        """
        if not hasattr(self, 'text_area'):
            raise ValueError("Text area not set. Please set it before finding intersected areas.")

        intersected_areas = self.find_intersected_areas()
        if not intersected_areas:
            return self.text_area

        to_remove = set()
        for box1, box2 in intersected_areas:
            area1, area2 = box1[2] * box1[3], box2[2] * box2[3]
            smaller_area = tuple(box1) if area1 < area2 else tuple(box2)
            to_remove.add(smaller_area)

        self.text_area = np.array([box for box in self.text_area if tuple(box) not in to_remove])

    def add_text_to_regions(self):
        """
        Adds text to the detected text areas in the image.
        """
        if not hasattr(self, 'texts') and not hasattr(self, 'generate_text_func'):
            raise ValueError("Either texts or generate_text_func must be set before adding text to regions.")
        if hasattr(self, 'texts') and not isinstance(self.texts, np.ndarray):
            raise ValueError("texts must be a numpy array.")
        if hasattr(self, 'texts'):
            if self.texts.shape[0] != self.text_area.shape[0]:
                raise ValueError("Number of text regions must match the number of detected text areas.")

        self.image_with_rect = self.image.copy()
        self.image_without_rect = self.image.copy()
        font, thickness, space = cv2.FONT_HERSHEY_SIMPLEX, np.random.randint(2, 10), np.random.randint(5, 20)

        for i, box in enumerate(self.text_area):
            x, box_y, w, h = box
            current_y = box_y
            best_scale = self._find_best_scale(h, font, thickness)
            if not hasattr(self, 'generate_text_func'):
                self._add_text_to_box(image_with_rect, box, current_y, best_scale, font, thickness, space, self.texts[i])
            else:
                out_of_bounds = False
                while not out_of_bounds:
                    text = self.generate_text_func(np.random.randint(2, 10))
                    current_y = self._add_text_to_box(image_with_rect, box, current_y, best_scale, font, thickness, space, text)
                    if current_y + space > box_y + h:
                        out_of_bounds = True
                        break
                    

        return self.image_with_rect, self.image_without_rect

    def _find_best_scale(self, box_height, font, thickness):
        for scale in np.linspace(0.1, 5.0, num=100)[::-1]:
            _, word_height = cv2.getTextSize("Test", font, scale, thickness)[0]
            if box_height // (word_height + 5) > 0:
                return scale
        return 1.0

    def _add_text_to_box(self, image, box, sentence_y, scale, font, thickness, space, text):
        x, y, w, h = box
        current_x = x
        line_height = cv2.getTextSize("Test", font, scale, thickness)[0][1] + space
        current_y = sentence_y + line_height

        words = text.split()
        for word in words:
            word_size, baseline = cv2.getTextSize(word, font, scale, thickness)
            word_width, word_height = word_size
            if current_x + word_width > x + w:
                current_x = x
                current_y += line_height
                if current_y + word_height > y + h:
                    break
            cv2.putText(image, word, (current_x, current_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
            top_left, bottom_right = (current_x, current_y - word_height), (current_x + word_width, current_y)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 10)
            self.words.append(word)
            self.word_boxes.append((top_left, bottom_right))
            current_x += word_width + space
        return current_y
    def show_image(self, img, title="Image"):
        """
        Displays the image using OpenCV.
        """
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == "__main__":
    image_path = "test/images/pexels-karolina-grabowska-4032977.jpg"

    def find_white_regions(img, white_threshold=200, top_n=5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

        return np.array([cv2.boundingRect(cnt) for cnt in contours_sorted][:1])

    from text_handler import TextHandler
    text_obj = TextHandler(hf_dataset="Maximax67/English-Valid-Words", config_name="sorted_by_frequency", version="0.1.0")
    image_obj = ImageHandler(image_path=image_path, text_area_func=find_white_regions, text_area_func_args={"white_threshold": 200, "top_n": 5}, generate_text_func=text_obj.get_text)

    
    image_with_text = image_obj.add_text_to_regions()

    for box in image_obj.text_area:
        x, y, w, h = box
        cv2.rectangle(image_with_text, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Image with Text", image_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
