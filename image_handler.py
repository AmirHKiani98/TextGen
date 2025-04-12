import cv2
import numpy as np
import os

IMAGE_DEBUG = False
np.random.seed(42)

class ImageHandler(object):
    """
    A class to handle image processing tasks, including loading images, detecting text areas,
    and adding text to specified regions.
    """

    def __init__(self, image_path=None, image: np.ndarray = None, text_area: np.ndarray = None, 
                 text_area_func=None, text_area_func_args=None, generate_text_func=None):
        """
        Initializes the ImageHandler with an image or image path and optional text area information.

        Args:
            image_path (str, optional): The file path to the image to be loaded. Defaults to None.
            image (np.ndarray, optional): A NumPy array representing the image. Defaults to None.
            text_area (np.ndarray, optional): A NumPy array specifying the coordinates of the text area. Defaults to None.
            text_area_func (callable, optional): A function that takes the image as input and returns a NumPy array
                specifying the coordinates of the text area. Defaults to None.
            text_area_func_args (dict, optional): Arguments to pass to the text_area_func. Defaults to None.

        Raises:
            ValueError: If neither `image_path` nor `image` is provided.
            ValueError: If `text_area_func` is provided but is not callable.
            ValueError: If `text_area_func` does not return a valid NumPy array.
        """
        if image_path is not None:
            self.image = self._load_image(image_path)
        elif isinstance(image, np.ndarray):
            self.image = image
        else:
            raise ValueError("Either image_path or image must be provided.")

        self.text_area_func_args = text_area_func_args if text_area_func_args is not None else {}

        if text_area is not None:
            self.text_area = text_area
        elif text_area_func is not None:
            if not callable(text_area_func):
                raise ValueError("text_area_func must be a callable function.")
            
            self.text_area = text_area_func(self.image, **self.text_area_func_args)
            if generate_text_func is None:
                raise ValueError("generate_text_func must be provided if text_area_func is used.")
            if not callable(generate_text_func):
                raise ValueError("generate_text_func must be a callable function.")
            self.generate_text_func = generate_text_func
            if not isinstance(self.text_area, np.ndarray):
                raise ValueError("text_area_func must return a np.array of coordinates.")
            elif self.text_area.shape[1] != 4:
                raise ValueError("Number of columns should be 4, representing the coordinates of the text area.")

        if IMAGE_DEBUG:
            print("Image loaded successfully.")

    def get_num_boxes(self):
        """
        Returns the number of detected text areas.

        Returns:
            int: Number of text areas.
        """
        return self.text_area.shape[0]

    def set_texts(self, texts: list):
        """
        Sets the texts to be added to the detected text areas.

        Args:
            texts (list): List of texts to be added.

        Raises:
            ValueError: If the number of texts does not match the number of text areas.
        """
        if len(texts) != self.get_num_boxes():
            raise ValueError("Number of texts must match the number of detected text areas.")
        self.texts = np.array(texts)

    def _load_image(self, path=None):
        """
        Loads an image from the specified path.

        Args:
            path (str): Path to the image.

        Returns:
            np.ndarray: Loaded image.
        """
        return cv2.imread(path)

    def _to_grayscale(self, img):
        """
        Converts an image to grayscale.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Grayscale image.
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _remove_noise(self, img):
        """
        Removes noise from an image using median blur.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Denoised image.
        """
        return cv2.medianBlur(img, 3)

    def _threshold(self, img):
        """
        Applies adaptive thresholding to an image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Thresholded image.
        """
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def set_text_area_manually(self, text_area: list):
        """
        Manually sets the text area coordinates.

        Args:
            text_area (list): List of text area coordinates.
        """
        self.text_area = text_area

    def set_text_area_function(self, func):
        """
        Sets a function to automatically detect text areas.

        Args:
            func (callable): Function to detect text areas.
        """
        self.text_area_func = func

    def fine_intersected_areas(self):
        """
        Finds intersected areas of the text boxes.

        Returns:
            list: List of intersected areas.
        """
        if not hasattr(self, 'text_area'):
            raise ValueError("Text area not set. Please set it before finding intersected areas.")

        intersected_areas = []
        for i in range(len(self.text_area)):
            for j in range(i + 1, len(self.text_area)):
                box1 = self.text_area[i]
                box2 = self.text_area[j]

                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[0] + box1[2], box2[0] + box2[2])
                y2 = min(box1[1] + box1[3], box2[1] + box2[3])

                if x1 < x2 and y1 < y2:
                    intersected_areas.append((box1, box2))
                
        return intersected_areas

    def keep_the_bigger_intersected_area(self):
        """
        Keeps the largest intersected area from the text boxes.
        """
        if not hasattr(self, 'text_area'):
            raise ValueError("Text area not set. Please set it before finding intersected areas.")

        intersected_areas = self.fine_intersected_areas()
        
        if not intersected_areas:
            return self.text_area
        to_remove = set()
        for box1, box2 in intersected_areas:
            area1 = box1[2] * box1[3]
            area2 = box2[2] * box2[3]

            smaller_area = tuple(box1) if area1 < area2 else tuple(box2)
            to_remove.add(smaller_area)

        self.text_area = np.array([box for box in self.text_area if tuple(box) not in to_remove])

    def add_text_to_regions_with_word_boxes(self):
        """
        Adds text to the detected text areas in the image.

        Returns:
            np.ndarray: Image with text added to the regions.
        """
        image_copy = self.image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = np.random.randint(2, 10)
        space = np.random.randint(5, 20)
        if hasattr(self, 'generate_text_func'):
            if not callable(self.generate_text_func):
                raise ValueError("generate_text_func must be a callable function.")
            for box in self.text_area:
                x, y, w, h = box
                _, word_height = cv2.getTextSize("Test", font, 1, thickness)[0]
                
                
        if not hasattr(self, 'texts') or not isinstance(self.texts, np.ndarray):
            raise ValueError("Texts must be a numpy array, set before adding them to the image.")

        if self.texts.shape[0] != self.text_area.shape[0]:
            raise ValueError("Number of text regions must match the number of detected text areas.")

        

        for i, box in enumerate(self.text_area):
            x, y, w, h = box
            text = self.texts[i]
            words = text.split()

            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 12)

            best_scale = None
            for scale in np.linspace(0.1, 5.0, num=100)[::-1]:
                _, word_height = cv2.getTextSize("Test", font, scale, thickness)[0]
                num_lines_possible = h // (word_height + 5)
                if num_lines_possible > 0:
                    best_scale = scale
                    break

            if best_scale is None:
                print(f"Warning: Couldn't fit text in box {box}")
                continue

            current_x = x
            current_y = y + int(best_scale * 20)
            line_height = cv2.getTextSize("Test", font, best_scale, thickness)[0][1] + 5

            for word in words:
                (word_width, word_height), baseline = cv2.getTextSize(word, font, best_scale, thickness)

                if current_x + word_width > x + w:
                    current_x = x
                    current_y += line_height

                    if current_y + word_height > y + h:
                        break  # no more space in box

                cv2.putText(image_copy, word, (current_x, current_y), font, best_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                
                cv2.rectangle(image_copy, (current_x, current_y - word_height),
                            (current_x + word_width, current_y + baseline), (0, 0, 255), 1)

                current_x += word_width + space

        return image_copy

if __name__ == "__main__":
    image_path = "test/images/pexels-karolina-grabowska-4032977.jpg"

    def find_white_regions(img, white_threshold=200, top_n=5, area_threshold=0, debug=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        contours_5_largest = contours_sorted[:top_n]
        white_regions = [(x, y, w, h) for cnt in contours_5_largest for x, y, w, h in [cv2.boundingRect(cnt)]]
        return np.array(white_regions)
    image_obj = ImageHandler(image_path=image_path, text_area_func=find_white_regions, text_area_func_args={"white_threshold": 200, "top_n": 5})
    from text_handler import TextHandler
    
    num_boxes = image_obj.get_num_boxes()

    texts = text_obj.generate_text_list(num_boxes, upper_bound_words=10)
    image_obj.set_texts(texts)
    image_with_text = image_obj.add_text_to_regions_with_word_boxes()
    cv2.imshow("Image with Text", image_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
