import cv2
import numpy as np
import os
import numpy as np


IMAGE_DEBUG = False
class ImageHandler(object):

    def __init__(self, image_path=None, image: np.ndarray = None, text_area:np.ndarray=None, text_area_func=None, text_area_func_args=None):
        """
        Initializes the image handler with an image or image path and optional text area information.
        Args:
            image_path (str, optional): The file path to the image to be loaded. Defaults to None.
            image (np.ndarray, optional): A NumPy array representing the image. Defaults to None.
            text_area (np.ndarray, optional): A NumPy array specifying the coordinates of the text area. Defaults to None.
            text_area_func (callable, optional): A function that takes the image as input and returns a NumPy array
                specifying the coordinates of the text area. Defaults to None.
        Raises:
            ValueError: If neither `image_path` nor `image` is provided.
            ValueError: If `text_area_func` is provided but is not callable.
            ValueError: If `text_area_func` does not return a NumPy array.
            ValueError: If the returned NumPy array from `text_area_func` does not have 4 columns, representing
                the coordinates of the text area.
        Notes:
            - Either `image_path` or `image` must be provided.
            - If `text_area_func` is used, it must return a NumPy array with 4 columns, representing the coordinates
              of the text area.
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
            if not isinstance(self.text_area, np.ndarray):
                raise ValueError("text_area_func must return a np.array of coordinates. Returned: {}".format(type(self.text_area)))
            elif self.text_area.shape[1] != 4:
                raise ValueError("Number of columns should be 4, showing the coordinates of the text area.") #TODO you can make this more general instead of having the user input only 4 coordinates
        
        
        
        if IMAGE_DEBUG:
            print("Image loaded successfully.")
        
        
    def get_num_boxes(self):
        return self.text_area.shape[0]
    
    def set_texts(self, texts: list):
        
        if len(texts) != self.get_num_boxes():
            raise ValueError("Number of texts must match the number of detected text areas.")
        self.texts = np.array(texts)
        # TODO check the length of each text



    def _load_image(self, path=None):
        if path is None:
            path = self.image_path
        return cv2.imread(path)

    def _to_grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _remove_noise(self, img):
        return cv2.medianBlur(img, 3)

    def _threshold(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def set_text_area_manually(self, text_area: list):
        self.text_area = text_area
    
    def set_text_area_function(self, func):
        self.text_area_func = func

    def add_text_to_regions_with_word_boxes(self):
        if not hasattr(self, 'texts') or not isinstance(self.texts, np.ndarray):
            raise ValueError("Texts must be a numpy array, set before adding them to the image.")
        
        if self.texts.shape[0] != self.text_area.shape[0]:
            raise ValueError("Number of text regions must match the number of detected text areas.")

        image_copy = self.image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = np.random.randint(2,10)

        for i, box in enumerate(self.text_area):
            x, y, w, h = box
            random_x = np.random.randint(x, x + w)
            random_y = np.random.randint(y, y + h)
            

            text = self.texts[i]
            words = text.split()

            # Fit font scale so the tallest word fits vertically
            font_scale = 10
            current_x = x
            space = np.random.randint(20, 40)
            
            for word in words:
                (word_width, word_height), baseline = cv2.getTextSize(word, font, font_scale, thickness)
                top_left_y = random_y - word_height
                
                word_y = y + (h + word_height) // 2
                cv2.putText(image_copy, word, (current_x, word_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.rectangle(image_copy, (current_x, y), (current_x + word_width, y + h), (0, 0, 255), 1)

                current_x += word_width + 5  # Small space between words

        return image_copy

        
    def add_curve(self, image):
        pass
    

        

if __name__ == "__main__":
    image_path = "test/images/pexels-karolina-grabowska-4032977.jpg"
    def find_white_regions(img, white_threshold=200, top_n=5, area_threshold=0, debug=False):
        # Convert to HSV for better color filtering
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Find contours in the white mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        
        contours_5_largest = contours_sorted[:top_n]

        white_regions = [(x, y, w, h) for cnt in contours_5_largest for x, y, w, h in [cv2.boundingRect(cnt)]]

        if debug:
            cv2.imshow("White Regions", img)
            cv2.imshow("White Mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return np.array(white_regions)
    image_obj = ImageHandler(image_path=image_path, text_area_func=find_white_regions, text_area_func_args={"white_threshold": 200, "top_n": 5})
    from text_handler import TextHandler
    text_obj = TextHandler()
    num_boxes = image_obj.get_num_boxes()
    texts = text_obj.generate_text_list(num_boxes, upper_bound_words=10)
    image_obj.set_texts(texts)
    image_with_text = image_obj.add_text_to_regions_with_word_boxes()
    cv2.imshow("Image with Text", image_with_text)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for x, y, w, h in image_obj.text_area:
    #     # Plot the rectangles on the image
    #     cv2.rectangle(image_obj.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("Detected Text Areas", image_obj.image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
