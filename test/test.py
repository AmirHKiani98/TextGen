import sys
import cv2
import numpy as np
from pathlib import Path

# Add the parent directory to the system path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from textgen_class import TextGen
from text_handler import TextHandler
from image_handler import ImageHandler
from glob import glob

def find_white_regions(img, white_threshold=200, top_n=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

    return np.array([cv2.boundingRect(cnt) for cnt in contours_sorted])



def main(text_obj):
    
    for image_path in glob("./images/*.jpg"):
        image_obj = ImageHandler(image_path=image_path, text_area_func=find_white_regions, text_area_func_args={"white_threshold": 200, "top_n": 1}, generate_text_func=text_obj.get_text)
        image_with_text = image_obj.add_text_to_regions()
        image_obj.show_image(image_with_text, title="Image with Text")
        

if __name__ == "__main__":
    text_obj = TextHandler(hf_dataset="Maximax67/English-Valid-Words", config_name="sorted_by_frequency", version="0.1.0")
    main(text_obj)
