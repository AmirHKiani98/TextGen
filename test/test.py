import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import json
# Add the parent directory to the system path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from textgen_class import TextGen
from text_handler import TextHandler
from image_handler import ImageHandler
from glob import glob
from random import shuffle
from multiprocessing import Pool, cpu_count
random_string = lambda x: "".join([chr(np.random.randint(97, 123)) for _ in range(x)])

def find_white_regions(img, white_threshold=200, top_n=5):
    w, h = img.shape[1], img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([[0,0, w, h]])
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

    return np.array([cv2.boundingRect(cnt) for cnt in contours_sorted])

def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def process_image(image_path, text_obj):
    image_obj = ImageHandler(image_path=image_path, text_area_func=find_white_regions, text_area_func_args={"white_threshold": 200, "top_n": 1}, generate_text_func=text_obj.get_text)
    image_with_rect, image_without_rect = image_obj.add_text_to_regions()
    words, word_boxes = image_obj.words, image_obj.word_boxes
    name = random_string(100)
    while os.path.isfile(f"./output/annotations/{name}.txt"):
        name = random_string(100)
    
    with open(f"./output/annotations/{name}.json", "w") as f:
        data = {
            "words": words,
            "word_boxes": word_boxes,
            "image_path": image_path
        }
        json.dump(convert_to_python_types(data), f)
    cv2.imwrite(f"./output/images/{name}.jpg", image_without_rect)

def main(text_obj):
    files = glob("./images/*.jpg")
    # os.makedirs("./output/", exist_ok=True)
    os.makedirs("./output/images", exist_ok=True)
    os.makedirs("./output/annotations", exist_ok=True)
    simulation_number = 1_000_000
    shuffle(files)
    for image_path in files:
        print(f"starting {image_path}")
        args_list = [(image_path, text_obj) for _ in range(simulation_number)]
        print(f"processing {len(args_list)} images")
        with Pool(int(cpu_count()/2)) as pool:
            tqdm(pool.starmap(process_image, args_list), total=len(args_list), desc="Processing images", unit="image")
        print(f"finished {image_path}")
        
        

if __name__ == "__main__":
    text_obj = TextHandler(hf_dataset="Maximax67/English-Valid-Words", config_name="sorted_by_frequency", version="0.1.0")
    main(text_obj)
