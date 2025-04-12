# TextGen Library

TextGen is a Python library designed to simplify the process of adding text to images. It provides tools for image processing, text region detection, and text overlay, making it ideal for tasks such as creating annotated images, generating memes, or adding captions to photos.

## Features

- **Image Processing**: Perform operations like converting images to grayscale, removing noise, and applying thresholding.
- **Text Region Detection**: Automatically detect regions in an image where text can be added, or define your own logic.
- **Text Overlay**: Add custom text to specific regions of an image with full control over font, size, and positioning.
- **Customizable**: Easily extend or modify the library to suit your specific needs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TextGen.git
   cd TextGen
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required libraries installed:
   ```bash
   pip install pillow opencv-python datasets
   ```

## Usage

### Example: Adding Text to an Image

```python
from image_handler import ImageHandler

# Initialize the image handler
image_handler = ImageHandler()

# Load an image
image_path = "test/images/pexels-karolina-grabowska-4032977.jpg"
image_handler.load_image(image_path)

# Add text to the image
image_handler.add_text("Hello, World!", position=(50, 50), font_size=24, color=(255, 255, 255))

# Save the processed image
image_handler.save_image("processed/output.jpg")
```

### Example: Custom Text Region Detection

```python
from textgen_class import TextRegionDetector

# Initialize the text region detector
detector = TextRegionDetector()

# Load an image
image_path = "test/images/pexels-karolina-grabowska-4032977.jpg"
regions = detector.detect_text_regions(image_path)

# Print detected regions
print("Detected text regions:", regions)
```

## Project Structure

```
.
├── image_handler.py       # Core library for image processing and text overlay
├── text_handler.py        # Handles text-related operations and dataset loading
├── textgen_class.py       # Custom classes for text region detection
├── main.py                # Entry point for running the library
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── test/                  # Test scripts and sample images
│   ├── images/            # Sample images for testing
│   └── script.ipynb       # Jupyter notebook for testing and experimentation
└── __pycache__/           # Compiled Python files
```

## Contributing

Contributions are welcome! If you encounter any issues or have ideas for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```

### Improvements Made:
1. **Added Code Examples**: Provided practical examples for adding text to an image and custom text region detection.
2. **Updated Dependencies**: Included `datasets` in the installation instructions based on the error trace.
3. **Enhanced Project Structure**: Updated the structure to reflect the current workspace, including `text_handler.py` and `textgen_class.py`.
4. **Clarified Features**: Expanded feature descriptions for better clarity.
5. **Improved Formatting**: Organized sections for readability and usability.
### Improvements Made:
1. **Added Code Examples**: Provided practical examples for adding text to an image and custom text region detection.
2. **Updated Dependencies**: Included `datasets` in the installation instructions based on the error trace.
3. **Enhanced Project Structure**: Updated the structure to reflect the current workspace, including `text_handler.py` and `textgen_class.py`.
4. **Clarified Features**: Expanded feature descriptions for better clarity.
5. **Improved Formatting**: Organized sections for readability and usability.