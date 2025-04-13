import requests
import os
from dotenv import load_dotenv
load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
if not PEXELS_API_KEY:
    raise ValueError("PEXELS_API_KEY environment variable is not set.")

# Define the URL for fetching curated photos
url = "https://api.pexels.com/v1/curated?per_page=1"

# Send the request with authentication
headers = {"Authorization": PEXELS_API_KEY}

queries = ["paper"]  # Change this to your search term
i = 0
for query in queries:
    url = f"https://api.pexels.com/v1/search?query={query}&per_page=40"
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Print all image URLs
    # Create directory if it doesn't exist
    os.makedirs('./v3/pexel_background_images', exist_ok=True)
    for idx, photo in enumerate(data["photos"]):
        i += 1
        image_url = photo["src"]["large"]
        image_data = requests.get(image_url).content
        
        with open(f'./images/image_{i}.jpg', 'wb') as handler:
            handler.write(image_data)
