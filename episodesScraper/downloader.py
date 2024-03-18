import os

import settings
import json
import requests
from pathlib import Path

data_path = settings.DATA_DIR

# Define the directory where videos will be saved
DOWNLOAD_DIR = settings.VIDEO_DIR
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Define the path to the JSON file
JSON_FILE_PATH = data_path / 'successful_urls.json'

def load_urls():
    with open(JSON_FILE_PATH, 'r') as file:
        return json.load(file)

def save_urls(data):
    temp_file_path = JSON_FILE_PATH.with_suffix('.tmp')
    # Write the updated data to a temporary file
    with open(temp_file_path, 'w') as file:
        json.dump(data, file, indent=4)
    # Atomically replace the original file with the temp file
    os.replace(temp_file_path, JSON_FILE_PATH)

def download_video(url, path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    urls_data = load_urls()

    for video_info in urls_data:
        if not video_info['downloaded']:
            print(f"Downloading {video_info['url']}...")
            file_path = DOWNLOAD_DIR / video_info['filename']
            success = download_video(video_info['url'], file_path)
            video_info['downloaded'] = success
            # Save the updated status immediately after each download attempt
            save_urls(urls_data)
            if success:
                print(f"Successfully downloaded to {file_path}")
            else:
                print(f"Failed to download {video_info['url']}")

    print("All downloads attempted. JSON file updated.")

if __name__ == "__main__":
    main()