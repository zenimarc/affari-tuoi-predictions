import settings

data_path = settings.DATA_DIR
successful_urls_path = data_path / 'successful_urls.json'

import json
from pathlib import Path
from urllib.parse import urlparse, unquote

def load_urls():
    with open(successful_urls_path, 'r') as file:
        return json.load(file)

def extract_filename(url):
    # Parses the URL to extract the filename. Uses unquote to decode any percent-encoded characters.
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = Path(unquote(path)).name
    return filename

def migrate_urls(urls):
    migrated_data = []
    for url in urls:
        filename = extract_filename(url)
        video_info = {
            "url": url,
            "downloaded": False,  # Initially set to False; update after downloading
            "filename": filename
        }
        migrated_data.append(video_info)
    return migrated_data

def save_migrated_data(data):
    with open('../data/migrated_urls.json', 'w') as file:
        json.dump(data, file, indent=4)

def main():
    urls = load_urls()
    migrated_data = migrate_urls(urls)
    save_migrated_data(migrated_data)
    print("Migration completed. Migrated data saved to 'migrated_urls.json'.")

if __name__ == "__main__":
    main()