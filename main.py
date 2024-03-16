import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json

THREAD_POOL_SIZE = 400
RETRY_POOL_SIZE = 50
PROXY_TIMEOUT = 20

# Initialize global variables
current_proxies = []
successful_requests = 0
successful_urls = []  # List of successful URLs
timeout_urls = []  # List of URLs that timeout
lock = threading.Lock()

def retry_timeout_urls():
    global timeout_urls, current_proxies
    if not timeout_urls:
        print("No timeout URLs to retry.")
        return

    print(f"Retrying {len(timeout_urls)} URLs that previously timed out...")
    temp_timeout_urls = timeout_urls[:]
    timeout_urls.clear()  # Clear the list to start fresh for the new session

    with ThreadPoolExecutor(max_workers=RETRY_POOL_SIZE) as executor:  # Adjust the number of workers as needed
        future_to_url = {executor.submit(scrape_url, url, random.choice(current_proxies), "Retry"): url for url in temp_timeout_urls}
        for future in as_completed(future_to_url):
            try:
                future.result()
            except Exception as exc:
                print(f"URL generated an exception during retry: {exc}")

def remove_duplicates_from_successful_urls():
    global successful_urls
    with lock:
        successful_urls = list(set(successful_urls))
        print(f"Removed duplicates from successful URLs. {len(successful_urls)} unique URLs remain.")

def get_proxies_from_api(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        proxies = [{'http': f"http://{line}"} for line in response.text.strip().split('\n') if line]
        return proxies
    else:
        print(f"Failed to get proxies from API with status code: {response.status_code}")
        return []

def load_successful_urls():
    global successful_urls
    try:
        with open('data/successful_urls.json', 'r') as file:
            successful_urls = json.load(file)
    except FileNotFoundError:
        successful_urls = []

def load_timeout_urls():
    global timeout_urls
    try:
        with open('data/timeout_urls.json', 'r') as file:
            timeout_urls = json.load(file)
    except FileNotFoundError:
        timeout_urls = []

def scrape_url(url, proxy, sequential_number):
    global successful_requests, successful_urls, timeout_urls
    try:
        response = requests.get(url, proxies=proxy, timeout=PROXY_TIMEOUT)
        if response.status_code == 200:
            with lock:
                successful_requests += 1
                successful_urls.append(url)
            print(f"Success #{successful_requests}: {url} - Sequential Number: {sequential_number}")
            return True
    except requests.exceptions.Timeout:
        with lock:
            timeout_urls.append(url)  # Add the URL to the timeout list
        print(f"Timeout fetching {url} - Sequential Number: {sequential_number}")
        return False
    except Exception as e:
        print(f"Error fetching {url}: {e} - Sequential Number: {sequential_number}")
        return False

def save_successful_urls():
    with lock:
        with open('data/successful_urls.json', 'w') as file:
            json.dump(successful_urls, file)

def save_timeout_urls():
    with lock:
        with open('data/timeout_urls.json', 'w') as file:
            json.dump(timeout_urls, file)

def save_progress(last_number):
    with open('data/progress.json', 'w') as file:
        json.dump({'last_number': last_number}, file)

def load_progress():
    try:
        with open('data/progress.json', 'r') as file:
            progress = json.load(file)
            return progress['last_number']
    except (FileNotFoundError, KeyError):
        return None

def main(api_url, base_url, start, end):
    global current_proxies
    current_proxies = get_proxies_from_api(api_url)
    load_successful_urls()  # Load the list of successful URLs
    remove_duplicates_from_successful_urls()  # Remove duplicates from successful URLs
    load_timeout_urls()  # Load the list of timeout URLs
    retry_timeout_urls()  # Retry URLs that previously timed out

    last_number = load_progress()
    if last_number:
        start = last_number + 1  # Resume from the next number

    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        future_to_url = {executor.submit(scrape_url, base_url.format(number), random.choice(current_proxies), number): number for number in range(start, end + 1)}
        for future in as_completed(future_to_url):
            number = future_to_url[future]
            try:
                success = future.result()
                if success:
                    save_successful_urls()  # Save the updated list of successful URLs
                save_timeout_urls()  # Save the updated list of timeout URLs
                save_progress(number)  # Save progress after each attempt
            except Exception as exc:
                print(f"URL with Sequential Number: {number} generated an exception: {exc}")

    print(f"Successfully scraped {len(successful_urls)} URLs. {len(timeout_urls)} URLs timed out.")

if __name__ == "__main__":
    api_url = "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&proxy_format=ipport&format=text"
    base_url = "https://creativemedia4-rai-it.akamaized.net/podcastcdn/raiuno_2/Affari_Tuoi/Affarituoi_puntate/2{}_1800.mp4"
    # fatto 2410000 to 2499999 e 2500000 to 2699999 e 2200000 to 2399999 e 1500000 to 2199999
    start = 1000000
    end = 1499999  # Example range, adjust as needed
    main(api_url, base_url, start, end)