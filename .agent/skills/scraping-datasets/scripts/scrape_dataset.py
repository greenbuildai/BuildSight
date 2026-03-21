import requests
from bs4 import BeautifulSoup
import os
import time
import argparse
from urllib.parse import urljoin, urlparse

# --- CONFIGURATION (Modify these for your target) ---
DEFAULT_URL = "https://example.com/dataset"
DEFAULT_OUTPUT_DIR = "dataset_raw"
# Selector for the container of each item (optional, can scrape directly)
ITEM_SELECTOR = ".gallery-item" 
# Selector to find the actual data link/image source within the item or page
# Example: "img" to find image tags, then verify 'src' attribute
DATA_TAG = "img"
DATA_ATTR = "src" 
# Pagination: Selector for the 'Next' button
NEXT_PAGE_SELECTOR = "a.next-page"
# Standard headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def download_file(url, folder):
    """Downloads a file from a URL to the specified folder."""
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=10)
        response.raise_for_status()
        
        # Parse filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"file_{int(time.time()*1000)}.dat"
            
        filepath = os.path.join(folder, filename)
        
        # Avoid overwriting
        if os.path.exists(filepath):
            base, ext = os.path.splitext(filename)
            filename = f"{base}_{int(time.time())}{ext}"
            filepath = os.path.join(folder, filename)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def scrape_page(url, output_dir):
    """Scrapes a single page for data."""
    print(f"🔍 Scraping: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find items
        if ITEM_SELECTOR:
            items = soup.select(ITEM_SELECTOR)
        else:
            # If no item selector, look for data tags globally
            items = [soup] 

        count = 0
        for item in items:
            # Find the data element
            if ITEM_SELECTOR:
                # If we selected a container, look inside it
                elements = item.select(DATA_TAG)
            else:
                # Global search
                elements = item.select(DATA_TAG)
            
            for el in elements:
                src = el.get(DATA_ATTR)
                if src:
                    # Handle relative URLs
                    abs_url = urljoin(url, src)
                    if download_file(abs_url, output_dir):
                        count += 1
                        time.sleep(0.5) # Rate limit per file
        
        print(f"   -> Found {count} items on this page.")
        
        # Pagination
        next_link = soup.select_one(NEXT_PAGE_SELECTOR)
        if next_link and next_link.get('href'):
            return urljoin(url, next_link.get('href'))
        
        return None

    except Exception as e:
        print(f"❌ Failed to scrape page {url}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Web Scraper for Datasets")
    parser.add_argument("--url", default=DEFAULT_URL, help="Target URL to scrape")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--limit", type=int, default=100, help="Max pages to scrape")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"📂 Created directory: {args.output}")

    current_url = args.url
    page_count = 0
    
    while current_url and page_count < args.limit:
        next_url = scrape_page(current_url, args.output)
        page_count += 1
        
        if next_url:
            current_url = next_url
            print("⏳ Waiting before next page...")
            time.sleep(2) # Rate limit between pages
        else:
            print("🏁 No more pages found or end of pagination.")
            break

    print(f"\n🎉 Scrape complete! Checked {page_count} pages.")

if __name__ == "__main__":
    main()
