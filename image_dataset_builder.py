"""
╔══════════════════════════════════════════════════════════════╗
║           AI IMAGE DATASET BUILDER  v1.0                     ║
║           by Mateo Algarra | github.com/MateoAlgarra         ║
║                                                              ║
║  Automatically collects, filters, deduplicates and          ║
║  organizes image datasets ready for ML training             ║
║  (TensorFlow / PyTorch compatible)                          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import csv
import hashlib
import time
import random
import requests
import argparse
from io import BytesIO
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse

try:
    from PIL import Image
    from bs4 import BeautifulSoup
    from tqdm import tqdm
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    print("Installing required libraries...")
    os.system("pip install pillow beautifulsoup4 requests tqdm colorama --break-system-packages -q")
    from PIL import Image
    from bs4 import BeautifulSoup
    from tqdm import tqdm
    from colorama import Fore, Style, init
    init(autoreset=True)


# ─── CONFIG ───────────────────────────────────────────────────
DEFAULT_MIN_WIDTH  = 224
DEFAULT_MIN_HEIGHT = 224
DEFAULT_OUTPUT_DIR = "dataset_output"
DEFAULT_DELAY_MIN  = 2.0
DEFAULT_DELAY_MAX  = 5.0

HEADERS_POOL = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/119.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/118.0 Safari/537.36"},
]
# ──────────────────────────────────────────────────────────────


def banner():
    print(Fore.CYAN + """
  ╔══════════════════════════════════════════╗
  ║      AI IMAGE DATASET BUILDER  v1.0     ║
  ╚══════════════════════════════════════════╝
""")


def log_info(msg):    print(Fore.CYAN    + f"  [INFO]  " + Style.RESET_ALL + msg)
def log_ok(msg):      print(Fore.GREEN   + f"  [ OK ]  " + Style.RESET_ALL + msg)
def log_skip(msg):    print(Fore.YELLOW  + f"  [SKIP]  " + Style.RESET_ALL + msg)
def log_error(msg):   print(Fore.RED     + f"  [ERR ]  " + Style.RESET_ALL + msg)


def md5_hash(data: bytes) -> str:
    """Returns MD5 hash of raw bytes — used to detect duplicate images."""
    return hashlib.md5(data).hexdigest()


def is_valid_image(data: bytes, min_w: int, min_h: int) -> tuple[bool, int, int]:
    """
    Validates image quality:
    - Must be a real image (not corrupt)
    - Width  >= min_w
    - Height >= min_h
    Returns (is_valid, width, height)
    """
    try:
        img = Image.open(BytesIO(data))
        w, h = img.size
        if w >= min_w and h >= min_h:
            return True, w, h
        return False, w, h
    except Exception:
        return False, 0, 0


def scrape_image_urls(page_url: str) -> list[str]:
    """
    Scrapes all <img> tag sources from a given URL.
    Returns a list of absolute image URLs.
    """
    try:
        headers = random.choice(HEADERS_POOL)
        resp = requests.get(page_url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        urls = []
        for tag in soup.find_all("img"):
            src = tag.get("src") or tag.get("data-src") or tag.get("data-lazy-src")
            if src:
                abs_url = urljoin(page_url, src)
                if abs_url.startswith("http"):
                    urls.append(abs_url)
        return urls

    except Exception as e:
        log_error(f"Could not scrape {page_url}: {e}")
        return []


def download_image(url: str) -> bytes | None:
    """Downloads an image and returns raw bytes. Returns None on failure."""
    try:
        headers = random.choice(HEADERS_POOL)
        resp = requests.get(url, headers=headers, timeout=15, stream=True)
        if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
            return resp.content
        return None
    except Exception:
        return None


def safe_filename(url: str, index: int) -> str:
    """Generates a safe filename from URL or fallback index."""
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    name = "".join(c for c in name if c.isalnum() or c in "._-")
    if not name or "." not in name:
        name = f"image_{index:04d}.jpg"
    return name


def build_dataset(
    sources: list[str],
    categories: list[str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    min_w: int = DEFAULT_MIN_WIDTH,
    min_h: int = DEFAULT_MIN_HEIGHT,
    max_per_category: int = 100,
):
    """
    Main pipeline:
    1. Scrapes image URLs from source pages
    2. Downloads each image
    3. Validates resolution
    4. Removes duplicates (MD5)
    5. Saves to category folder
    6. Writes CSV manifest
    """
    banner()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    manifest_rows: list[dict] = []
    total_saved = 0
    total_skipped = 0
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for cat_index, (url, category) in enumerate(zip(sources, categories)):
        log_info(f"Processing category: {Fore.MAGENTA}{category}{Style.RESET_ALL} ← {url}")

        cat_dir = output_path / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        image_urls = scrape_image_urls(url)
        log_info(f"Found {len(image_urls)} image URLs on page")

        saved_count = 0
        for i, img_url in enumerate(tqdm(image_urls, desc=f"  {category}", unit="img")):
            if saved_count >= max_per_category:
                log_info(f"Reached max ({max_per_category}) for '{category}', moving on.")
                break

            # Polite delay to avoid rate limiting
            delay = random.uniform(DEFAULT_DELAY_MIN, DEFAULT_DELAY_MAX)
            time.sleep(delay)

            # Download
            raw = download_image(img_url)
            if not raw:
                log_skip(f"Download failed: {img_url[:60]}...")
                total_skipped += 1
                continue

            # Duplicate check
            h = md5_hash(raw)
            if h in seen_hashes:
                log_skip("Duplicate detected — skipped")
                total_skipped += 1
                continue
            seen_hashes.add(h)

            # Resolution validation
            valid, w, height = is_valid_image(raw, min_w, min_h)
            if not valid:
                log_skip(f"Too small ({w}x{height}) — min {min_w}x{min_h}")
                total_skipped += 1
                continue

            # Save file
            filename = safe_filename(img_url, i)
            filepath = cat_dir / filename
            filepath.write_bytes(raw)

            manifest_rows.append({
                "filename": str(filepath.relative_to(output_path)),
                "class_label": category,
                "width": w,
                "height": height,
                "source_url": img_url,
                "md5": h,
                "collected_at": run_id,
            })

            saved_count  += 1
            total_saved  += 1
            log_ok(f"Saved: {filename}  ({w}x{height})")

    # ── CSV Manifest ──────────────────────────────────────────
    csv_path = output_path / f"manifest_{run_id}.csv"
    fieldnames = ["filename", "class_label", "width", "height", "source_url", "md5", "collected_at"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # ── Summary ───────────────────────────────────────────────
    print(Fore.CYAN + "\n  ╔══════════════════ SUMMARY ══════════════════╗")
    print(Fore.GREEN  + f"  ║  ✔ Images saved   : {total_saved:<26}║")
    print(Fore.YELLOW + f"  ║  ✗ Skipped        : {total_skipped:<26}║")
    print(Fore.CYAN   + f"  ║  📄 Manifest CSV  : {str(csv_path):<26}║")
    print(Fore.CYAN   + f"  ║  📁 Output folder : {output_dir:<26}║")
    print(Fore.CYAN   +  "  ╚═════════════════════════════════════════════╝\n")


# ─── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Image Dataset Builder — Mateo Algarra"
    )
    parser.add_argument("--sources",    nargs="+", required=True,  help="List of page URLs to scrape")
    parser.add_argument("--categories", nargs="+", required=True,  help="Category label for each URL")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT_DIR, help="Output folder")
    parser.add_argument("--min-width",  type=int, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_HEIGHT)
    parser.add_argument("--max-per-cat",type=int, default=100, help="Max images per category")
    args = parser.parse_args()

    if len(args.sources) != len(args.categories):
        print(Fore.RED + "  [ERR] --sources and --categories must have the same number of items.")
        exit(1)

    build_dataset(
        sources=args.sources,
        categories=args.categories,
        output_dir=args.output,
        min_w=args.min_width,
        min_h=args.min_height,
        max_per_category=args.max_per_cat,
    )
