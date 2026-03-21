"""
Re-analyze PPE_SASTRA_Dataset_3 & 4 specifically for Low Light conditions.
Uses a targeted prompt that focuses on low-light visual indicators.
Moves confirmed low-light images to the Low_Light_Condition folder.
"""
import os, sys, time, shutil, threading, logging, argparse
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from PIL import Image
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset")

SOURCE_DATASETS = ["PPE_SASTRA_Dataset_3", "PPE_SASTRA_Dataset_4"]

LOW_LIGHT_DEST = BASE_DIR / "Low_Light_Condition"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".mpo"}

SAFE_RPM_PER_KEY = 7   # Using 7 RPM since two scripts will run concurrently
MAX_RETRIES      = 7
MODEL_NAME       = "gemini-2.5-flash"

# Targeted prompt specifically for low-light detection
LOW_LIGHT_PROMPT = """
You are an expert image analyst specializing in lighting conditions on construction sites.

Analyze this construction site image and determine if it shows LOW LIGHT conditions.

LOW LIGHT indicators (answer YES if ANY of these are clearly present):
- Dark or dim overall scene
- Heavy shadows covering large portions of the image
- Artificial lighting being the primary light source (construction lights, LED panels)
- Dusk, dawn, or nighttime shooting conditions
- Indoor construction areas with insufficient lighting
- Underexposed image where details are hard to see
- Workers or equipment appear as silhouettes
- Visible light sources (lamps, headlights) contrasting against dark surroundings
- Overcast/very cloudy conditions causing significant dimming
- Tunnel, basement, or underground construction areas

NOT low light (answer NO):
- Bright sunlight, clear daytime conditions
- Well-lit outdoor scenes even if slightly overcast
- Normal indoor lighting where everything is clearly visible

Reply with ONLY one word: YES or NO
"""

# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════
log = logging.getLogger("LowLightRescan")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
fh = logging.FileHandler("lowlight_rescan_log_v2.txt", encoding="utf-8")
fh.setFormatter(fmt); log.addHandler(fh)

# ══════════════════════════════════════════════════════════════════════════════
# TokenBucket (same as main script)
# ══════════════════════════════════════════════════════════════════════════════
class TokenBucket:
    def __init__(self, rpm: int):
        self.capacity    = rpm
        self.tokens      = 0.0
        self.refill_rate = rpm / 60.0
        self.lock        = threading.Lock()
        self.last_refill = time.monotonic()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            time.sleep(0.2)

# ══════════════════════════════════════════════════════════════════════════════
# API Key Pool
# ══════════════════════════════════════════════════════════════════════════════
class ApiKeyPool:
    def __init__(self, api_keys: list):
        if not api_keys:
            raise ValueError("No API keys provided.")
        self.pool = []
        for key in api_keys:
            client = genai.Client(api_key=key)
            bucket = TokenBucket(SAFE_RPM_PER_KEY)
            self.pool.append({"client": client, "bucket": bucket, "key_suffix": key[-7:]})
        self._idx = 0
        self._lock = threading.Lock()
        log.info(f"[Pool] Initialized {len(self.pool)} API key(s). Max combined RPM: {len(self.pool) * SAFE_RPM_PER_KEY}")

    def get_slot(self):
        with self._lock:
            slot = self.pool[self._idx % len(self.pool)]
            self._idx += 1
        return slot

# ══════════════════════════════════════════════════════════════════════════════
# Image Analysis
# ══════════════════════════════════════════════════════════════════════════════
def is_low_light(image_path: Path, pool: ApiKeyPool) -> tuple:
    """Returns (path, True/False/None) — True if low light, False if not, None on failure."""
    slot = pool.get_slot()
    client = slot["client"]
    bucket = slot["bucket"]
    key_tag = slot["key_suffix"]

    try:
        img = Image.open(image_path).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        img_for_api = Image.open(buf)
    except Exception as e:
        log.error(f"  [ImgErr] {image_path.name}: {e}")
        return (image_path, None)

    for attempt in range(1, MAX_RETRIES + 1):
        bucket.acquire()
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[LOW_LIGHT_PROMPT, img_for_api],
            )
            answer = response.text.strip().upper()
            if "YES" in answer:
                return (image_path, True)
            else:
                return (image_path, False)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = min(4 * (2 ** (attempt - 1)), 120)
                log.warning(f"  [RateLimit] key=...{key_tag}  {image_path.name}  wait={wait}s  attempt={attempt}/{MAX_RETRIES}")
                time.sleep(wait)
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                wait = min(4 * (2 ** (attempt - 1)), 60)
                log.warning(f"  [Server503] key=...{key_tag}  {image_path.name}  wait={wait}s  attempt={attempt}/{MAX_RETRIES}")
                time.sleep(wait)
            else:
                log.error(f"  [Error] key=...{key_tag}  {image_path.name}  {e}")
                return (image_path, None)

    return (image_path, None)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Re-scan for low-light images")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load API keys
    api_keys = [
        "AIzaSyCLL1gsvh9fu0ayvZG5NHSsgW1PJkNl6iA"
    ]

    pool = ApiKeyPool(api_keys)
    LOW_LIGHT_DEST.mkdir(parents=True, exist_ok=True)


    # Collect images from source datasets
    all_images = []
    for dataset in SOURCE_DATASETS:
        src = BASE_DIR / dataset
        if not src.exists():
            log.warning(f"[Skip] {dataset} not found")
            continue
        found = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        log.info(f"[Discover] {dataset}: {len(found)} images")
        all_images.extend(found)

    if not all_images:
        log.error("[FATAL] No images found.")
        sys.exit(1)

    log.info(f"\n🔍  Re-scanning {len(all_images)} images for LOW LIGHT conditions")
    log.info(f"    Workers: {args.workers}  |  Keys: {len(api_keys)}  |  Max RPM: {len(api_keys) * SAFE_RPM_PER_KEY}\n")

    # Process
    low_light_found = []
    not_low_light = 0
    failed = 0
    stats_lock = threading.Lock()

    # Read previous log to resume if possible
    processed_files = set()
    old_log = Path("lowlight_rescan_log.txt")
    if old_log.exists():
        with open(old_log, "r", encoding="utf-8") as f:
            for line in f:
                if "[LOW_LIGHT]" in line or "[SKIP]" in line or "failed" in line:
                    parts = line.split("]")
                    if len(parts) > 1:
                        # Extract DSC0000.JPG
                        import re
                        m = re.search(r'(DSC\d+\.JPG)', line)
                        if m:
                            processed_files.add(m.group(1))

    # Also skip any file already in Low_Light_Condition
    for p in LOW_LIGHT_DEST.glob("*.JPG"):
        processed_files.add(p.name)

    filtered_images = [img for img in all_images if img.name not in processed_files]
    log.info(f"Skipping {len(all_images) - len(filtered_images)} images based on older runs.")
    all_images = filtered_images

    tasks = [(img, pool) for img in all_images]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(is_low_light, t[0], t[1]): t[0] for t in tasks}

        with tqdm(total=len(all_images), unit="img", ncols=90,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

            for future in as_completed(futures):
                path, result = future.result()

                if result is None:
                    with stats_lock:
                        failed += 1
                    log.error(f"  [SKIP] {path.name} — failed")
                elif result is True:
                    dest = LOW_LIGHT_DEST / path.name
                    if dest.exists():
                        dest = LOW_LIGHT_DEST / f"{path.stem}_{path.parent.name}{path.suffix}"

                    if not args.dry_run:
                        shutil.copy2(path, dest)
                    
                    with stats_lock:
                        low_light_found.append(path.name)
                    log.info(f"  [LOW_LIGHT] {path.name} → Low_Light_Condition")
                else:
                    with stats_lock:
                        not_low_light += 1
                    log.info(f"  [NOT_LOW_LIGHT] {path.name}")

                pbar.update(1)

    # Summary
    log.info("\n" + "=" * 55)
    log.info("  LOW LIGHT RE-SCAN COMPLETE — SUMMARY")
    log.info("=" * 55)
    log.info(f"  Total scanned:          {len(all_images)}")
    log.info(f"  Low Light found:        {len(low_light_found)}")
    log.info(f"  Not Low Light:          {not_low_light}")
    log.info(f"  Failed:                 {failed}")
    log.info("=" * 55)

    if low_light_found:
        log.info(f"\n  📁 {len(low_light_found)} images {'would be' if args.dry_run else 'were'} copied to Low_Light_Condition/")


if __name__ == "__main__":
    main()
