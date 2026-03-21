"""
BuildSight — PPE SASTRA Dataset Categorizer (v3 — Fixed Rate Limiter)
=====================================================================
ROOT CAUSE FIX: Added a GLOBAL per-key TokenBucket so that parallel
workers SHARE a single 14 RPM quota per API key. Previous version had
no shared state → all workers fired simultaneously → instant 429/503.

Usage:
  python categorize_site_conditions.py --workers 2
  python categorize_site_conditions.py --workers 2 --dry-run

Setup:
  pip install google-genai pillow tqdm

Environment Variables (set before running):
  GEMINI_KEY_1=AIzaSy...
  GEMINI_KEY_2=AIzaSy...
  GEMINI_KEY_3=AIzaSy...   # optional, adds 14 more RPM
"""

import os
import sys
import time
import shutil
import logging
import argparse
import threading
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

# ── Use the NEW google-genai SDK (supports per-client API keys) ──────────────
try:
    from google import genai
except ImportError:
    print("[FATAL] google-genai not installed. Run: pip install google-genai pillow tqdm")
    sys.exit(1)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("categorize_log.txt", encoding="utf-8"),
    ],
)
log = logging.getLogger("BuildSight")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset")

SOURCE_DATASETS = [
    "PPE_SASTRA_Dataset_3",
    "PPE_SASTRA_Dataset_4",
    "Extracted_Video_Frames",
]

DEST_FOLDERS = {
    "Normal_Site_Condition": BASE_DIR / "Normal_Site_Condition",
    "Dusty_Condition":       BASE_DIR / "Dusty_Condition",
    "Low_Light_Condition":   BASE_DIR / "Low_Light_Condition",
    "Crowded_Condition":     BASE_DIR / "Crowded_Condition",
}

VALID_LABELS = set(DEST_FOLDERS.keys())

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".mpo"}

# Safe RPM per key (14 instead of 15 — 1 slot buffer to prevent edge-case bursts)
SAFE_RPM_PER_KEY = 12
MAX_RETRIES      = 5
MODEL_NAME       = "gemini-2.5-flash"   # Only model working across all 3 keys

CLASSIFICATION_PROMPT = """
You are a construction site safety expert analyzing images for the BuildSight AI system.
Classify this image into EXACTLY ONE of these four categories:

1. Normal_Site_Condition  — Clear visibility, balanced worker density, proper lighting, minimal obstruction.
2. Dusty_Condition        — Visible airborne dust, reduced image clarity, haze from construction activity.
3. Low_Light_Condition    — Poor illumination, night-time or dim environment, low visibility.
4. Crowded_Condition      — High concentration of workers, congested area, workers in close proximity.

Reply with ONLY the category name. No explanation. No punctuation. One of:
Normal_Site_Condition
Dusty_Condition
Low_Light_Condition
Crowded_Condition
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# Token Bucket (Global per-key rate limiter) — THE CORE FIX
# ══════════════════════════════════════════════════════════════════════════════
class TokenBucket:
    """
    Thread-safe token bucket that enforces RPM limits GLOBALLY across
    all workers sharing the same API key. This prevents burst collisions.
    """

    def __init__(self, rpm: int):
        self.capacity    = rpm
        self.tokens      = 0.0            # Start EMPTY — prevents initial burst
        self.refill_rate = rpm / 60.0   # tokens per second
        self.lock        = threading.Lock()
        self.last_refill = time.monotonic()

    def acquire(self):
        """Block until a token is available."""
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                self.last_refill = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return  # token granted
            # No token yet — sleep a short interval and retry
            time.sleep(0.2)


# ══════════════════════════════════════════════════════════════════════════════
# API Key Pool — round-robin with per-key rate limiting
# ══════════════════════════════════════════════════════════════════════════════
class ApiKeyPool:
    """
    Round-robin pool of Gemini clients, each with its own TokenBucket.
    Uses the NEW google-genai SDK which supports per-client API keys
    (unlike the old google-generativeai SDK which uses global config).
    """

    def __init__(self, api_keys: list):
        if not api_keys:
            raise ValueError("No API keys provided.")
        self.pool = []
        for key in api_keys:
            # Each client gets its OWN api_key — no global state collision
            client = genai.Client(api_key=key)
            bucket = TokenBucket(SAFE_RPM_PER_KEY)
            self.pool.append((client, bucket, key[-6:]))  # last 6 chars for logging
        self._idx  = 0
        self._lock = threading.Lock()
        log.info(f"[Pool] Initialized {len(self.pool)} API key(s). "
                 f"Max combined RPM: {len(self.pool) * SAFE_RPM_PER_KEY}")

    def get_slot(self):
        """Returns (client, bucket, key_hint) in round-robin."""
        with self._lock:
            slot = self.pool[self._idx % len(self.pool)]
            self._idx += 1
        return slot


# ── Image Utilities ────────────────────────────────────────────────────────────
def load_image_as_pil(path: Path) -> Image.Image:
    """Load any image (including MPO) and return as a clean RGB PIL Image."""
    with Image.open(path) as img:
        img.seek(0)                      # MPO: use first frame only
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return Image.open(buf)


# ── Classification ─────────────────────────────────────────────────────────────
def classify_image(path: Path, pool: ApiKeyPool) -> str | None:
    """
    Classify a single image. Retries with exponential backoff on rate
    limits or 503 errors. Returns the category label or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        client, bucket, key_hint = pool.get_slot()

        # ── Wait for a token from THIS key's bucket ──
        bucket.acquire()

        try:
            img_pil = load_image_as_pil(path)

            # Use the new google-genai SDK API
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[CLASSIFICATION_PROMPT, img_pil]
            )
            label = response.text.strip()

            if label in VALID_LABELS:
                return label
            else:
                # Model returned unexpected text — default to Normal
                log.warning(f"[Parse] Unexpected label '{label}' for {path.name}. "
                            f"Defaulting to Normal_Site_Condition.")
                return "Normal_Site_Condition"

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate = "429" in err_str or "quota" in err_str or "rate" in err_str
            is_503  = "503" in err_str or "unavailable" in err_str

            if (is_rate or is_503) and attempt < MAX_RETRIES:
                wait = min(4 * (2 ** (attempt - 1)), 120)  # cap at 120s
                log.warning(f"  [{'RateLimit' if is_rate else 'Server503'}] "
                            f"key=...{key_hint}  {path.name}  "
                            f"wait={wait}s  attempt={attempt}/{MAX_RETRIES}")
                time.sleep(wait)
            else:
                log.error(f"  [FAILED] {path.name}  attempt={attempt}  error={exc}")
                if attempt == MAX_RETRIES:
                    return None

    return None


# ── Worker Task ────────────────────────────────────────────────────────────────
def process_image(args: tuple) -> tuple:
    path, pool = args
    label = classify_image(path, pool)
    return path, label


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BuildSight PPE Dataset Categorizer v3")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers. Keep <= (num_keys * 2). Default: 2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify without moving files (testing mode)")
    args = parser.parse_args()

    # ── Load API keys from environment ──────────────────────────────────────
    api_keys = []
    for i in range(1, 10):
        key = os.environ.get(f"GEMINI_KEY_{i}", "").strip()
        if key:
            api_keys.append(key)

    if not api_keys:
        log.error("[FATAL] No API keys found in environment variables.")
        log.error("  Set: $env:GEMINI_KEY_1='AIza...'   $env:GEMINI_KEY_2='AIza...'")
        sys.exit(1)

    # Warn if workers exceed safe parallelism
    safe_workers = len(api_keys) * 2
    if args.workers > safe_workers:
        log.warning(f"[Config] {args.workers} workers with {len(api_keys)} key(s) "
                    f"may still hit limits. Recommended max: {safe_workers}")

    pool = ApiKeyPool(api_keys)

    # ── Ensure destination folders exist ────────────────────────────────────
    for folder in DEST_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)

    # ── Collect all images from source datasets (Filtering out already categorized) ──
    all_images: list = []
    skipped_count = 0
    
    # Pre-examine destination folders to build a set of already-categorized filenames
    categorized_filenames = set()
    for folder in DEST_FOLDERS.values():
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file():
                    categorized_filenames.add(f.name)

    for dataset in SOURCE_DATASETS:
        src = BASE_DIR / dataset
        if not src.exists():
            log.warning(f"[Skip] Dataset not found: {src}")
            continue
        
        for p in src.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                if p.name in categorized_filenames:
                    skipped_count += 1
                    continue
                all_images.append(p)

    log.info(f"[Discover] Found {len(all_images)} new images to process (Skipped {skipped_count} already categorized)")

    if not all_images:
        log.info("[Summary] All images are already categorized! Work is complete.")
        sys.exit(0)

    log.info(f"\n🚀  Processing {len(all_images)} images  |  "
             f"Workers: {args.workers}  |  "
             f"Keys: {len(api_keys)}  |  "
             f"Max RPM: {len(api_keys) * SAFE_RPM_PER_KEY}\n")

    # ── Stats ────────────────────────────────────────────────────────────────
    stats = {label: 0 for label in VALID_LABELS}
    stats["failed"] = 0
    stats_lock = threading.Lock()

    # ── Process ──────────────────────────────────────────────────────────────
    tasks = [(img, pool) for img in all_images]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, t): t[0] for t in tasks}

        with tqdm(total=len(all_images), unit="img", ncols=90,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

            for future in as_completed(futures):
                path, label = future.result()

                if label is None:
                    log.error(f"  [SKIP] {path.name} — max retries exceeded, skipping.")
                    with stats_lock:
                        stats["failed"] += 1
                else:
                    dest = DEST_FOLDERS[label] / path.name

                    # Avoid filename collision
                    if dest.exists():
                        stem   = path.stem
                        suffix = path.suffix
                        dest   = DEST_FOLDERS[label] / f"{stem}_{path.parent.name}{suffix}"

                    if not args.dry_run:
                        shutil.copy2(path, dest)
                        log.info(f"  [OK] {path.name}  →  {label}")
                    else:
                        log.info(f"  [DRY] {path.name}  →  {label}")

                    with stats_lock:
                        stats[label] += 1

                pbar.update(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "═" * 55)
    log.info("  CATEGORIZATION COMPLETE — SUMMARY")
    log.info("═" * 55)
    for label, count in stats.items():
        log.info(f"  {label:<30} {count:>5} images")
    log.info("═" * 55)

    if stats["failed"] > 0:
        log.warning(f"\n  ⚠  {stats['failed']} images failed. "
                    f"Re-run the script — they will be skipped if already copied.")


if __name__ == "__main__":
    main()
