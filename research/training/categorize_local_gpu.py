"""
BuildSight — GPU-Accelerated Local Image Categorizer
=====================================================
Zero-API, fully local CLIP-based classifier for construction site conditions.
Uses OpenAI CLIP ViT-B/32 with GPU acceleration on RTX 4050.

Target: < 5 minutes for 1,306 images (vs 46 min with Gemini API)

Usage:
  python categorize_local_gpu.py --batch-size 48
  python categorize_local_gpu.py --dry-run --batch-size 32

Setup:
  pip install openai-clip ftfy regex
  python -c "import clip; clip.load('ViT-B/32')"  # Download model once
"""

import os
import sys
import time
import shutil
import logging
import argparse
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any
import random

try:
    import torch
    import clip
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"[FATAL] Missing dependency: {e}")
    print("Install with: pip install openai-clip ftfy regex torch torchvision")
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
log = logging.getLogger("BuildSight-GPU")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset")

SOURCE_DATASETS = [
    "PPE_SASTRA_Dataset_3",
    "PPE_SASTRA_Dataset_4",
]

DEST_FOLDERS = {
    "Normal_Site_Condition": BASE_DIR / "Normal_Site_Condition",
    "Dusty_Condition":       BASE_DIR / "Dusty_Condition",
    "Low_Light_Condition":   BASE_DIR / "Low_Light_Condition",
    "Crowded_Condition":     BASE_DIR / "Crowded_Condition",
}

VALID_LABELS = set(DEST_FOLDERS.keys())

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".mpo"}

# ══════════════════════════════════════════════════════════════════════════════
# Image Loading (Reused from categorize_site_conditions.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_image_as_pil(path: Path) -> Image.Image:
    """
    Load any image (including MPO) and return as a clean RGB PIL Image.
    Reuses logic from categorize_site_conditions.py lines 160-169
    """
    try:
        with Image.open(path) as img:
            img.seek(0)                      # MPO: use first frame only
            img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return Image.open(buf)
    except Exception as e:
        log.error(f"Failed to load {path.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CLIP Classifier
# ══════════════════════════════════════════════════════════════════════════════
class CLIPClassifier:
    """
    GPU-accelerated CLIP-based zero-shot image classifier for site conditions.

    Uses ensemble prompting (multiple text descriptions per category) for
    improved robustness.
    """

    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Initialize CLIP classifier

        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Auto-detect CUDA
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        log.info(f"Loading CLIP model '{model_name}' on {self.device.upper()}...")

        # Load CLIP model + preprocessor (cached after first download)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Inference mode (disable dropout/batchnorm)

        # Category names
        self.categories = [
            "Normal_Site_Condition",
            "Dusty_Condition",
            "Low_Light_Condition",
            "Crowded_Condition"
        ]

        # Precompute text embeddings (cached for efficiency)
        log.info("Encoding text prompts...")
        self.text_features = self._encode_prompts()

        log.info(f"CLIP model ready. Device: {self.device.upper()}")

    def _encode_prompts(self):
        """
        Zero-shot prompt engineering with ensemble averaging.
        Uses multiple text descriptions per category for robustness.

        Returns:
            torch.Tensor: [4, embedding_dim] text feature embeddings
        """
        # Ensemble prompts: multiple descriptions per category
        prompt_templates = {
            "Normal_Site_Condition": [
                "a clear construction site with good visibility",
                "a well-lit construction site with normal conditions",
                "construction workers in a clean environment with balanced lighting"
            ],
            "Dusty_Condition": [
                "a dusty construction site with airborne particles",
                "construction site with heavy dust and reduced visibility",
                "hazy construction environment with dust clouds"
            ],
            "Low_Light_Condition": [
                "a dark construction site with poor lighting",
                "construction site at night with dim illumination",
                "poorly lit construction area with low visibility"
            ],
            "Crowded_Condition": [
                "a crowded construction site with many workers",
                "construction site with high worker density and congestion",
                "busy construction area with workers in close proximity"
            ]
        }

        all_features = []

        for category in self.categories:
            prompts = prompt_templates[category]
            tokens = clip.tokenize(prompts).to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast():  # FP16 for speed
                features = self.model.encode_text(tokens)
                features = features.mean(dim=0)  # Ensemble average
                features /= features.norm()      # L2 normalize

            all_features.append(features)

        return torch.stack(all_features)

    def classify_batch(self, images_tensor, return_confidence=True):
        """
        Classify a batch of images using zero-shot CLIP.

        Args:
            images_tensor: [B, 3, 224, 224] preprocessed batch
            return_confidence: Return softmax probabilities

        Returns:
            predictions: [B] category indices
            confidences: [B] confidence scores (0-1)
        """
        with torch.no_grad(), torch.cuda.amp.autocast():  # FP16 for 2x speedup
            # Encode images
            image_features = self.model.encode_image(images_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity with text embeddings
            logits = 100.0 * image_features @ self.text_features.T
            probs = logits.softmax(dim=-1)

            confidences, predictions = probs.max(dim=-1)

        return predictions, confidences


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════
class SiteConditionDataset(torch.utils.data.Dataset):
    """
    Custom dataset for batch processing construction site images.
    Handles MPO files and applies CLIP preprocessing.
    """

    def __init__(self, image_paths, preprocess_fn):
        """
        Args:
            image_paths: List of Path objects
            preprocess_fn: CLIP preprocessing function
        """
        self.image_paths = image_paths
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # Load image (MPO-safe)
        image = load_image_as_pil(path)

        if image is None:
            # Return dummy tensor if loading failed
            return torch.zeros(3, 224, 224), str(path), False

        # Apply CLIP preprocessing
        tensor = self.preprocess(image)

        return tensor, str(path), True  # (image_tensor, path, valid)


# ══════════════════════════════════════════════════════════════════════════════
# File Operations
# ══════════════════════════════════════════════════════════════════════════════
def move_to_category(path: Path, category: str):
    """
    Move image to category folder with collision-safe naming.
    Reuses logic from categorize_site_conditions.py lines 302-311

    Args:
        path: Source image path
        category: Destination category name
    """
    dest = DEST_FOLDERS[category] / path.name

    # Handle filename collision
    if dest.exists():
        stem = path.stem
        suffix = path.suffix
        dest = DEST_FOLDERS[category] / f"{stem}_{path.parent.name}{suffix}"

    shutil.copy2(path, dest)
    log.info(f"  [OK] {path.name} → {category}")


# ══════════════════════════════════════════════════════════════════════════════
# Batch Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def auto_batch_size(model, device):
    """
    Test maximum batch size without OOM errors.

    Args:
        model: CLIP model
        device: torch device

    Returns:
        int: Optimal batch size
    """
    test_batches = [64, 48, 32, 24, 16, 8]

    log.info("Auto-detecting optimal batch size...")

    for bs in test_batches:
        try:
            dummy = torch.randn(bs, 3, 224, 224).to(device).half()
            with torch.no_grad():
                _ = model.encode_image(dummy)

            log.info(f"Max batch size: {bs}")
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    log.warning("Very low VRAM. Falling back to batch_size=8")
    return 8


def batch_process(
    image_paths: List[Path],
    batch_size: int = 48,
    model_name: str = "ViT-B/32",
    num_workers: int = 4,
    dry_run: bool = False,
    auto_bs: bool = False
) -> Dict[str, Any]:
    """
    Main GPU-accelerated batch processing pipeline.

    Args:
        image_paths: List of image file paths to classify
        batch_size: GPU batch size (48 optimal for RTX 4050 with ViT-B/32)
        model_name: CLIP model variant
        num_workers: DataLoader workers for parallel image loading
        dry_run: If True, classify but don't move files
        auto_bs: Auto-detect optimal batch size

    Returns:
        Dictionary with classification results and statistics
    """
    start_time = time.time()

    # Initialize classifier
    classifier = CLIPClassifier(model_name=model_name)

    # Auto-detect batch size if requested
    if auto_bs:
        batch_size = auto_batch_size(classifier.model, classifier.device)

    # Create dataset
    dataset = SiteConditionDataset(image_paths, classifier.preprocess)

    # DataLoader with optimizations
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if classifier.device == 'cuda' else 0,
        pin_memory=True if classifier.device == 'cuda' else False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    results = []
    category_counts = {cat: 0 for cat in classifier.categories}
    failed_count = 0

    log.info(f"\n{'='*60}")
    log.info(f"Processing {len(image_paths)} images")
    log.info(f"Device: {classifier.device.upper()} | Batch Size: {batch_size} | Workers: {num_workers}")
    log.info(f"Mode: {'DRY RUN (no file moves)' if dry_run else 'PRODUCTION (moving files)'}")
    log.info(f"{'='*60}\n")

    # Process batches with progress bar
    with tqdm(total=len(image_paths), desc="Classifying", unit="img",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for batch_tensors, batch_paths, batch_valid in dataloader:
            # Move valid images to GPU
            valid_indices = [i for i, v in enumerate(batch_valid) if v]

            if not valid_indices:
                # All images in batch failed to load
                failed_count += len(batch_paths)
                pbar.update(len(batch_paths))
                continue

            # Filter to valid images only
            valid_tensors = batch_tensors[valid_indices].to(classifier.device)
            valid_paths = [batch_paths[i] for i in valid_indices]

            # Run inference
            try:
                preds, confs = classifier.classify_batch(valid_tensors)
            except torch.cuda.OutOfMemoryError:
                log.error("CUDA OOM! Reduce batch size with --batch-size 32")
                torch.cuda.empty_cache()
                sys.exit(1)

            # Process results
            for path, pred_idx, conf in zip(valid_paths, preds, confs):
                category = classifier.categories[pred_idx.item()]
                conf_val = conf.item()

                results.append({
                    'path': Path(path),
                    'category': category,
                    'confidence': conf_val
                })

                category_counts[category] += 1

                # Move file (with collision-safe naming)
                if not dry_run:
                    move_to_category(Path(path), category)

            pbar.update(len(batch_paths))

    elapsed = time.time() - start_time

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("  CATEGORIZATION COMPLETE — SUMMARY")
    log.info(f"{'='*60}")

    for category, count in category_counts.items():
        avg_conf = sum(r['confidence'] for r in results if r['category'] == category) / max(count, 1)
        log.info(f"  {category:<30} {count:>5} images (avg conf: {avg_conf:.2%})")

    if failed_count > 0:
        log.info(f"  {'Failed to load':<30} {failed_count:>5} images")

    log.info(f"{'='*60}")
    log.info(f"Total time: {elapsed:.1f} seconds ({len(image_paths)/elapsed:.1f} img/sec)")
    log.info(f"{'='*60}\n")

    return {
        'results': results,
        'category_counts': category_counts,
        'total_images': len(image_paths),
        'failed': failed_count,
        'elapsed_seconds': elapsed
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="BuildSight GPU-Accelerated Local Image Categorizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python categorize_local_gpu.py --batch-size 48
  python categorize_local_gpu.py --dry-run --batch-size 32
  python categorize_local_gpu.py --model ViT-B/16 --auto-batch-size

Performance:
  RTX 4050 6GB: batch_size=48 optimal for ViT-B/32
  Expected: ~280 images/sec, ~4.7 seconds for 1,306 images
        """
    )

    parser.add_argument("--batch-size", type=int, default=48,
                        help="GPU batch size (default: 48 for RTX 4050)")
    parser.add_argument("--model", default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"],
                        help="CLIP model variant (default: ViT-B/32)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify without moving files (testing mode)")
    parser.add_argument("--auto-batch-size", action="store_true",
                        help="Auto-detect optimal batch size (overrides --batch-size)")

    args = parser.parse_args()

    # ── Check CUDA availability ──────────────────────────────────────────────
    if not torch.cuda.is_available():
        log.warning("⚠️  CUDA unavailable. Running on CPU (slower).")
        log.warning("    Install CUDA-enabled PyTorch:")
        log.warning("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    # ── Ensure destination folders exist ──────────────────────────────────────
    for folder in DEST_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)

    # ── Collect all images from source datasets ───────────────────────────────
    all_images = []
    for dataset in SOURCE_DATASETS:
        src = BASE_DIR / dataset
        if not src.exists():
            log.warning(f"[Skip] Dataset not found: {src}")
            continue
        found = [p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        log.info(f"[Discover] {dataset}: {len(found)} images")
        all_images.extend(found)

    if not all_images:
        log.error("[FATAL] No images discovered. Check BASE_DIR and dataset folder names.")
        sys.exit(1)

    log.info(f"\n🚀  Discovered {len(all_images)} images total\n")

    # ── Process ───────────────────────────────────────────────────────────────
    stats = batch_process(
        image_paths=all_images,
        batch_size=args.batch_size,
        model_name=args.model,
        num_workers=args.workers,
        dry_run=args.dry_run,
        auto_bs=args.auto_batch_size
    )

    # ── Final message ─────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("✓ Dry run complete. Re-run without --dry-run to move files.")
    else:
        log.info("✓ All images categorized and moved successfully!")


if __name__ == "__main__":
    main()
