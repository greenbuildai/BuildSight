"""
BuildSight — Hybrid GPU + API Image Categorizer
================================================
Two-phase pipeline: Local CLIP first, Gemini API fallback for low-confidence.
Expected: 85% API reduction with 0.75 confidence threshold.

Phase 1: Local GPU classification (all images)
Phase 2: Gemini API verification (low-confidence only)
Phase 3: Move files based on final labels

Usage:
  $env:GEMINI_KEY_1='AIza...'
  $env:GEMINI_KEY_2='AIza...'
  python categorize_hybrid.py --batch-size 48 --confidence-threshold 0.75

  python categorize_hybrid.py --dry-run  # Test without moving files

Setup:
  pip install openai-clip ftfy regex google-genai
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

try:
    import torch
    import clip
    from tqdm import tqdm
except ImportError as e:
    print(f"[FATAL] Missing dependency: {e}")
    print("Install with: pip install openai-clip ftfy regex torch torchvision")
    sys.exit(1)

# Import local CLIP components
from categorize_local_gpu import (
    CLIPClassifier,
    SiteConditionDataset,
    load_image_as_pil,
    move_to_category,
    BASE_DIR,
    SOURCE_DATASETS,
    DEST_FOLDERS,
    IMAGE_EXTENSIONS,
    log
)

# Import Gemini API components
try:
    from categorize_site_conditions import classify_image, ApiKeyPool
except ImportError:
    log.warning("categorize_site_conditions.py not found. API fallback disabled.")
    classify_image = None
    ApiKeyPool = None


# ══════════════════════════════════════════════════════════════════════════════
# Hybrid Classification Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def hybrid_classify(
    all_images: List[Path],
    confidence_threshold: float = 0.75,
    batch_size: int = 48,
    model_name: str = "ViT-B/32",
    num_workers: int = 4,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Two-phase hybrid classification pipeline.

    Phase 1: Local CLIP classification (all images)
    Phase 2: Gemini API verification (low-confidence only)
    Phase 3: Move files based on final labels

    Args:
        all_images: List of image paths to classify
        confidence_threshold: API fallback threshold (default: 0.75)
        batch_size: GPU batch size (default: 48)
        model_name: CLIP model variant
        num_workers: DataLoader workers
        dry_run: If True, classify but don't move files

    Returns:
        Dictionary with results and statistics
    """
    start_time = time.time()

    # ── Phase 1: Local CLIP Classification ───────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("[HYBRID] Phase 1: Local GPU Classification")
    log.info(f"{'='*60}\n")

    classifier = CLIPClassifier(model_name=model_name)
    dataset = SiteConditionDataset(all_images, classifier.preprocess)

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
    low_confidence = []

    log.info(f"Device: {classifier.device.upper()} | Batch: {batch_size} | "
             f"Threshold: {confidence_threshold:.2%}")

    with tqdm(total=len(all_images), desc="Local CLIP", unit="img",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for batch_tensors, batch_paths, batch_valid in dataloader:
            # Filter valid images
            valid_indices = [i for i, v in enumerate(batch_valid) if v]

            if not valid_indices:
                pbar.update(len(batch_paths))
                continue

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

                result = {
                    'path': Path(path),
                    'category': category,
                    'confidence': conf_val,
                    'source': 'local'
                }

                # Flag low-confidence predictions for API fallback
                if conf_val < confidence_threshold:
                    low_confidence.append(result)

                results.append(result)

            pbar.update(len(batch_paths))

    phase1_time = time.time() - start_time

    log.info(f"\nPhase 1 complete: {len(results)} images classified in {phase1_time:.1f}s")
    log.info(f"High confidence (>= {confidence_threshold:.0%}): {len(results) - len(low_confidence)}")
    log.info(f"Low confidence (< {confidence_threshold:.0%}): {len(low_confidence)} "
             f"({len(low_confidence)/len(results)*100:.1f}%)")

    # ── Phase 2: API Fallback for Low-Confidence ─────────────────────────────
    api_count = 0

    if low_confidence and classify_image is not None:
        log.info(f"\n{'='*60}")
        log.info(f"[HYBRID] Phase 2: API Verification ({len(low_confidence)} images)")
        log.info(f"{'='*60}\n")

        # Initialize Gemini API
        api_keys = []
        for i in range(1, 10):
            key = os.environ.get(f"GEMINI_KEY_{i}", "").strip()
            if key:
                api_keys.append(key)

        if not api_keys:
            log.warning("[HYBRID] No API keys found in environment variables.")
            log.warning("  Set: $env:GEMINI_KEY_1='AIza...'  $env:GEMINI_KEY_2='AIza...'")
            log.warning("  Skipping API verification phase. Using local predictions only.")
        else:
            pool = ApiKeyPool(api_keys)

            with tqdm(total=len(low_confidence), desc="API Fallback", unit="img",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

                for result in low_confidence:
                    # Call Gemini API
                    gemini_label = classify_image(result['path'], pool)

                    if gemini_label:
                        # Update with API result
                        result['category'] = gemini_label
                        result['confidence'] = 1.0  # Trust API
                        result['source'] = 'api'
                        api_count += 1

                    pbar.update(1)

        phase2_time = time.time() - start_time - phase1_time
        log.info(f"\nPhase 2 complete: {api_count} API verifications in {phase2_time:.1f}s")

    elif low_confidence:
        log.warning("[HYBRID] API fallback unavailable (categorize_site_conditions.py not found)")
        log.warning("Using local predictions for all images.")

    # ── Phase 3: Move Files Based on Final Labels ────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("[HYBRID] Phase 3: Moving Files to Category Folders")
    log.info(f"{'='*60}\n")

    category_counts = {cat: 0 for cat in classifier.categories}

    if not dry_run:
        for result in tqdm(results, desc="Moving files", unit="img"):
            move_to_category(result['path'], result['category'])
            category_counts[result['category']] += 1
    else:
        log.info("DRY RUN: Skipping file moves")
        for result in results:
            category_counts[result['category']] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    log.info(f"\n{'='*60}")
    log.info("  HYBRID CATEGORIZATION COMPLETE — SUMMARY")
    log.info(f"{'='*60}")

    for category, count in category_counts.items():
        avg_conf = sum(r['confidence'] for r in results if r['category'] == category) / max(count, 1)
        log.info(f"  {category:<30} {count:>5} images (avg conf: {avg_conf:.2%})")

    log.info(f"{'='*60}")
    log.info(f"Source Breakdown:")
    log.info(f"  Local CLIP:     {len(results) - api_count:>5} images")
    log.info(f"  API Fallback:   {api_count:>5} images")

    if api_count > 0:
        api_reduction = (1 - api_count / len(results)) * 100
        log.info(f"  API Reduction:  {api_reduction:>5.1f}%")

    log.info(f"{'='*60}")
    log.info(f"Total time: {elapsed:.1f} seconds ({len(results)/elapsed:.1f} img/sec)")
    log.info(f"{'='*60}\n")

    return {
        'results': results,
        'category_counts': category_counts,
        'total_images': len(results),
        'local_count': len(results) - api_count,
        'api_count': api_count,
        'api_reduction_percent': (1 - api_count / len(results)) * 100 if len(results) > 0 else 0,
        'elapsed_seconds': elapsed
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="BuildSight Hybrid GPU + API Image Categorizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set API keys first
  $env:GEMINI_KEY_1='AIzaSy...'
  $env:GEMINI_KEY_2='AIzaSy...'

  # Run hybrid classification
  python categorize_hybrid.py --batch-size 48 --confidence-threshold 0.75

  # Dry run (test without moving files)
  python categorize_hybrid.py --dry-run

Performance:
  Expected: ~12 minutes for 1,306 images
  API reduction: ~85% with threshold=0.75
  Local: ~1,110 images (2 min), API: ~196 images (10 min)
        """
    )

    parser.add_argument("--batch-size", type=int, default=48,
                        help="GPU batch size for local CLIP (default: 48)")
    parser.add_argument("--confidence-threshold", type=float, default=0.75,
                        help="API fallback threshold (default: 0.75)")
    parser.add_argument("--model", default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"],
                        help="CLIP model variant (default: ViT-B/32)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify without moving files (testing mode)")

    args = parser.parse_args()

    # Validate confidence threshold
    if not 0.0 < args.confidence_threshold < 1.0:
        log.error(f"Invalid confidence threshold: {args.confidence_threshold}")
        log.error("Must be between 0.0 and 1.0")
        sys.exit(1)

    # ── Check CUDA availability ──────────────────────────────────────────────
    if not torch.cuda.is_available():
        log.warning("⚠️  CUDA unavailable. Running on CPU (slower).")

    # ── Ensure destination folders exist ──────────────────────────────────────
    for folder in DEST_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)

    # ── Collect all images from source datasets (Filtering out already categorized) ──
    all_images = []
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

    log.info(f"\n🚀  Processing {len(all_images)} images\n")

    # ── Process ───────────────────────────────────────────────────────────────
    stats = hybrid_classify(
        all_images=all_images,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
        model_name=args.model,
        num_workers=args.workers,
        dry_run=args.dry_run
    )

    # ── Final message ─────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("✓ Dry run complete. Re-run without --dry-run to move files.")
    else:
        log.info("✓ All images categorized and moved successfully!")

    if stats['api_count'] == 0:
        log.info("ℹ️  No API calls were made. All predictions had high confidence.")


if __name__ == "__main__":
    main()
