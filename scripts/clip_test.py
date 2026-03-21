"""
BuildSight — CLIP Validation Script
====================================
Tests CLIP accuracy on existing categorized images before full deployment.
Samples N images from each category folder and measures classification accuracy.

Usage:
  python clip_test.py                    # Default: 50 samples per category
  python clip_test.py --sample-size 100  # Test 100 images per category
  python clip_test.py --model ViT-B/16   # Test different CLIP variant

Expected Accuracy:
  Target: > 80% overall (safe for fully local mode)
  Acceptable: 70-80% (recommend hybrid mode)
  Warning: < 70% (use hybrid mode or API only)
"""

import sys
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

try:
    import torch
    from tqdm import tqdm
except ImportError as e:
    print(f"[FATAL] Missing dependency: {e}")
    print("Install with: pip install openai-clip ftfy regex torch torchvision")
    sys.exit(1)

# Import local CLIP components
from categorize_local_gpu import (
    CLIPClassifier,
    load_image_as_pil,
    BASE_DIR,
    log
)


# ══════════════════════════════════════════════════════════════════════════════
# Validation Functions
# ══════════════════════════════════════════════════════════════════════════════
def validate_accuracy(
    sample_size: int = 50,
    model_name: str = "ViT-B/32",
    verbose: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Test CLIP accuracy on already-categorized images.

    Args:
        sample_size: Number of images to sample per category
        model_name: CLIP model variant to test
        verbose: Print detailed mismatch information

    Returns:
        overall_accuracy: Overall accuracy (0-1)
        per_category: Dict of per-category accuracy
    """
    categories = [
        "Normal_Site_Condition",
        "Dusty_Condition",
        "Low_Light_Condition",
        "Crowded_Condition"
    ]

    log.info(f"\n{'='*70}")
    log.info(f"  CLIP Validation Test — Model: {model_name}")
    log.info(f"{'='*70}\n")

    # Initialize classifier
    classifier = CLIPClassifier(model_name=model_name)

    total_correct = 0
    total_tested = 0
    per_category = {}
    mismatches = []

    for true_category in categories:
        folder = BASE_DIR / true_category

        if not folder.exists():
            log.warning(f"Folder not found: {folder}")
            continue

        # Sample random images
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))

        if not images:
            log.warning(f"No images found in {folder}")
            continue

        sample = random.sample(images, min(sample_size, len(images)))

        correct = 0
        category_mismatches = []

        for img_path in tqdm(sample, desc=f"Testing {true_category:<25}", leave=False, unit="img"):
            # Load image
            image = load_image_as_pil(img_path)

            if image is None:
                log.warning(f"Failed to load: {img_path.name}")
                continue

            # Classify single image
            img_tensor = classifier.preprocess(image)
            img_tensor = img_tensor.unsqueeze(0).to(classifier.device)

            try:
                preds, confs = classifier.classify_batch(img_tensor)
                pred_category = classifier.categories[preds[0].item()]
                conf_val = confs[0].item()
            except Exception as e:
                log.error(f"Inference failed for {img_path.name}: {e}")
                continue

            # Check correctness
            if pred_category == true_category:
                correct += 1
                total_correct += 1
            else:
                mismatch = {
                    'file': img_path.name,
                    'true': true_category,
                    'pred': pred_category,
                    'conf': conf_val
                }
                category_mismatches.append(mismatch)

            total_tested += 1

        cat_accuracy = correct / len(sample) if sample else 0
        per_category[true_category] = cat_accuracy

        # Print category results
        symbol = "✓" if cat_accuracy >= 0.80 else ("⚠" if cat_accuracy >= 0.70 else "✗")
        log.info(f"  {symbol} {true_category:<30} {cat_accuracy:>6.1%}  ({correct}/{len(sample)})")

        # Store mismatches
        if category_mismatches:
            mismatches.extend(category_mismatches)

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_tested if total_tested > 0 else 0

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info(f"  Overall Accuracy: {overall_accuracy:.1%} ({total_correct}/{total_tested})")
    log.info(f"{'='*70}\n")

    # Recommendations
    if overall_accuracy < 0.70:
        log.warning("⚠️  WARNING: Accuracy < 70%")
        log.warning("    Recommend using HYBRID mode or API-only.")
        log.warning("    Run: python categorize_hybrid.py --confidence-threshold 0.75")
    elif overall_accuracy < 0.80:
        log.info("✓ GOOD: Accuracy 70-80%")
        log.info("  Local mode viable, but hybrid mode recommended for higher accuracy.")
        log.info("  Run: python categorize_hybrid.py --confidence-threshold 0.75")
    else:
        log.info("✅ EXCELLENT: Accuracy > 80%")
        log.info("   Safe for fully local mode!")
        log.info("   Run: python categorize_local_gpu.py --batch-size 48")

    # Print detailed mismatches if verbose
    if verbose and mismatches and overall_accuracy < 0.85:
        log.info(f"\n{'='*70}")
        log.info(f"  Detailed Mismatches ({len(mismatches)} total)")
        log.info(f"{'='*70}\n")

        # Group by true -> pred pattern
        patterns = {}
        for m in mismatches:
            key = (m['true'], m['pred'])
            if key not in patterns:
                patterns[key] = []
            patterns[key].append(m)

        # Sort by frequency
        for (true_cat, pred_cat), items in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True):
            log.info(f"  {true_cat} → {pred_cat}: {len(items)} mismatches")
            if len(items) <= 5:
                for item in items:
                    log.info(f"    - {item['file']} (conf: {item['conf']:.2%})")

    return overall_accuracy, per_category


# ══════════════════════════════════════════════════════════════════════════════
# Confidence Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════════
def analyze_confidence_distribution(
    sample_size: int = 100,
    model_name: str = "ViT-B/32"
):
    """
    Analyze confidence score distribution to optimize hybrid threshold.

    Args:
        sample_size: Number of images to sample per category
        model_name: CLIP model variant
    """
    categories = [
        "Normal_Site_Condition",
        "Dusty_Condition",
        "Low_Light_Condition",
        "Crowded_Condition"
    ]

    log.info(f"\n{'='*70}")
    log.info("  Confidence Distribution Analysis")
    log.info(f"{'='*70}\n")

    classifier = CLIPClassifier(model_name=model_name)

    all_confidences = []

    for true_category in categories:
        folder = BASE_DIR / true_category

        if not folder.exists():
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
        sample = random.sample(images, min(sample_size, len(images)))

        for img_path in tqdm(sample, desc=f"Analyzing {true_category}", leave=False):
            image = load_image_as_pil(img_path)
            if image is None:
                continue

            img_tensor = classifier.preprocess(image).unsqueeze(0).to(classifier.device)

            try:
                preds, confs = classifier.classify_batch(img_tensor)
                conf_val = confs[0].item()
                all_confidences.append(conf_val)
            except Exception:
                continue

    if not all_confidences:
        log.error("No confidences collected. Check dataset folders.")
        return

    # Statistics
    import numpy as np
    confidences = np.array(all_confidences)

    log.info(f"Confidence Statistics (n={len(confidences)}):")
    log.info(f"  Mean:   {confidences.mean():.3f}")
    log.info(f"  Median: {np.median(confidences):.3f}")
    log.info(f"  Std:    {confidences.std():.3f}")
    log.info(f"  Min:    {confidences.min():.3f}")
    log.info(f"  Max:    {confidences.max():.3f}")

    # Percentile thresholds
    log.info(f"\nPercentile Distribution:")
    for p in [10, 25, 50, 75, 90, 95]:
        val = np.percentile(confidences, p)
        log.info(f"  {p:2d}th percentile: {val:.3f}")

    # Hybrid mode estimates
    log.info(f"\nHybrid Mode API Usage Estimates:")
    for threshold in [0.65, 0.70, 0.75, 0.80, 0.85]:
        below = (confidences < threshold).sum()
        percent = below / len(confidences) * 100
        log.info(f"  Threshold {threshold:.2f}: {below:>4d} API calls ({percent:>5.1f}%)")

    log.info(f"\n{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="BuildSight CLIP Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clip_test.py                     # Quick test (50 per category)
  python clip_test.py --sample-size 100   # More thorough test
  python clip_test.py --model ViT-B/16    # Test larger model
  python clip_test.py --analyze           # Confidence distribution analysis

Interpretation:
  > 80%: Safe for fully local mode
  70-80%: Local mode viable, hybrid recommended
  < 70%: Use hybrid mode or API-only
        """
    )

    parser.add_argument("--sample-size", type=int, default=50,
                        help="Images per category (default: 50)")
    parser.add_argument("--model", default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"],
                        help="CLIP model variant (default: ViT-B/32)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed mismatch information")
    parser.add_argument("--analyze", action="store_true",
                        help="Run confidence distribution analysis")

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        log.warning("⚠️  CUDA unavailable. Running on CPU (slower).")

    # Run validation
    accuracy, per_cat = validate_accuracy(
        sample_size=args.sample_size,
        model_name=args.model,
        verbose=args.verbose
    )

    # Run confidence analysis if requested
    if args.analyze:
        analyze_confidence_distribution(
            sample_size=min(args.sample_size, 100),
            model_name=args.model
        )

    # Exit code based on accuracy
    if accuracy < 0.70:
        sys.exit(1)  # Failed validation
    else:
        sys.exit(0)  # Passed validation


if __name__ == "__main__":
    main()
