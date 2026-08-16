"""
evaluate.py — Evaluation and benchmark comparison for the physics-constrained
defect segmentation model.

Computes:
    Pixel-level:  accuracy, precision, recall, F1, AUROC
    Image-level:  AUROC (anomaly score = max predicted probability in the image)

Compares against published MVTec AD benchmark numbers for the "bottle"
category (PatchCore, CVPR 2022 -- Roth et al., "Towards Total Recall in
Industrial Anomaly Detection"): 99.4% image AUROC, 98.6% pixel AUROC.

IMPORTANT CAVEAT (read before trusting any "beats the benchmark" claim)
--------------------------------------------------------------------------
Published MVTec AD benchmarks (PatchCore, PaDiM, etc.) evaluate on the
OFFICIAL test split: unsupervised methods trained on 100% normal images,
evaluated on the full original test set (20 good + 63 defective for
bottle). Our training pipeline uses a custom re-split (see dataset.py's
MVTecHFDataset docstring) that moves 60% of labeled defects INTO
training, because our model does *supervised* mask-based segmentation,
not unsupervised anomaly detection -- it needs positive examples to
learn from. That means:
  1) Our test set here is smaller and different (only the held-out 40%
     of defects + normal test images), not the full official test set.
  2) Our model has seen real defect examples during training; published
     unsupervised methods have not.
These numbers are therefore NOT a fair apples-to-apples comparison to
the literature -- they're printed for context/orientation only. This
script prints that caveat directly in its output, not just here.

Run this file directly to evaluate the best saved checkpoint for a
given category against its held-out test set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import MVTecHFDataset  # noqa: E402
from model import PhysicsConstrainedCNN  # noqa: E402

# Published PatchCore (CVPR 2022) results for MVTec AD "bottle" category.
# Source: Roth et al., "Towards Total Recall in Industrial Anomaly Detection".
PUBLISHED_BENCHMARK = {
    "bottle": {"image_auroc": 0.994, "pixel_auroc": 0.986, "method": "PatchCore (CVPR 2022)"},
}


@torch.no_grad()
def run_inference(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Runs the model over `loader`, collecting flattened pixel probabilities/
    labels and per-image max-probability anomaly scores/labels."""
    model.eval()

    all_pixel_probs = []
    all_pixel_labels = []
    image_scores = []
    image_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)  # (B, 1, H, W)

        logits = model(images)
        probs = torch.sigmoid(logits)  # (B, 1, H, W)

        # Per-image anomaly score: max predicted defect probability anywhere
        # in the image (standard convention for image-level AUROC on MVTec).
        batch_image_scores = probs.amax(dim=(1, 2, 3)).cpu().numpy()
        batch_image_labels = (masks.sum(dim=(1, 2, 3)) > 0).float().cpu().numpy()

        image_scores.append(batch_image_scores)
        image_labels.append(batch_image_labels)

        all_pixel_probs.append(probs.cpu().numpy().reshape(-1))
        all_pixel_labels.append(masks.cpu().numpy().reshape(-1))

    return {
        "pixel_probs": np.concatenate(all_pixel_probs),
        "pixel_labels": np.concatenate(all_pixel_labels),
        "image_scores": np.concatenate(image_scores),
        "image_labels": np.concatenate(image_labels),
    }


def compute_metrics(results: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, float]:
    pixel_preds = (results["pixel_probs"] >= threshold).astype(int)
    pixel_labels = results["pixel_labels"].astype(int)

    metrics = {
        "pixel_accuracy": accuracy_score(pixel_labels, pixel_preds),
        "pixel_precision": precision_score(pixel_labels, pixel_preds, zero_division=0),
        "pixel_recall": recall_score(pixel_labels, pixel_preds, zero_division=0),
        "pixel_f1": f1_score(pixel_labels, pixel_preds, zero_division=0),
    }

    # AUROC needs both classes present; guard against a degenerate all-one-class batch.
    if len(np.unique(pixel_labels)) > 1:
        metrics["pixel_auroc"] = roc_auc_score(pixel_labels, results["pixel_probs"])
    else:
        metrics["pixel_auroc"] = float("nan")

    if len(np.unique(results["image_labels"])) > 1:
        metrics["image_auroc"] = roc_auc_score(results["image_labels"], results["image_scores"])
    else:
        metrics["image_auroc"] = float("nan")

    n_images = len(results["image_labels"])
    n_defective = int(results["image_labels"].sum())
    metrics["n_images"] = n_images
    metrics["n_defective_images"] = n_defective
    metrics["n_normal_images"] = n_images - n_defective
    metrics["n_pixels"] = len(pixel_labels)
    metrics["n_defect_pixels"] = int(pixel_labels.sum())

    return metrics


def print_report(category: str, metrics: dict[str, float]) -> None:
    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT — category: {category}")
    print(f"{'=' * 60}")

    print(f"\nTest set composition:")
    print(f"  Images:  {metrics['n_images']} total "
          f"({metrics['n_normal_images']} normal, {metrics['n_defective_images']} defective)")
    print(f"  Pixels:  {metrics['n_pixels']:,} total, {metrics['n_defect_pixels']:,} labeled defect")

    print(f"\nPixel-level metrics (threshold=0.5):")
    print(f"  Accuracy:  {metrics['pixel_accuracy']:.4f}")
    print(f"  Precision: {metrics['pixel_precision']:.4f}")
    print(f"  Recall:    {metrics['pixel_recall']:.4f}")
    print(f"  F1:        {metrics['pixel_f1']:.4f}")
    print(f"  AUROC:     {metrics['pixel_auroc']:.4f}")

    print(f"\nImage-level metrics:")
    print(f"  AUROC:     {metrics['image_auroc']:.4f}")

    if category in PUBLISHED_BENCHMARK:
        ref = PUBLISHED_BENCHMARK[category]
        print(f"\n{'-' * 60}")
        print(f"Reference (published, NOT directly comparable — see caveat below):")
        print(f"  {ref['method']}: image AUROC {ref['image_auroc']:.3f}, "
              f"pixel AUROC {ref['pixel_auroc']:.3f}")
        print(f"\n  *** CAVEAT ***")
        print(f"  This model was trained with real defect examples in its training")
        print(f"  set (custom re-split), and evaluated on a smaller, different held-out")
        print(f"  set than MVTec's official test split. Published unsupervised methods")
        print(f"  (PatchCore, PaDiM, etc.) train ONLY on normal images and evaluate on")
        print(f"  the FULL official test set. These numbers are for orientation only,")
        print(f"  not a rigorous apples-to-apples benchmark comparison.")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the physics-constrained defect CNN.")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to checkpoint (default: outputs/model_<category>_best.pt)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or f"outputs/model_{args.category}_best.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train.py first, "
            f"or pass --checkpoint explicitly."
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = PhysicsConstrainedCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint from epoch {ckpt['epoch']}, best_val_loss={ckpt['best_val_loss']:.6f}")

    print(f"\nLoading test set for category='{args.category}'...")
    test_ds = MVTecHFDataset(
        category=args.category, split="test", image_size=args.image_size, augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test batches: {len(test_loader)} ({len(test_ds)} images)")

    print("\nRunning inference...")
    results = run_inference(model, test_loader, device)

    metrics = compute_metrics(results, threshold=args.threshold)
    print_report(args.category, metrics)


if __name__ == "__main__":
    main()