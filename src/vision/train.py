"""
train.py — Training loop for the physics-constrained defect segmentation CNN.

Ties together:
    dataset.py  -> MVTecHFDataset (real MVTec AD via HuggingFace/FiftyOne)
    model.py    -> PhysicsConstrainedCNN (ResNet-34-depth U-Net)
    losses.py   -> PhysicsConstrainedLoss (CE + topology + surface continuity)

Features:
    - Per-epoch train/val loop with running loss breakdown (CE/topo/surf)
    - Checkpointing: saves best model (by val loss) to outputs/
    - CPU-friendly defaults (small batch size, no AMP dependency)
    - Resumable: --resume <path> loads a checkpoint and continues

Run this file directly for a short smoke-test run (2 epochs, tiny
subset) to confirm the full pipeline trains without errors before
committing to a full run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import MVTecHFDataset  # noqa: E402
from losses import PhysicsConstrainedLoss  # noqa: E402
from model import PhysicsConstrainedCNN  # noqa: E402

def build_dataloaders(
    category: str, image_size: int, batch_size: int, num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    train_ds = MVTecHFDataset(category=category, split="train", image_size=image_size, augment=True)
    test_ds = MVTecHFDataset(category=category, split="test", image_size=image_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def run_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    """One pass over `loader`. If optimizer is None, runs in eval mode (no backward)."""
    model.train() if train else model.eval()

    totals = {"total": 0.0, "ce": 0.0, "topo": 0.0, "surf": 0.0}
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            losses = loss_fn(logits, masks)

            if train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train the physics-constrained defect CNN.")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (L2 regularization)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability in decoder/bottleneck")
    parser.add_argument("--lambda1", type=float, default=0.1, help="TopologyLoss weight")
    parser.add_argument("--lambda2", type=float, default=0.05, help="SurfaceContinuityLoss weight")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="2 epochs, tiny run, for pipeline validation")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2
        print("=== SMOKE TEST MODE: 2 epochs, for pipeline validation only ===")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("WARNING: --device cuda requested but CUDA not available, falling back to CPU.")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"model_{args.category}_best.pt"

    print(f"\nLoading data for category='{args.category}'...")
    train_loader, val_loader = build_dataloaders(
        args.category, args.image_size, args.batch_size
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = PhysicsConstrainedCNN(dropout_p=args.dropout).to(device)
    loss_fn = PhysicsConstrainedLoss(lambda_1=args.lambda1, lambda_2=args.lambda2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed at epoch {start_epoch}, best_val_loss so far: {best_val_loss:.6f}")

    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}, "
          f"weight_decay={args.weight_decay}, dropout={args.dropout}")
    print(f"{'Epoch':>6} | {'Train Total':>11} {'CE':>8} {'Topo':>8} {'Surf':>8} "
          f"| {'Val Total':>11} {'CE':>8} {'Topo':>8} {'Surf':>8} | Time")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        train_metrics = run_epoch(model, loss_fn, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, loss_fn, val_loader, None, device, train=False)
        elapsed = time.time() - t0

        print(
            f"{epoch:>6} | "
            f"{train_metrics['total']:>11.5f} {train_metrics['ce']:>8.5f} "
            f"{train_metrics['topo']:>8.5f} {train_metrics['surf']:>8.5f} | "
            f"{val_metrics['total']:>11.5f} {val_metrics['ce']:>8.5f} "
            f"{val_metrics['topo']:>8.5f} {val_metrics['surf']:>8.5f} | "
            f"{elapsed:>5.1f}s"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, checkpoint_path)
            print(f"         -> New best val loss {best_val_loss:.6f}, saved to {checkpoint_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()