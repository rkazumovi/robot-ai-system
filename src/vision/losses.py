"""
losses.py — Physics-constrained loss functions for defect detection.

Implements:
    L_total = CrossEntropy(y, y_hat)
            + lambda_1 * TopologyLoss(y_hat)
            + lambda_2 * SurfaceContinuityLoss(y_hat)

Mathematical background
------------------------
TopologyLoss (Betti-number proxy):
    True persistent homology tracks how connected components (Betti_0) and
    holes (Betti_1) of a shape are born/die as a filtration threshold sweeps
    over the prediction. Computing this exactly requires cubical-complex
    libraries (e.g. gudhi) that are not GPU-differentiable and are unstable
    to build on Windows.

    Instead we use a *differentiable soft-morphology proxy* for Betti_0:
    soft erosion followed by soft dilation (a "soft opening") approximates
    binary morphological opening, which removes small spurious connected
    components. We then measure how much probability mass is lost during
    this soft-opening pass. Predictions with many small disconnected blobs
    (topologically noisy / high Betti_0) lose more mass than predictions
    that are one coherent region (low Betti_0, closer to ground truth
    defect topology) -> this penalizes topological fragmentation without
    ever computing a persistence diagram explicitly.

SurfaceContinuityLoss (gradient / PDE boundary-smoothness penalty):
    L_surf = mean( ||grad(y_hat)||_2 )
    This is a total-variation-style penalty on the predicted probability
    field, directly analogous to smoothness/boundary-condition penalties
    used in PINN loss terms (penalizing large spatial derivatives at
    non-boundary points). It discourages jagged, physically-implausible
    defect boundaries.

Run this file directly to sanity-check shapes, gradients, and values on
random synthetic masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologyLoss(nn.Module):
    """Differentiable Betti-0 (connected-component) fragmentation penalty.

    Uses soft morphological opening (soft-erosion -> soft-dilation) via
    negative/positive average pooling as a smooth relaxation of min/max
    pooling. The mass lost between the raw prediction and its soft-opened
    version approximates topological noise (small disconnected blobs).
    """

    def __init__(self, kernel_size: int = 3, softness: float = 4.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.softness = softness  # higher = closer to true min/max pooling

    def _soft_erode(self, x: torch.Tensor) -> torch.Tensor:
        # Soft-min via -logsumexp(-x) trick, implemented with avg_pool on
        # exponentials for a smooth, differentiable approximation of erosion.
        neg = -x * self.softness
        pooled = F.avg_pool2d(
            torch.exp(neg), self.kernel_size, stride=1, padding=self.padding
        )
        return -torch.log(pooled + 1e-8) / self.softness

    def _soft_dilate(self, x: torch.Tensor) -> torch.Tensor:
        pos = x * self.softness
        pooled = F.avg_pool2d(
            torch.exp(pos), self.kernel_size, stride=1, padding=self.padding
        )
        return torch.log(pooled + 1e-8) / self.softness

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_hat: (B, 1, H, W) predicted defect probability map in [0, 1].
        Returns:
            scalar loss: mean absolute mass lost during soft-opening,
            i.e. how much of the prediction consists of small,
            topologically-spurious fragments.
        """
        eroded = self._soft_erode(y_hat)
        opened = self._soft_dilate(eroded)
        opened = torch.clamp(opened, 0.0, 1.0)
        fragmentation = torch.abs(y_hat - opened)
        return fragmentation.mean()


class SurfaceContinuityLoss(nn.Module):
    """Gradient-penalty smoothness loss on the predicted probability field."""

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_hat: (B, 1, H, W) predicted defect probability map.
        Returns:
            scalar loss: mean L2 norm of the spatial gradient.
        """
        dy = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]
        dx = y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]
        # Crop to common shape so dx, dy can be combined elementwise.
        dy = dy[:, :, :, :-1]
        dx = dx[:, :, :-1, :]
        grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
        return grad_mag.mean()


class PhysicsConstrainedLoss(nn.Module):
    """Total loss: CrossEntropy + lambda_1 * TopologyLoss + lambda_2 * SurfaceContinuityLoss."""

    def __init__(self, lambda_1: float = 0.1, lambda_2: float = 0.05):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.ce = nn.BCEWithLogitsLoss()
        self.topo = TopologyLoss()
        self.surf = SurfaceContinuityLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, 1, H, W) raw model output (pre-sigmoid).
            target: (B, 1, H, W) binary ground-truth defect mask.
        Returns:
            dict with 'total', 'ce', 'topo', 'surf' scalar losses.
        """
        ce_loss = self.ce(logits, target)
        probs = torch.sigmoid(logits)
        topo_loss = self.topo(probs)
        surf_loss = self.surf(probs)
        total = ce_loss + self.lambda_1 * topo_loss + self.lambda_2 * surf_loss
        return {"total": total, "ce": ce_loss, "topo": topo_loss, "surf": surf_loss}


if __name__ == "__main__":
    torch.manual_seed(0)

    B, H, W = 4, 64, 64
    logits = torch.randn(B, 1, H, W, requires_grad=True)
    target = (torch.rand(B, 1, H, W) > 0.8).float()

    loss_fn = PhysicsConstrainedLoss(lambda_1=0.1, lambda_2=0.05)
    losses = loss_fn(logits, target)

    print("=== PhysicsConstrainedLoss sanity check ===")
    for k, v in losses.items():
        print(f"{k:>6}: {v.item():.6f}")

    losses["total"].backward()
    grad_norm = logits.grad.norm().item()
    print(f"grad norm on logits: {grad_norm:.6f}")
    assert grad_norm > 0, "Gradient did not flow — check the graph."

    # Sanity: a single coherent blob should have lower TopologyLoss than
    # many scattered single-pixel blobs.
    coherent = torch.zeros(1, 1, H, W)
    coherent[0, 0, 20:40, 20:40] = 1.0  # one solid square

    scattered = torch.zeros(1, 1, H, W)
    idx = torch.randint(0, H, (50,)), torch.randint(0, W, (50,))
    scattered[0, 0, idx[0], idx[1]] = 1.0  # 50 isolated pixels

    topo = TopologyLoss()
    coherent_score = topo(coherent).item()
    scattered_score = topo(scattered).item()
    print(f"\nTopologyLoss(coherent blob)   = {coherent_score:.6f}")
    print(f"TopologyLoss(scattered pixels) = {scattered_score:.6f}")
    assert scattered_score > coherent_score, (
        "Expected scattered/fragmented mask to score higher topological "
        "noise than one coherent region."
    )
    print("\nAll checks passed.")