"""
model.py — Physics-constrained CNN for defect segmentation.

Architecture: genuine pretrained ResNet-34 encoder (ImageNet weights via
torchvision) + U-Net-style decoder with skip connections, outputting a
full-resolution single-channel logit map consumed by
PhysicsConstrainedLoss (see losses.py).

Why pretrained, not from-scratch (design change from the first version)
-------------------------------------------------------------------------
The original version of this file used a custom 5-stage encoder trained
from random initialization, deliberately deeper than standard ResNet-34
for extra capacity. In practice, on MVTec AD's per-category splits
(~200 training images per category), that 51.7M-parameter from-scratch
encoder overfit badly: train loss went to ~0 while val loss bottomed out
by epoch 4-5 and climbed steadily afterward, even with dropout (p=0.3)
and AdamW weight decay (1e-4) added.

That's expected: a from-scratch encoder must learn low-level visual
features (edges, textures, gradients) from only ~200 images, which is
nowhere near enough. The standard, production-grade fix — and the
approach actually used in real industrial defect-detection systems at
this data scale — is transfer learning: reuse a ResNet-34 encoder
pretrained on ImageNet (1.28M images), so the network already knows
general visual features and only needs to *adapt* them to defects,
not learn them from zero.

This requires matching torchvision's exact ResNet-34 architecture (4
downsampling stages: layer1 stride=1, layer2/3/4 stride=2, following the
stem's own /4 downsampling) so the pretrained weights load correctly.
This replaces the earlier "5 custom stages, deeper than ResNet-34"
design — the depth claim was novel but incompatible with using real
pretrained weights, and generalizing correctly matters more than
architectural novelty for a production system.

Why an encoder-decoder instead of a plain classifier
------------------------------------------------------
TopologyLoss and SurfaceContinuityLoss both operate on a *spatial*
probability field, not a single class score. This forces a segmentation
architecture: predict a defect probability at every pixel, at the
original input resolution.

Why skip connections matter mathematically here
--------------------------------------------------
Downsampling discards fine spatial detail. Skip connections let the
decoder reconstruct sharp boundaries, so SurfaceContinuityLoss's
gradient penalty measures genuine boundary discontinuities in the
prediction rather than architecture-induced blur.

Freezing strategy
--------------------
By default, the stem + layer1 (the earliest, most generic
edge/texture/color features) are frozen — these transfer almost
unchanged from ImageNet to any visual domain. layer2/3/4 (the more
task-specific, higher-level features) remain trainable, since defect
patterns are visually distinct enough from ImageNet's object classes
that these deeper layers benefit from adapting. This is the standard
"partial fine-tuning" recipe for small target datasets.

Run this file directly to sanity-check output shape and gradient flow.
Requires internet access on first run to download ImageNet ResNet-34
weights (~87MB, cached afterward by torch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

class DecoderStage(nn.Module):
    """One decoder stage: upsample, concat skip connection, then conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class PhysicsConstrainedCNN(nn.Module):
    """
    Pretrained-ResNet-34-encoder U-Net for defect segmentation.

    Input:  (B, 3, H, W)  RGB image, H and W divisible by 32 recommended.
    Output: (B, 1, H, W)  raw logits (feed through sigmoid for probabilities).

    Args:
        pretrained: load ImageNet-pretrained ResNet-34 weights (recommended
            for small datasets like per-category MVTec AD splits).
        freeze_backbone: freeze the stem + layer1 (earliest, most generic
            features). layer2/3/4 remain trainable.
        dropout_p: dropout probability applied in the decoder (bottleneck
            gets the full value, tapering toward the output).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.dropout_p = dropout_p

        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet34(weights=weights)

        if in_channels != 3:
            # Pretrained conv1 expects 3 channels; replace if a different
            # input channel count is needed (rare for this project, but
            # keeps the class general-purpose).
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Stem: conv1 -> bn1 -> relu (kept separate from maxpool so we can
        # use its /2-resolution output as a skip connection for the decoder's
        # final stage, preserving fine detail near the output resolution).
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool  # /4 resolution after this

        self.layer1 = backbone.layer1  # /4,  64 channels (stride 1: no further downsample)
        self.layer2 = backbone.layer2  # /8,  128 channels
        self.layer3 = backbone.layer3  # /16, 256 channels
        self.layer4 = backbone.layer4  # /32, 512 channels (deepest / bottleneck)

        if freeze_backbone:
            for module in [self.stem, self.maxpool, self.layer1]:
                for param in module.parameters():
                    param.requires_grad = False

        self.bottleneck_dropout = nn.Dropout2d(p=dropout_p)

        # Decoder: 4 upsampling stages mirroring layer4->layer3->layer2->layer1->stem,
        # then one final upsample (undo conv1's stride-2) + head to reach full input resolution.
        # (stem output, /2, 64ch) -> (layer1 output, /4, 64ch) -> (layer2, /8, 128ch)
        # -> (layer3, /16, 256ch) -> (layer4, /32, 512ch, bottleneck)
        decoder_specs = [
            # (in_channels, skip_channels, out_channels)
            (512, 256, 256),  # from layer4 -> meet layer3 skip
            (256, 128, 128),  # -> meet layer2 skip
            (128, 64, 64),    # -> meet layer1 skip
            (64, 64, 64),     # -> meet stem (pre-maxpool) skip
        ]
        self.decoder_stages = nn.ModuleList()
        self.decoder_dropouts = nn.ModuleList()
        n_stages = len(decoder_specs)
        for i, (in_ch, skip_ch, out_ch) in enumerate(decoder_specs):
            self.decoder_stages.append(DecoderStage(in_ch, skip_ch, out_ch))
            stage_dropout_p = dropout_p * (1.0 - i / n_stages)
            self.decoder_dropouts.append(nn.Dropout2d(p=stage_dropout_p))

        # Final upsample: stem's conv1 was stride 2, so decoder output here
        # is at /2 resolution -> one more upsample by 2 reaches full input res.
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        stem_out = self.stem(x)          # /2,  64ch  -> skip for final decoder stage
        pooled = self.maxpool(stem_out)  # /4

        l1 = self.layer1(pooled)  # /4,  64ch  -> skip
        l2 = self.layer2(l1)      # /8,  128ch -> skip
        l3 = self.layer3(l2)      # /16, 256ch -> skip
        l4 = self.layer4(l3)      # /32, 512ch -> bottleneck

        feat = self.bottleneck_dropout(l4)

        skips = [l3, l2, l1, stem_out]
        for decoder_stage, dropout, skip in zip(self.decoder_stages, self.decoder_dropouts, skips):
            feat = decoder_stage(feat, skip)
            feat = dropout(feat)

        out = self.final_upsample(feat)
        if out.shape[-2:] != input_size:
            out = nn.functional.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        logits = self.head(out)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)

    model = PhysicsConstrainedCNN(in_channels=3, out_channels=1, pretrained=True, freeze_backbone=True)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,} ({100 * n_trainable / n_params:.1f}%)")

    for size in [256, 224]:
        x = torch.randn(2, 3, size, size, requires_grad=True)
        logits = model(x)
        print(f"\nInput  shape: {tuple(x.shape)}")
        print(f"Output shape: {tuple(logits.shape)}")
        assert logits.shape == (2, 1, size, size), "Output must match input spatial size, 1 channel."

        logits.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0, "Gradient did not reach the input."
        model.zero_grad()
        x.grad = None
        print("Shape check passed, gradient flows end-to-end.")

    # Confirm frozen layers actually have no gradient after a backward pass,
    # and unfrozen layers do.
    model.zero_grad()
    x = torch.randn(2, 3, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    logits = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    frozen_grad_sum = sum(
        p.grad.abs().sum().item() for p in model.stem.parameters() if p.grad is not None
    )
    trainable_grad_sum = sum(
        p.grad.abs().sum().item() for p in model.layer4.parameters() if p.grad is not None
    )
    print(f"\nFrozen stem grad sum (expect 0 or no grad): {frozen_grad_sum}")
    print(f"Trainable layer4 grad sum (expect > 0):      {trainable_grad_sum}")
    assert frozen_grad_sum == 0, "Frozen stem should not accumulate gradients."
    assert trainable_grad_sum > 0, "layer4 should be trainable and receive gradients."
    print("Freeze/unfreeze behavior verified correctly.")

    # Dropout sanity check.
    model.train()
    x_fixed = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out_a = model(x_fixed)
        out_b = model(x_fixed)
    assert not torch.allclose(out_a, out_b), "Expected dropout to vary outputs in train mode."

    model.eval()
    with torch.no_grad():
        out_c = model(x_fixed)
        out_d = model(x_fixed)
    assert torch.allclose(out_c, out_d), "Expected eval mode (dropout off) to be deterministic."
    print("Dropout train/eval behavior verified correctly.")

    print("\nAll checks passed.")