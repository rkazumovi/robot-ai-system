"""
test_vision.py — Unit tests for Component 1: CV defect detection.

Covers losses.py, model.py, and the mask_to_summary conversion logic
from integration/pipeline.py. Deliberately does NOT retrain or run full
dataset downloads here -- those are slow, network-dependent validation
steps already covered manually (see the project's development history);
these are fast, hermetic tests of the underlying logic and architecture.
"""

import numpy as np
import pytest
import torch

from losses import PhysicsConstrainedLoss, SurfaceContinuityLoss, TopologyLoss
from model import PhysicsConstrainedCNN

class TestTopologyLoss:
    def test_scattered_mask_scores_higher_than_coherent(self):
        """Fragmented/scattered defect predictions should be penalized
        more than one coherent region, since that's the whole point of
        the topology-aware loss term."""
        torch.manual_seed(0)
        loss_fn = TopologyLoss()

        coherent = torch.zeros(1, 1, 64, 64)
        coherent[0, 0, 20:40, 20:40] = 1.0

        scattered = torch.zeros(1, 1, 64, 64)
        idx = torch.randint(0, 64, (50,)), torch.randint(0, 64, (50,))
        scattered[0, 0, idx[0], idx[1]] = 1.0

        assert loss_fn(scattered).item() > loss_fn(coherent).item()

    def test_output_is_nonnegative(self):
        loss_fn = TopologyLoss()
        x = torch.rand(2, 1, 32, 32)
        assert loss_fn(x).item() >= 0


class TestSurfaceContinuityLoss:
    def test_smooth_field_scores_lower_than_noisy_field(self):
        loss_fn = SurfaceContinuityLoss()
        smooth = torch.zeros(1, 1, 32, 32)
        torch.manual_seed(0)
        noisy = torch.rand(1, 1, 32, 32)
        assert loss_fn(smooth).item() < loss_fn(noisy).item()


class TestPhysicsConstrainedLoss:
    def test_gradients_flow_to_logits(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = (torch.rand(2, 1, 32, 32) > 0.8).float()
        loss_fn = PhysicsConstrainedLoss()

        losses = loss_fn(logits, target)
        losses["total"].backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0

    def test_returns_all_expected_components(self):
        logits = torch.randn(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        loss_fn = PhysicsConstrainedLoss()
        losses = loss_fn(logits, target)
        assert set(losses.keys()) == {"total", "ce", "topo", "surf"}


class TestPhysicsConstrainedCNN:
    @pytest.fixture(scope="class")
    @staticmethod
    def model():
        torch.manual_seed(0)
        return PhysicsConstrainedCNN(pretrained=False, freeze_backbone=False)

    def test_output_shape_matches_input_spatial_size(self, model):
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_gradient_flows_to_input(self, model):
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

    def test_dropout_active_in_train_mode_only(self, model):
        model.train()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out_a = model(x)
            out_b = model(x)
        assert not torch.allclose(out_a, out_b)

        model.eval()
        with torch.no_grad():
            out_c = model(x)
            out_d = model(x)
        assert torch.allclose(out_c, out_d)


class TestMaskToSummary:
    @pytest.fixture
    def mask_to_summary(self):
        from pipeline import mask_to_summary
        return mask_to_summary

    def test_empty_mask_has_zero_fraction_and_components(self, mask_to_summary):
        mask = np.zeros((32, 32))
        result = mask_to_summary(mask)
        assert result["defect_pixel_fraction"] == 0.0
        assert result["num_components"] == 0

    def test_single_blob_counts_as_one_component(self, mask_to_summary):
        mask = np.zeros((32, 32))
        mask[10:15, 10:15] = 1.0
        result = mask_to_summary(mask)
        assert result["num_components"] == 1
        assert result["defect_pixel_fraction"] == pytest.approx(25 / 1024)

    def test_two_separated_blobs_count_as_two_components(self, mask_to_summary):
        mask = np.zeros((32, 32))
        mask[0:3, 0:3] = 1.0
        mask[20:23, 20:23] = 1.0
        result = mask_to_summary(mask)
        assert result["num_components"] == 2