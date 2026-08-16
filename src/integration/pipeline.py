"""
pipeline.py — Full system integration: Component 1 (CV) -> Component 3 (agents).

This is the top-level entry point tying the whole project together:

    Robot image + joint sensor data
        -> Component 1: PhysicsConstrainedCNN defect segmentation
        -> convert raw mask to defect_mask_summary (pixel fraction, component count)
        -> Component 3: 4-agent LangGraph pipeline
             (Inspector -> Predictor [queries Component 2 digital twin] -> Reporter -> Optimizer)
        -> final structured InspectionState: severity, risk, report, recommendations

Component 2 (the digital twin) is already wired in at the agent level --
predictor.py directly calls digital_twin.pinn and digital_twin.anomaly --
so this file's job is specifically closing the loop from raw CV model
output into the agent pipeline's expected input format.

Run this file directly to test the CV-output-to-agent-input conversion
logic (testable without a trained checkpoint, using a synthetic mask),
and to run the full pipeline end-to-end if a trained checkpoint is
available at outputs/model_bottle_best.pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vision"))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

from model import PhysicsConstrainedCNN  # noqa: E402
from orchestrator import build_graph  # noqa: E402
from state import InspectionState  # noqa: E402


def mask_to_summary(mask: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Converts a raw predicted probability mask (H, W) into the structured
    summary format Agent 1 (Inspector) expects: defect_pixel_fraction and
    num_components (connected regions), via connected-component labeling.
    """
    binary_mask = (mask >= threshold).astype(np.int32)
    labeled, num_components = ndimage.label(binary_mask)
    defect_pixel_fraction = float(binary_mask.sum()) / binary_mask.size

    return {
        "defect_pixel_fraction": defect_pixel_fraction,
        "num_components": int(num_components),
    }


def run_cv_inspection(
    model: torch.nn.Module, image: torch.Tensor, device: torch.device
) -> dict:
    """
    Runs Component 1's CV model on a single image, returns the CV-side
    fields needed to build the agent pipeline's initial state.

    Args:
        model: trained PhysicsConstrainedCNN.
        image: (3, H, W) preprocessed image tensor (already normalized,
            matching dataset.py's MVTecHFDataset transform).
        device: torch device to run inference on.

    Returns:
        dict with 'defect_mask_summary' and 'defect_confidence'.
    """
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    summary = mask_to_summary(probs)
    confidence = float(probs.max())

    return {"defect_mask_summary": summary, "defect_confidence": confidence}


def run_full_pipeline(
    image: torch.Tensor,
    joint_id: int,
    joint_angles: list[float],
    joint_velocities: list[float],
    model: torch.nn.Module,
    device: torch.device,
) -> InspectionState:
    """
    Full end-to-end pipeline: CV inference -> 4-agent orchestration.

    Args:
        image: (3, H, W) preprocessed robot joint image.
        joint_id: which joint this inspection is for.
        joint_angles: current joint angles (radians), e.g. from kinematics.py.
        joint_velocities: current joint velocities (rad/s).
        model: trained PhysicsConstrainedCNN (Component 1).
        device: torch device.

    Returns:
        Final InspectionState after all 4 agents have run.
    """
    cv_result = run_cv_inspection(model, image, device)

    initial_state: InspectionState = {
        "joint_id": joint_id,
        "joint_angles": joint_angles,
        "joint_velocities": joint_velocities,
        "defect_mask_summary": cv_result["defect_mask_summary"],
        "defect_confidence": cv_result["defect_confidence"],
    }

    app = build_graph()
    final_state = app.invoke(initial_state)
    return final_state


def load_checkpoint_or_fresh_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, bool]:
    """Loads a trained checkpoint if available, else returns a freshly-
    initialized model (for testing plumbing without requiring training
    to have been run first). Returns (model, loaded_from_checkpoint)."""
    model = PhysicsConstrainedCNN(pretrained=False)  # pretrained=False: fast to construct for testing
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model, True
    model.to(device)
    return model, False


if __name__ == "__main__":
    print("=== mask_to_summary conversion tests ===")

    # All-background mask: no defect.
    empty_mask = np.zeros((64, 64))
    summary = mask_to_summary(empty_mask)
    print(f"  Empty mask:    fraction={summary['defect_pixel_fraction']:.4f}  "
          f"components={summary['num_components']}")
    assert summary["defect_pixel_fraction"] == 0.0
    assert summary["num_components"] == 0

    # Single solid blob: one component.
    single_blob = np.zeros((64, 64))
    single_blob[20:30, 20:30] = 1.0
    summary = mask_to_summary(single_blob)
    print(f"  Single blob:   fraction={summary['defect_pixel_fraction']:.4f}  "
          f"components={summary['num_components']}")
    assert summary["num_components"] == 1
    assert abs(summary["defect_pixel_fraction"] - (100 / 4096)) < 1e-6

    # Two separate blobs: two components.
    two_blobs = np.zeros((64, 64))
    two_blobs[5:10, 5:10] = 1.0
    two_blobs[50:55, 50:55] = 1.0
    summary = mask_to_summary(two_blobs)
    print(f"  Two blobs:     fraction={summary['defect_pixel_fraction']:.4f}  "
          f"components={summary['num_components']}")
    assert summary["num_components"] == 2

    print("\nAll mask_to_summary tests passed.")

    print("\n=== Full pipeline test ===")
    device = torch.device("cpu")
    checkpoint_path = PROJECT_ROOT.parent / "outputs" / "model_bottle_best.pt"
    model, loaded = load_checkpoint_or_fresh_model(checkpoint_path, device)

    if loaded:
        print(f"Loaded trained checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path} -- using freshly-initialized "
              f"model to test pipeline PLUMBING only (predictions will be meaningless, "
              f"which is fine, this just verifies the integration wiring end-to-end).")

    torch.manual_seed(0)
    test_image = torch.randn(3, 256, 256)

    final_state = run_full_pipeline(
        image=test_image,
        joint_id=7,
        joint_angles=[0.1, 0.2, -0.3, 0.0, 0.4, 0.0],
        joint_velocities=[1.5],
        model=model,
        device=device,
    )

    print(f"\nJoint ID: {final_state.get('joint_id')}")
    print(f"Inspection summary: {final_state.get('inspection_summary')}")
    print(f"Failure risk: {final_state.get('failure_risk_level')} "
          f"({final_state.get('failure_risk_score'):.3f})")
    print(f"Report: {final_state.get('report_text')}")
    print(f"Recommended actions: {len(final_state.get('recommended_actions', []))} action(s)")

    required_fields = [
        "defect_detected", "defect_severity", "inspection_summary",
        "thermal_risk", "vibration_anomaly", "failure_risk_score", "failure_risk_level",
        "report_text", "recommended_actions", "parameter_adjustments",
    ]
    missing = [f for f in required_fields if f not in final_state]
    assert not missing, f"Pipeline output missing fields: {missing}"
    print("\nFull pipeline integration test passed: CV output correctly "
          "flows through all 4 agents to a complete final state.")