"""
inspector.py — Agent 1 (Inspector): analyzes CV defect detection results.

Takes the raw output from Component 1's PhysicsConstrainedCNN (defect
mask summary + confidence) and produces a structured severity
classification and human-readable summary. Deterministic/rule-based --
not every node in an agentic pipeline needs to be an LLM call; using
clear rules here keeps this stage fast, reliable, and easy to test,
while the Reporter agent (later in the pipeline) handles natural-
language generation.

Severity rule (applied to defect_pixel_fraction, the fraction of the
image's pixels flagged as defective by the CV model):
    fraction == 0                    -> "none"
    0 < fraction <= 0.01 (1%)        -> "minor"
    0.01 < fraction <= 0.05 (5%)     -> "moderate"
    fraction > 0.05                  -> "severe"
Confidence below a floor downgrades severity by one level (a low-
confidence severe detection is treated as moderate, etc.) to avoid
overreacting to a shaky prediction -- this mirrors how a human inspector
would treat an uncertain reading.

Run this file directly to test the severity classification logic
against a range of synthetic CV outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from state import InspectionState  # noqa: E402

SEVERITY_LEVELS = ["none", "minor", "moderate", "severe"]
LOW_CONFIDENCE_THRESHOLD = 0.6

def classify_severity(defect_pixel_fraction: float, confidence: float) -> str:
    """Maps (defect_pixel_fraction, confidence) to a severity level."""
    if defect_pixel_fraction <= 0:
        return "none"
    elif defect_pixel_fraction <= 0.01:
        base_idx = 1  # minor
    elif defect_pixel_fraction <= 0.05:
        base_idx = 2  # moderate
    else:
        base_idx = 3  # severe

    if confidence < LOW_CONFIDENCE_THRESHOLD and base_idx > 0:
        base_idx -= 1  # downgrade one level under low confidence

    return SEVERITY_LEVELS[base_idx]


def build_summary(
    joint_id: int, severity: str, defect_pixel_fraction: float,
    num_components: int, confidence: float,
) -> str:
    if severity == "none":
        return f"Joint {joint_id}: no defects detected (confidence {confidence:.2f})."
    return (
        f"Joint {joint_id}: {severity} defect detected -- "
        f"{defect_pixel_fraction*100:.2f}% of surface area affected across "
        f"{num_components} region(s), confidence {confidence:.2f}."
    )


def run(state: InspectionState) -> dict:
    """
    LangGraph node function: reads defect_mask_summary/defect_confidence
    from state, returns the Inspector's partial state update.
    """
    mask_summary = state.get("defect_mask_summary", {})
    fraction = mask_summary.get("defect_pixel_fraction", 0.0)
    num_components = mask_summary.get("num_components", 0)
    confidence = state.get("defect_confidence", 0.0)
    joint_id = state.get("joint_id", -1)

    severity = classify_severity(fraction, confidence)
    detected = severity != "none"
    summary = build_summary(joint_id, severity, fraction, num_components, confidence)

    return {
        "defect_detected": detected,
        "defect_severity": severity,
        "inspection_summary": summary,
    }


if __name__ == "__main__":
    print("=== Inspector severity classification tests ===")
    test_cases = [
        # (fraction, confidence, expected_severity)
        (0.0, 0.99, "none"),
        (0.005, 0.95, "minor"),
        (0.03, 0.90, "moderate"),
        (0.10, 0.95, "severe"),
        (0.10, 0.40, "moderate"),   # low confidence downgrades severe -> moderate
        (0.03, 0.30, "minor"),      # low confidence downgrades moderate -> minor
        (0.005, 0.20, "none"),      # low confidence downgrades minor -> none
    ]

    all_passed = True
    for fraction, confidence, expected in test_cases:
        result = classify_severity(fraction, confidence)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  [{status}] fraction={fraction:.3f} confidence={confidence:.2f} "
              f"-> {result} (expected {expected})")

    assert all_passed, "Some severity classification tests failed."
    print("\nAll severity classification tests passed.")

    print("\n=== Full node run test ===")
    test_state: InspectionState = {
        "joint_id": 3,
        "defect_mask_summary": {"defect_pixel_fraction": 0.032, "num_components": 2},
        "defect_confidence": 0.91,
    }
    result = run(test_state)
    print(f"  defect_detected:    {result['defect_detected']}")
    print(f"  defect_severity:    {result['defect_severity']}")
    print(f"  inspection_summary: {result['inspection_summary']}")
    assert result["defect_detected"] is True
    assert result["defect_severity"] == "moderate"
    assert "Joint 3" in result["inspection_summary"]
    print("\nNode run test passed.")