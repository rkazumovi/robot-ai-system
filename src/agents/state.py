"""
state.py — Shared state schema for the LangGraph agentic pipeline.

The four agents (Inspector -> Predictor -> Reporter -> Optimizer) read
from and write to this single shared state as they run in sequence.
LangGraph merges each node's returned dict into the overall state.
"""

from __future__ import annotations

from typing import Optional, TypedDict

class InspectionState(TypedDict, total=False):
    # --- Input: raw CV inspection result (from Component 1) ---
    joint_id: int
    image_path: str
    defect_mask_summary: dict          # e.g. {"defect_pixel_fraction": 0.03, "num_components": 2}
    defect_confidence: float           # model's max predicted probability, 0-1
    joint_angles: list[float]          # current robot pose (radians)
    joint_velocities: list[float]      # current joint velocities (rad/s)

    # --- Agent 1 (Inspector) output ---
    defect_detected: bool
    defect_severity: str               # "none" | "minor" | "moderate" | "severe"
    inspection_summary: str

    # --- Agent 2 (Predictor) output ---
    thermal_risk: dict                 # from digital_twin.pinn: predicted temps, margin to limit
    vibration_anomaly: dict            # from digital_twin.anomaly: detection result
    failure_risk_score: float          # 0-1, combined risk estimate
    failure_risk_level: str            # "low" | "medium" | "high" | "critical"

    # --- Agent 3 (Reporter) output ---
    report_text: str

    # --- Agent 4 (Optimizer) output ---
    recommended_actions: list[dict]    # structured list of {action, priority, reason}
    parameter_adjustments: dict        # e.g. {"max_velocity_scale": 0.8, "duty_cycle_limit": 0.6}