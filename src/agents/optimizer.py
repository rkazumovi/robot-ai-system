"""
optimizer.py — Agent 4 (Optimizer): recommends robot parameter adjustments.

Takes the accumulated findings (defect severity, thermal risk, vibration
anomaly, combined failure risk) and produces concrete, actionable
recommendations: which operational parameters to adjust (e.g. max
velocity scale, duty cycle limit) and what maintenance actions to
schedule, each with a priority and a stated reason.

Deterministic/rule-based, like the Inspector: for parameter
recommendations that could actually change how hardware operates, a
transparent, auditable rule set is more appropriate than an opaque
LLM call -- an operator needs to be able to see exactly why a specific
adjustment was recommended.

Run this file directly to test the recommendation logic against a range
of risk scenarios (low/medium/high/critical), verifying that
recommendations scale appropriately with risk level and reference
whichever findings (thermal, vibration, defect) actually triggered them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from state import InspectionState  # noqa: E402

# Baseline (100% = no restriction) operational parameters, scaled down as
# risk increases to reduce load on a joint showing signs of trouble.
PARAMETER_SCALES_BY_RISK = {
    "low":      {"max_velocity_scale": 1.0, "duty_cycle_limit": 1.0},
    "medium":   {"max_velocity_scale": 0.85, "duty_cycle_limit": 0.8},
    "high":     {"max_velocity_scale": 0.6, "duty_cycle_limit": 0.5},
    "critical": {"max_velocity_scale": 0.3, "duty_cycle_limit": 0.2},
}


def build_recommendations(state: InspectionState) -> list[dict]:
    """
    Builds a prioritized, explainable list of recommended actions based
    on which specific findings are present. Each entry states its own
    reason, so an operator can audit why each recommendation fired.
    """
    actions = []

    severity = state.get("defect_severity", "none")
    thermal = state.get("thermal_risk", {})
    vibration = state.get("vibration_anomaly", {})
    risk_level = state.get("failure_risk_level", "low")

    if severity in ("moderate", "severe"):
        actions.append({
            "action": "Schedule physical inspection",
            "priority": "high" if severity == "severe" else "medium",
            "reason": f"CV system detected a {severity} surface defect.",
        })

    margin = thermal.get("margin_c")
    if margin is not None and margin < 15:
        priority = "critical" if margin < 5 else "high"
        actions.append({
            "action": "Reduce duty cycle / add cooling downtime",
            "priority": priority,
            "reason": f"Predicted core temperature is only {margin:.1f}C below the thermal limit.",
        })

    if vibration.get("is_anomalous"):
        freqs = vibration.get("anomaly_freqs", [])
        actions.append({
            "action": "Schedule bearing/gear inspection",
            "priority": "high",
            "reason": f"Vibration analysis detected anomalous spectral content near {freqs} Hz, "
                      f"consistent with possible mechanical wear.",
        })

    if risk_level == "critical":
        actions.append({
            "action": "Consider taking joint offline pending inspection",
            "priority": "critical",
            "reason": "Combined failure risk score has reached the critical threshold.",
        })

    if not actions:
        actions.append({
            "action": "Continue normal operation",
            "priority": "low",
            "reason": "No significant defect, thermal, or vibration risk factors detected.",
        })

    # Sort by priority, most urgent first.
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    actions.sort(key=lambda a: priority_order.get(a["priority"], 99))

    return actions


def compute_parameter_adjustments(risk_level: str) -> dict:
    """Maps risk level to concrete operational parameter scale factors."""
    return dict(PARAMETER_SCALES_BY_RISK.get(risk_level, PARAMETER_SCALES_BY_RISK["low"]))


def run(state: InspectionState) -> dict:
    """LangGraph node function: produces the final recommendations and parameter adjustments."""
    risk_level = state.get("failure_risk_level", "low")
    actions = build_recommendations(state)
    adjustments = compute_parameter_adjustments(risk_level)

    return {
        "recommended_actions": actions,
        "parameter_adjustments": adjustments,
    }


if __name__ == "__main__":
    print("=== Parameter scaling monotonicity check ===")
    prev_velocity_scale = 2.0  # start above any real value
    for level in ["low", "medium", "high", "critical"]:
        adj = compute_parameter_adjustments(level)
        print(f"  {level:9s}: max_velocity_scale={adj['max_velocity_scale']:.2f}  "
              f"duty_cycle_limit={adj['duty_cycle_limit']:.2f}")
        assert adj["max_velocity_scale"] <= prev_velocity_scale, \
            "Velocity scale should not increase as risk level worsens."
        prev_velocity_scale = adj["max_velocity_scale"]
    print("Parameter scales correctly decrease monotonically with risk level.")

    print("\n=== Scenario: low risk, no issues ===")
    low_state: InspectionState = {
        "defect_severity": "none",
        "thermal_risk": {"margin_c": 50.0},
        "vibration_anomaly": {"is_anomalous": False},
        "failure_risk_level": "low",
    }
    result = run(low_state)
    for a in result["recommended_actions"]:
        print(f"  [{a['priority']}] {a['action']} -- {a['reason']}")
    print(f"  parameter_adjustments: {result['parameter_adjustments']}")
    assert result["recommended_actions"][0]["action"] == "Continue normal operation"
    assert result["parameter_adjustments"]["max_velocity_scale"] == 1.0
    print("Low-risk scenario correctly recommends normal operation, no restrictions.")

    print("\n=== Scenario: moderate defect + borderline thermal margin ===")
    medium_state: InspectionState = {
        "defect_severity": "moderate",
        "thermal_risk": {"margin_c": 11.5},
        "vibration_anomaly": {"is_anomalous": False},
        "failure_risk_level": "medium",
    }
    result = run(medium_state)
    for a in result["recommended_actions"]:
        print(f"  [{a['priority']}] {a['action']} -- {a['reason']}")
    print(f"  parameter_adjustments: {result['parameter_adjustments']}")
    actions_text = " ".join(a["action"] for a in result["recommended_actions"])
    assert "inspection" in actions_text.lower() or "duty cycle" in actions_text.lower()
    print("Medium-risk scenario correctly recommends inspection/cooling actions.")

    print("\n=== Scenario: critical -- severe defect + high vibration + low thermal margin ===")
    critical_state: InspectionState = {
        "defect_severity": "severe",
        "thermal_risk": {"margin_c": 3.0},
        "vibration_anomaly": {"is_anomalous": True, "anomaly_freqs": [37.0]},
        "failure_risk_level": "critical",
    }
    result = run(critical_state)
    for a in result["recommended_actions"]:
        print(f"  [{a['priority']}] {a['action']} -- {a['reason']}")
    print(f"  parameter_adjustments: {result['parameter_adjustments']}")
    assert result["recommended_actions"][0]["priority"] == "critical", \
        "Most urgent action should be sorted first."
    assert any("offline" in a["action"].lower() for a in result["recommended_actions"]), \
        "Critical risk should recommend taking the joint offline."
    assert result["parameter_adjustments"]["max_velocity_scale"] == 0.3
    print("Critical scenario correctly recommends most severe actions, sorted by priority.")

    print("\nAll checks passed.")