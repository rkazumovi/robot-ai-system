"""
predictor.py — Agent 2 (Predictor): queries the digital twin for failure prediction.

Combines two signals from Component 2 (digital twin):
  1) Thermal risk: how close the joint's predicted steady-state
     temperature is to a safety limit (via digital_twin.pinn's
     ThermalParams -- reusing the analytical steady-state formula
     directly, since re-running full PINN training per inspection call
     would be far too slow for a real-time pipeline; the trained
     network is for offline/simulation use, the analytical formula for
     fast online risk scoring).
  2) Vibration risk: whether the joint's current vibration signature
     shows spectral anomalies vs. a healthy baseline (via
     digital_twin.anomaly).

Combines both with defect severity (from Agent 1) into a single
failure_risk_score using a simple, explainable weighted rule -- not a
black-box model, since a maintenance-relevant risk score needs to be
justifiable to a human operator.

Run this file directly to test risk scoring against synthetic inputs
spanning low/medium/high/critical risk scenarios.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "digital_twin"))
from state import InspectionState  # noqa: E402
from pinn import ThermalParams  # noqa: E402
from anomaly import detect_spectral_anomalies, simulate_vibration_signal  # noqa: E402

SEVERITY_TO_SCORE = {"none": 0.0, "minor": 0.25, "moderate": 0.55, "severe": 0.9}

# Safety margin: joint temperature should stay this many degrees below a
# hard thermal limit before being flagged as elevated risk.
THERMAL_LIMIT_C = 80.0
THERMAL_WARNING_MARGIN_C = 15.0

RISK_WEIGHTS = {"defect": 0.4, "thermal": 0.35, "vibration": 0.25}


def assess_thermal_risk(joint_velocity: float, params: ThermalParams | None = None) -> dict:
    """
    Fast analytical thermal risk check: scales the heat-generation term
    q with joint activity level (velocity, as a simple proxy for
    friction/motor loading -- more motion, more heat), then evaluates
    the closed-form steady-state core temperature (see digital_twin/pinn.py)
    against the safety limit. This intentionally reuses the analytical
    formula (not a full PINN forward pass) since this needs to run fast,
    per-inspection-call, in a live pipeline.
    """
    params = params or ThermalParams()
    # Scale heat generation with joint activity (velocity), floor at some
    # baseline idle heat generation.
    activity_scale = max(abs(joint_velocity), 0.1)
    scaled_params = ThermalParams(
        k=params.k, rho=params.rho, c_p=params.c_p, h=params.h,
        q=params.q * activity_scale, L=params.L, T_amb=params.T_amb,
    )
    core_temp = scaled_params.steady_state(np.array([0.0]))[0]
    margin = THERMAL_LIMIT_C - core_temp

    if margin <= 0:
        risk = 1.0
    elif margin < THERMAL_WARNING_MARGIN_C:
        risk = 1.0 - (margin / THERMAL_WARNING_MARGIN_C)
    else:
        risk = 0.0

    return {
        "predicted_core_temp_c": float(core_temp),
        "thermal_limit_c": THERMAL_LIMIT_C,
        "margin_c": float(margin),
        "thermal_risk_score": float(np.clip(risk, 0.0, 1.0)),
    }


def assess_vibration_risk(
    healthy_baseline: np.ndarray | None, test_signal: np.ndarray, dt: float
) -> dict:
    """Wraps digital_twin.anomaly's spectral anomaly detection into a risk score."""
    if healthy_baseline is None:
        return {"is_anomalous": False, "anomaly_freqs": [], "vibration_risk_score": 0.0}

    result = detect_spectral_anomalies(healthy_baseline, test_signal, dt)
    n_flagged = int(result["anomaly_mask"].sum())
    # Simple severity proxy: more flagged bands -> higher risk, capped at 1.0.
    risk = min(n_flagged / 5.0, 1.0) if result["is_anomalous"] else 0.0

    return {
        "is_anomalous": result["is_anomalous"],
        "anomaly_freqs": result["anomaly_freqs"].tolist(),
        "vibration_risk_score": float(risk),
    }


def combine_risk(defect_severity: str, thermal_risk: float, vibration_risk: float) -> tuple[float, str]:
    defect_score = SEVERITY_TO_SCORE.get(defect_severity, 0.0)
    combined = (
        RISK_WEIGHTS["defect"] * defect_score
        + RISK_WEIGHTS["thermal"] * thermal_risk
        + RISK_WEIGHTS["vibration"] * vibration_risk
    )
    combined = float(np.clip(combined, 0.0, 1.0))

    if combined < 0.25:
        level = "low"
    elif combined < 0.5:
        level = "medium"
    elif combined < 0.75:
        level = "high"
    else:
        level = "critical"

    return combined, level


def run(state: InspectionState) -> dict:
    """LangGraph node function: queries the digital twin, returns risk assessment."""
    joint_velocities = state.get("joint_velocities", [0.0])
    joint_velocity = joint_velocities[0] if joint_velocities else 0.0

    thermal = assess_thermal_risk(joint_velocity)

    # Vibration check: in a real deployment this would come from a live
    # sensor stream; here we simulate a representative signal for the
    # demo/test pipeline via digital_twin.anomaly's synthetic generator.
    dt = 0.001
    healthy = simulate_vibration_signal(drive_freq_hz=8.0, duration=2.0, dt=dt)["theta_dot"]
    test = simulate_vibration_signal(drive_freq_hz=8.0, duration=2.0, dt=dt)["theta_dot"]
    vibration = assess_vibration_risk(healthy, test, dt)

    defect_severity = state.get("defect_severity", "none")
    combined_score, risk_level = combine_risk(
        defect_severity, thermal["thermal_risk_score"], vibration["vibration_risk_score"]
    )

    return {
        "thermal_risk": thermal,
        "vibration_anomaly": vibration,
        "failure_risk_score": combined_score,
        "failure_risk_level": risk_level,
    }


if __name__ == "__main__":
    print("=== Thermal risk assessment tests ===")
    low_activity = assess_thermal_risk(joint_velocity=0.1)
    high_activity = assess_thermal_risk(joint_velocity=5.0)
    print(f"  Low activity  (v=0.1): core_temp={low_activity['predicted_core_temp_c']:.2f}C  "
          f"risk={low_activity['thermal_risk_score']:.2f}")
    print(f"  High activity (v=5.0): core_temp={high_activity['predicted_core_temp_c']:.2f}C  "
          f"risk={high_activity['thermal_risk_score']:.2f}")
    assert high_activity["predicted_core_temp_c"] > low_activity["predicted_core_temp_c"], \
        "Higher joint activity should predict higher core temperature."
    assert high_activity["thermal_risk_score"] >= low_activity["thermal_risk_score"], \
        "Higher activity should not have lower thermal risk."
    print("Thermal risk correctly increases with joint activity.")

    print("\n=== Combined risk scoring tests ===")
    scenarios = [
        ("none", 0.0, 0.0, "low"),
        ("minor", 0.1, 0.0, "low"),
        ("moderate", 0.3, 0.2, "medium"),
        ("severe", 0.8, 0.6, "critical"),
    ]
    for severity, thermal_r, vib_r, expected_level in scenarios:
        score, level = combine_risk(severity, thermal_r, vib_r)
        status = "PASS" if level == expected_level else "FAIL"
        print(f"  [{status}] severity={severity:9s} thermal={thermal_r:.2f} vib={vib_r:.2f} "
              f"-> score={score:.3f} level={level} (expected {expected_level})")
        assert level == expected_level, f"Expected {expected_level}, got {level}"
    print("\nAll combined risk scoring tests passed.")

    print("\n=== Full node run test (using real digital-twin modules) ===")
    test_state: InspectionState = {
        "joint_id": 2,
        "defect_severity": "moderate",
        "joint_velocities": [2.5],
    }
    result = run(test_state)
    print(f"  thermal_risk:        {result['thermal_risk']}")
    print(f"  vibration_anomaly:   is_anomalous={result['vibration_anomaly']['is_anomalous']}")
    print(f"  failure_risk_score:  {result['failure_risk_score']:.3f}")
    print(f"  failure_risk_level:  {result['failure_risk_level']}")
    assert 0.0 <= result["failure_risk_score"] <= 1.0
    assert result["failure_risk_level"] in ("low", "medium", "high", "critical")
    print("\nNode run test passed.")