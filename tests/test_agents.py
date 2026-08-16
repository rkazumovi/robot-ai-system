"""
test_agents.py — Unit tests for Component 3: the LangGraph agent pipeline.

Covers inspector.py, predictor.py, optimizer.py, and reporter.py's
template fallback path. Deliberately does NOT test reporter.py's real
LLM path here -- that requires network access to download/run
flan-t5-base, which is slow and not appropriate for a fast, hermetic
unit test suite (it was validated manually, see the project's
development history). Also includes an end-to-end orchestrator test
using the (fast, offline) template report path.
"""

import pytest
import inspector
import optimizer
import predictor
import reporter
from state import InspectionState

class TestInspector:
    @pytest.mark.parametrize("fraction,confidence,expected", [
        (0.0, 0.99, "none"),
        (0.005, 0.95, "minor"),
        (0.03, 0.90, "moderate"),
        (0.10, 0.95, "severe"),
        (0.10, 0.40, "moderate"),  # low confidence downgrades severe -> moderate
        (0.03, 0.30, "minor"),
        (0.005, 0.20, "none"),
    ])
    def test_severity_classification(self, fraction, confidence, expected):
        assert inspector.classify_severity(fraction, confidence) == expected

    def test_run_produces_expected_fields(self):
        state: InspectionState = {
            "joint_id": 3,
            "defect_mask_summary": {"defect_pixel_fraction": 0.032, "num_components": 2},
            "defect_confidence": 0.91,
        }
        result = inspector.run(state)
        assert result["defect_detected"] is True
        assert result["defect_severity"] == "moderate"
        assert "Joint 3" in result["inspection_summary"]


class TestPredictor:
    def test_thermal_risk_increases_with_joint_activity(self):
        low = predictor.assess_thermal_risk(joint_velocity=0.1)
        high = predictor.assess_thermal_risk(joint_velocity=5.0)
        assert high["predicted_core_temp_c"] > low["predicted_core_temp_c"]
        assert high["thermal_risk_score"] >= low["thermal_risk_score"]

    @pytest.mark.parametrize("severity,thermal_r,vib_r,expected_level", [
        ("none", 0.0, 0.0, "low"),
        ("minor", 0.1, 0.0, "low"),
        ("moderate", 0.3, 0.2, "medium"),
        ("severe", 0.8, 0.6, "critical"),
    ])
    def test_combined_risk_scoring(self, severity, thermal_r, vib_r, expected_level):
        _, level = predictor.combine_risk(severity, thermal_r, vib_r)
        assert level == expected_level

    def test_run_produces_expected_fields(self):
        state: InspectionState = {
            "joint_id": 2, "defect_severity": "moderate", "joint_velocities": [2.5],
        }
        result = predictor.run(state)
        assert 0.0 <= result["failure_risk_score"] <= 1.0
        assert result["failure_risk_level"] in ("low", "medium", "high", "critical")
        assert "predicted_core_temp_c" in result["thermal_risk"]


class TestOptimizer:
    def test_parameter_scales_decrease_monotonically_with_risk(self):
        prev = 2.0
        for level in ["low", "medium", "high", "critical"]:
            adj = optimizer.compute_parameter_adjustments(level)
            assert adj["max_velocity_scale"] <= prev
            prev = adj["max_velocity_scale"]

    def test_low_risk_recommends_normal_operation(self):
        state: InspectionState = {
            "defect_severity": "none",
            "thermal_risk": {"margin_c": 50.0},
            "vibration_anomaly": {"is_anomalous": False},
            "failure_risk_level": "low",
        }
        result = optimizer.run(state)
        assert result["recommended_actions"][0]["action"] == "Continue normal operation"
        assert result["parameter_adjustments"]["max_velocity_scale"] == 1.0

    def test_critical_risk_recommends_offline_and_sorts_by_priority(self):
        state: InspectionState = {
            "defect_severity": "severe",
            "thermal_risk": {"margin_c": 3.0},
            "vibration_anomaly": {"is_anomalous": True, "anomaly_freqs": [37.0]},
            "failure_risk_level": "critical",
        }
        result = optimizer.run(state)
        assert result["recommended_actions"][0]["priority"] == "critical"
        assert any("offline" in a["action"].lower() for a in result["recommended_actions"])
        assert result["parameter_adjustments"]["max_velocity_scale"] == 0.3


class TestReporter:
    def test_template_report_includes_key_findings(self):
        state: InspectionState = {
            "joint_id": 4,
            "inspection_summary": "Joint 4: moderate defect detected.",
            "defect_severity": "moderate",
            "thermal_risk": {"predicted_core_temp_c": 68.5, "thermal_limit_c": 80.0, "margin_c": 11.5},
            "vibration_anomaly": {"is_anomalous": True, "anomaly_freqs": [36.6, 37.6]},
            "failure_risk_score": 0.62,
            "failure_risk_level": "high",
        }
        report = reporter.generate_template_report(state)
        assert "moderate" in report
        assert "HIGH" in report

    def test_low_risk_template_report(self):
        state: InspectionState = {
            "joint_id": 1,
            "defect_severity": "none",
            "thermal_risk": {"predicted_core_temp_c": 30.0, "thermal_limit_c": 80.0, "margin_c": 50.0},
            "vibration_anomaly": {"is_anomalous": False, "anomaly_freqs": []},
            "failure_risk_score": 0.05,
            "failure_risk_level": "low",
        }
        report = reporter.generate_template_report(state)
        assert "No surface defects" in report
        assert "LOW" in report


class TestOrchestrator:
    """End-to-end pipeline test using the fast, offline template report
    path (generate_llm_report will fail without network access in a CI
    environment and fall back automatically -- this test verifies that
    fallback keeps the whole pipeline working, not just reporter.py alone)."""

    def test_full_pipeline_produces_complete_state(self, monkeypatch):
        # Force the template path so this test is fast and network-independent.
        monkeypatch.setattr(reporter, "generate_llm_report", lambda state: None)

        from orchestrator import build_graph

        app = build_graph()
        final_state = app.invoke({
            "joint_id": 5,
            "defect_mask_summary": {"defect_pixel_fraction": 0.028, "num_components": 1},
            "defect_confidence": 0.87,
            "joint_velocities": [3.0],
        })

        required_fields = [
            "defect_detected", "defect_severity", "inspection_summary",
            "thermal_risk", "vibration_anomaly", "failure_risk_score", "failure_risk_level",
            "report_text", "recommended_actions", "parameter_adjustments",
        ]
        for field in required_fields:
            assert field in final_state, f"Missing field: {field}"

    def test_risk_escalates_with_defect_severity(self, monkeypatch):
        monkeypatch.setattr(reporter, "generate_llm_report", lambda state: None)
        from orchestrator import build_graph
        app = build_graph()

        low = app.invoke({
            "joint_id": 1,
            "defect_mask_summary": {"defect_pixel_fraction": 0.0, "num_components": 0},
            "defect_confidence": 0.98, "joint_velocities": [0.2],
        })
        high = app.invoke({
            "joint_id": 2,
            "defect_mask_summary": {"defect_pixel_fraction": 0.12, "num_components": 3},
            "defect_confidence": 0.94, "joint_velocities": [6.5],
        })
        assert low["failure_risk_score"] < high["failure_risk_score"]