"""
orchestrator.py — LangGraph pipeline wiring the four agents together.

Graph structure (linear pipeline, matching the project spec):

    START -> inspector -> predictor -> reporter -> optimizer -> END

Each node is one of the agent modules' run(state) -> dict functions.
LangGraph merges each returned dict into the shared InspectionState as
it flows through the graph.

Run this file directly to execute the full pipeline end-to-end on a
few synthetic CV inputs (low/medium/high risk scenarios) and verify the
final state contains all expected fields, correctly populated by each
stage in sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).parent))
import inspector  # noqa: E402
import optimizer  # noqa: E402
import predictor  # noqa: E402
import reporter  # noqa: E402
from state import InspectionState  # noqa: E402

def build_graph():
    graph = StateGraph(InspectionState)

    graph.add_node("inspector", inspector.run)
    graph.add_node("predictor", predictor.run)
    graph.add_node("reporter", reporter.run)
    graph.add_node("optimizer", optimizer.run)

    graph.add_edge(START, "inspector")
    graph.add_edge("inspector", "predictor")
    graph.add_edge("predictor", "reporter")
    graph.add_edge("reporter", "optimizer")
    graph.add_edge("optimizer", END)

    return graph.compile()


def run_inspection(initial_state: InspectionState) -> InspectionState:
    """Runs the full 4-agent pipeline on a single CV inspection result."""
    app = build_graph()
    final_state = app.invoke(initial_state)
    return final_state


def print_report(state: InspectionState) -> None:
    print(f"\n{'=' * 70}")
    print(f"JOINT {state.get('joint_id')} -- INSPECTION PIPELINE RESULT")
    print(f"{'=' * 70}")
    print(f"\n[Inspector]  {state.get('inspection_summary')}")
    print(f"[Predictor]  Risk: {state.get('failure_risk_level', '').upper()} "
          f"(score {state.get('failure_risk_score', 0):.2f})")
    thermal = state.get("thermal_risk", {})
    print(f"             Thermal margin: {thermal.get('margin_c', 'N/A')}")
    vibration = state.get("vibration_anomaly", {})
    print(f"             Vibration anomaly: {vibration.get('is_anomalous')}")
    print(f"\n[Reporter]   {state.get('report_text')}")
    print(f"\n[Optimizer]  Parameter adjustments: {state.get('parameter_adjustments')}")
    print("             Recommended actions:")
    for a in state.get("recommended_actions", []):
        print(f"               [{a['priority']}] {a['action']} -- {a['reason']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    print("Building LangGraph pipeline...")
    app = build_graph()
    print("Graph compiled successfully.\n")

    scenarios = [
        {
            "joint_id": 1,
            "defect_mask_summary": {"defect_pixel_fraction": 0.0, "num_components": 0},
            "defect_confidence": 0.98,
            "joint_velocities": [0.2],
        },
        {
            "joint_id": 5,
            "defect_mask_summary": {"defect_pixel_fraction": 0.028, "num_components": 1},
            "defect_confidence": 0.87,
            "joint_velocities": [3.0],
        },
        {
            "joint_id": 2,
            "defect_mask_summary": {"defect_pixel_fraction": 0.12, "num_components": 3},
            "defect_confidence": 0.94,
            "joint_velocities": [6.5],
        },
    ]

    all_final_states = []
    for scenario in scenarios:
        final_state = app.invoke(scenario)
        all_final_states.append(final_state)
        print_report(final_state)

    print("\n\n=== Pipeline integration checks ===")
    required_fields = [
        "defect_detected", "defect_severity", "inspection_summary",
        "thermal_risk", "vibration_anomaly", "failure_risk_score", "failure_risk_level",
        "report_text", "recommended_actions", "parameter_adjustments",
    ]
    for i, state in enumerate(all_final_states):
        missing = [f for f in required_fields if f not in state]
        status = "PASS" if not missing else "FAIL"
        print(f"  [{status}] Scenario {i+1} (joint {state.get('joint_id')}): "
              f"{'all fields present' if not missing else f'missing {missing}'}")
        assert not missing, f"Scenario {i+1} missing fields: {missing}"

    # Sanity: risk should generally track defect severity across these
    # scenarios (not a strict guarantee given thermal/vibration also
    # contribute, but should hold for this deliberately separated set).
    risk_scores = [s["failure_risk_score"] for s in all_final_states]
    print(f"\n  Risk scores across scenarios (expect roughly increasing): "
          f"{[f'{r:.3f}' for r in risk_scores]}")
    assert risk_scores[0] < risk_scores[2], \
        "The no-defect/low-activity scenario should have lower risk than the severe/high-activity one."

    print("\nAll pipeline integration checks passed.")