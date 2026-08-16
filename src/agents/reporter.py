"""
reporter.py — Agent 3 (Reporter): generates a structured report via HuggingFace.

Uses google/flan-t5-small (~80M params, ~300MB) rather than a larger
chat-style model: this task is structured summarization from numeric
findings, not open-ended conversation, and a small instruction-tuned
model is fast enough for a live pipeline while a 3-4B model would add
minutes of latency per report on CPU for no real quality benefit here.

Falls back to a deterministic template-based report if the HF model
fails to load (no internet, out of disk space, etc.) -- a production
pipeline's reporting stage should never hard-fail just because a
language model download hiccups; the structured findings themselves
(from Inspector/Predictor) are what actually matter operationally.

Run this file directly to test prompt construction and the template
fallback path (testable offline). The actual HF model download/inference
path needs a machine with internet access to huggingface.co to verify.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from state import InspectionState  # noqa: E402

DEFAULT_MODEL = "google/flan-t5-base"

def build_report_prompt(state: InspectionState) -> str:
    """Constructs a structured instruction prompt from the pipeline's findings so far."""
    joint_id = state.get("joint_id", "unknown")
    inspection_summary = state.get("inspection_summary", "No inspection data available.")
    severity = state.get("defect_severity", "none")
    risk_score = state.get("failure_risk_score", 0.0)
    risk_level = state.get("failure_risk_level", "low")
    thermal = state.get("thermal_risk", {})
    vibration = state.get("vibration_anomaly", {})

    thermal_line = (
        f"Predicted core temperature {thermal.get('predicted_core_temp_c', 'N/A'):.1f}C "
        f"against a {thermal.get('thermal_limit_c', 'N/A')}C limit "
        f"(margin {thermal.get('margin_c', 'N/A'):.1f}C)."
        if thermal else "No thermal data available."
    )
    vibration_line = (
        f"Vibration anomaly detected at frequencies {vibration.get('anomaly_freqs', [])} Hz."
        if vibration.get("is_anomalous")
        else "No vibration anomalies detected."
    )

    prompt = (
        "You are a maintenance technician. Write ONE flowing paragraph (not a list, "
        "not field labels) summarizing this robot joint inspection in plain professional "
        "language, as if reporting to a supervisor.\n\n"
        f"Joint {joint_id}. {inspection_summary} "
        f"{thermal_line} {vibration_line} "
        f"Overall failure risk is assessed as {risk_level} (risk score {risk_score:.2f} out of 1.00).\n\n"
        "Paragraph:"
    )
    return prompt


def generate_template_report(state: InspectionState) -> str:
    """
    Deterministic, rule-based report -- the guaranteed-working fallback
    used if the HF model is unavailable, and also useful as a fast,
    fully-offline-testable report path in its own right.
    """
    joint_id = state.get("joint_id", "unknown")
    severity = state.get("defect_severity", "none")
    risk_level = state.get("failure_risk_level", "low")
    risk_score = state.get("failure_risk_score", 0.0)
    thermal = state.get("thermal_risk", {})
    vibration = state.get("vibration_anomaly", {})

    lines = [f"Inspection Report -- Joint {joint_id}", ""]

    if severity == "none":
        lines.append("No surface defects were detected during this inspection.")
    else:
        lines.append(
            f"A {severity} defect was identified during visual inspection. "
            f"{state.get('inspection_summary', '')}"
        )

    if thermal:
        margin = thermal.get("margin_c", None)
        if margin is not None and margin < 15:
            lines.append(
                f"Thermal analysis indicates elevated operating temperature "
                f"(predicted {thermal.get('predicted_core_temp_c', 0):.1f}C, "
                f"only {margin:.1f}C below the {thermal.get('thermal_limit_c', 0)}C limit)."
            )
        else:
            lines.append("Thermal analysis shows the joint operating within a safe temperature range.")

    if vibration.get("is_anomalous"):
        freqs = vibration.get("anomaly_freqs", [])
        lines.append(f"Vibration analysis detected anomalous spectral content near {freqs} Hz, "
                      f"consistent with possible mechanical wear.")
    else:
        lines.append("Vibration analysis shows no anomalous spectral content.")

    lines.append(
        f"Overall assessed failure risk: {risk_level.upper()} (score {risk_score:.2f}/1.00)."
    )

    return " ".join(lines[1:]) if len(lines) > 1 else lines[0]


_MODEL_CACHE: dict = {}


def _get_model_and_tokenizer(model_name: str):
    """Loads (and caches) the tokenizer/model once per process, so repeated
    report generations don't reload ~1GB of weights from scratch every call --
    a real cost in a pipeline processing many inspections in sequence."""
    if model_name not in _MODEL_CACHE:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


def generate_llm_report(state: InspectionState, model_name: str = DEFAULT_MODEL) -> str | None:
    """
    Attempts to generate a report via a small HuggingFace seq2seq model.
    Uses AutoModelForSeq2SeqLM + tokenizer.generate() directly rather
    than the pipeline() convenience wrapper: transformers v5 removed the
    "text2text-generation" pipeline task name entirely (a real breaking
    change, not a mistake in this code -- see
    github.com/huggingface/transformers/issues/44509), while the
    AutoModelForSeq2SeqLM class-based API has been stable for years and
    isn't affected by pipeline task-registry churn.

    Model/tokenizer are cached at module level (see _get_model_and_tokenizer)
    so repeated calls in the same process reuse already-loaded weights.

    Returns None (triggering the template fallback) on any failure --
    missing internet, missing package, out of memory, etc. -- rather than
    raising, since report generation should degrade gracefully.
    """
    try:
        tokenizer, model = _get_model_and_tokenizer(model_name)
    except Exception:
        return None

    try:
        prompt = build_report_prompt(state)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = model.generate(**inputs, max_new_tokens=120, num_beams=4)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return text if text else None
    except Exception:
        return None


def run(state: InspectionState) -> dict:
    """LangGraph node function: generates the report, preferring the LLM
    path but always falling back to the template path on any failure."""
    report = generate_llm_report(state)
    if not report:
        report = generate_template_report(state)
    return {"report_text": report}


if __name__ == "__main__":
    print("=== Prompt construction test ===")
    test_state: InspectionState = {
        "joint_id": 4,
        "inspection_summary": "Joint 4: moderate defect detected -- 3.20% of surface area affected.",
        "defect_severity": "moderate",
        "thermal_risk": {
            "predicted_core_temp_c": 68.5, "thermal_limit_c": 80.0, "margin_c": 11.5,
        },
        "vibration_anomaly": {"is_anomalous": True, "anomaly_freqs": [36.6, 37.6]},
        "failure_risk_score": 0.62,
        "failure_risk_level": "high",
    }
    prompt = build_report_prompt(test_state)
    print(prompt)
    assert "Joint 4" in prompt
    assert "moderate" in prompt
    assert "high" in prompt
    print("\nPrompt construction verified.")

    print("\n=== Template fallback report test ===")
    template_report = generate_template_report(test_state)
    print(template_report)
    assert "Joint 4" in template_report or "joint" in template_report.lower()
    assert "moderate" in template_report
    assert "HIGH" in template_report
    print("\nTemplate report verified.")

    print("\n=== Low-risk scenario template test ===")
    low_risk_state: InspectionState = {
        "joint_id": 1,
        "inspection_summary": "Joint 1: no defects detected.",
        "defect_severity": "none",
        "thermal_risk": {"predicted_core_temp_c": 30.0, "thermal_limit_c": 80.0, "margin_c": 50.0},
        "vibration_anomaly": {"is_anomalous": False, "anomaly_freqs": []},
        "failure_risk_score": 0.05,
        "failure_risk_level": "low",
    }
    low_risk_report = generate_template_report(low_risk_state)
    print(low_risk_report)
    assert "No surface defects" in low_risk_report
    assert "LOW" in low_risk_report
    print("\nLow-risk template report verified.")

    print("\n=== Full node run test (template path, since HF model needs internet) ===")
    result = run(test_state)
    print(result["report_text"])
    assert len(result["report_text"]) > 0
    print("\nNode run test passed (used template fallback -- expected in this sandbox, "
          "which cannot reach huggingface.co; the LLM path needs testing on a machine "
          "with internet access).")