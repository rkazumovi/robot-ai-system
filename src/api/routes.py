"""
routes.py — API endpoint definitions.

POST /inspect: accepts an uploaded robot joint image + sensor telemetry,
    runs the full 4-agent pipeline, returns the structured result.
GET  /health: reports whether the CV model loaded successfully at startup.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "integration"))
from pipeline import run_full_pipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "monitoring"))
from metrics import (  # noqa: E402
    cv_inference_duration_seconds,
    inspection_duration_seconds,
    record_error,
    record_inspection_result,
    render_metrics,
    time_histogram,
)

from schemas import HealthResponse, InspectionResponse  # noqa: E402

router = APIRouter()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@router.get("/metrics")
def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model = getattr(request.app.state, "model", None)
    checkpoint_loaded = getattr(request.app.state, "checkpoint_loaded", False)
    checkpoint_path = getattr(request.app.state, "checkpoint_path", None)
    return HealthResponse(
        status="ok" if model is not None else "model not loaded",
        model_loaded=model is not None,
        checkpoint_path=str(checkpoint_path) if checkpoint_loaded else None,
    )


def parse_number_list(value: str) -> list[float]:
    """
    Parses a comma-separated list of numbers, e.g. "0.1,0.2" or "0.1, 0.2".
    Empty string returns an empty list. Deliberately NOT JSON syntax
    (no brackets/quotes required) -- much more forgiving for manual
    entry through a web form or Swagger UI, where users naturally type
    "0.1,0.2" rather than "[0.1, 0.2]".
    """
    value = value.strip()
    if not value:
        return []
    try:
        return [float(x.strip()) for x in value.split(",")]
    except ValueError as e:
        raise ValueError(f"Could not parse '{value}' as comma-separated numbers: {e}")


@router.post("/inspect", response_model=InspectionResponse)
async def inspect(
    request: Request,
    image: UploadFile = File(..., description="Robot joint image (PNG/JPEG)."),
    joint_id: int = Form(..., description="Which joint this inspection is for."),
    joint_angles: str = Form("", description="Comma-separated joint angles in radians, e.g. 0.1,0.2,-0.3"),
    joint_velocities: str = Form("0.0", description="Comma-separated joint velocities in rad/s, e.g. 2.0"),
) -> InspectionResponse:
    model = getattr(request.app.state, "model", None)
    device = getattr(request.app.state, "device", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    try:
        angles = parse_number_list(joint_angles)
        velocities = parse_number_list(joint_velocities)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        record_error(stage="image_processing")
        raise HTTPException(status_code=422, detail=f"Could not read uploaded image: {e}")

    image_tensor = image_transform(pil_image)

    try:
        with time_histogram(inspection_duration_seconds):
            final_state = run_full_pipeline(
                image=image_tensor,
                joint_id=joint_id,
                joint_angles=angles,
                joint_velocities=velocities,
                model=model,
                device=device,
            )
    except Exception as e:
        record_error(stage="pipeline")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")

    record_inspection_result(
        severity=final_state.get("defect_severity", "none"),
        risk_level=final_state.get("failure_risk_level", "low"),
        confidence=final_state.get("defect_confidence", 0.0),
    )

    return InspectionResponse(
        joint_id=final_state.get("joint_id", joint_id),
        defect_detected=final_state.get("defect_detected", False),
        defect_severity=final_state.get("defect_severity", "none"),
        inspection_summary=final_state.get("inspection_summary", ""),
        thermal_risk=final_state.get("thermal_risk", {}),
        vibration_anomaly=final_state.get("vibration_anomaly", {}),
        failure_risk_score=final_state.get("failure_risk_score", 0.0),
        failure_risk_level=final_state.get("failure_risk_level", "low"),
        report_text=final_state.get("report_text", ""),
        recommended_actions=final_state.get("recommended_actions", []),
        parameter_adjustments=final_state.get("parameter_adjustments", {}),
    )