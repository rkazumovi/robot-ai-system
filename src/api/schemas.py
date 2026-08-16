"""
schemas.py — Pydantic request/response models for the inspection API.

These define the API's public contract (also used to auto-generate the
Swagger/OpenAPI docs at /docs).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

class JointSensorData(BaseModel):
    """Sensor telemetry accompanying an inspection image."""

    joint_id: int = Field(..., description="Which robot joint this inspection is for.")
    joint_angles: list[float] = Field(
        default_factory=list, description="Current joint angles in radians."
    )
    joint_velocities: list[float] = Field(
        default_factory=list, description="Current joint velocities in rad/s."
    )


class RecommendedAction(BaseModel):
    action: str
    priority: str
    reason: str


class ThermalRisk(BaseModel):
    predicted_core_temp_c: float
    thermal_limit_c: float
    margin_c: float
    thermal_risk_score: float


class VibrationAnomaly(BaseModel):
    is_anomalous: bool
    anomaly_freqs: list[float]
    vibration_risk_score: float


class InspectionResponse(BaseModel):
    """Full structured result of running the 4-agent pipeline on one image."""

    joint_id: int
    defect_detected: bool
    defect_severity: str
    inspection_summary: str

    thermal_risk: ThermalRisk
    vibration_anomaly: VibrationAnomaly
    failure_risk_score: float
    failure_risk_level: str

    report_text: str

    recommended_actions: list[RecommendedAction]
    parameter_adjustments: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_path: str | None = None