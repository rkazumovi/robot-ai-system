"""
test_api.py — Unit tests for the FastAPI inspection API.

Uses FastAPI's TestClient (synchronous, in-process -- no real network
socket needed) to test the actual HTTP request/response cycle: /health,
/inspect (happy path + error handling), and /metrics.
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

@pytest.fixture(scope="module")
def client():
    from main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def fast_reporter(monkeypatch):
    """Forces the template report path for every test in this file, so
    API tests don't depend on network access to download/run the LLM --
    keeping this suite fast and deterministic, appropriate for a CI
    environment that may have restricted or no internet access."""
    import reporter
    monkeypatch.setattr(reporter, "generate_llm_report", lambda state: None)


@pytest.fixture
def fake_image_bytes():
    img_array = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_reports_model_loaded(self, client):
        resp = client.get("/health")
        body = resp.json()
        assert "model_loaded" in body
        assert "status" in body


class TestInspect:
    def test_valid_request_returns_200_with_expected_schema(self, client, fake_image_bytes):
        files = {"image": ("test.png", fake_image_bytes, "image/png")}
        data = {"joint_id": 3, "joint_angles": "0.1,0.2", "joint_velocities": "2.0"}
        resp = client.post("/inspect", files=files, data=data)

        assert resp.status_code == 200
        body = resp.json()
        for field in [
            "joint_id", "defect_detected", "defect_severity", "inspection_summary",
            "thermal_risk", "vibration_anomaly", "failure_risk_score", "failure_risk_level",
            "report_text", "recommended_actions", "parameter_adjustments",
        ]:
            assert field in body

    def test_empty_angles_and_velocities_are_accepted(self, client, fake_image_bytes):
        files = {"image": ("test.png", fake_image_bytes, "image/png")}
        data = {"joint_id": 1, "joint_angles": "", "joint_velocities": ""}
        resp = client.post("/inspect", files=files, data=data)
        assert resp.status_code == 200

    def test_malformed_number_list_returns_422(self, client, fake_image_bytes):
        files = {"image": ("test.png", fake_image_bytes, "image/png")}
        data = {"joint_id": 1, "joint_angles": "not,numbers", "joint_velocities": "0"}
        resp = client.post("/inspect", files=files, data=data)
        assert resp.status_code == 422

    def test_corrupt_image_returns_422(self, client):
        files = {"image": ("test.png", io.BytesIO(b"not an image"), "image/png")}
        data = {"joint_id": 1, "joint_angles": "", "joint_velocities": "0"}
        resp = client.post("/inspect", files=files, data=data)
        assert resp.status_code == 422

    def test_missing_joint_id_returns_422(self, client, fake_image_bytes):
        files = {"image": ("test.png", fake_image_bytes, "image/png")}
        data = {"joint_angles": "", "joint_velocities": "0"}
        resp = client.post("/inspect", files=files, data=data)
        assert resp.status_code == 422


class TestMetrics:
    def test_metrics_endpoint_returns_prometheus_format(self, client, fake_image_bytes):
        # Generate at least one data point first.
        files = {"image": ("test.png", fake_image_bytes, "image/png")}
        data = {"joint_id": 1, "joint_angles": "", "joint_velocities": "0"}
        client.post("/inspect", files=files, data=data)

        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "inspection_requests_total" in resp.text
        assert "inspection_duration_seconds" in resp.text