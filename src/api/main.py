"""
main.py — FastAPI application entry point.

Loads Component 1's CV model ONCE at startup (not per-request -- a
~24M param model reload on every inspection request would make the API
unusably slow), then exposes REST endpoints that run the full 4-agent
pipeline (Components 1+2+3, via integration/pipeline.py) on uploaded
robot joint images.

Run locally with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Swagger/OpenAPI docs are auto-generated at /docs once running.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))  # this dir, so `routes`/`schemas` bare imports resolve
sys.path.insert(0, str(PROJECT_ROOT / "vision"))
sys.path.insert(0, str(PROJECT_ROOT / "integration"))

from model import PhysicsConstrainedCNN  # noqa: E402
from pipeline import load_checkpoint_or_fresh_model  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT.parent / "outputs" / "model_bottle_best.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaded = load_checkpoint_or_fresh_model(CHECKPOINT_PATH, device)
    app.state.model = model
    app.state.device = device
    app.state.checkpoint_loaded = loaded
    app.state.checkpoint_path = CHECKPOINT_PATH
    print(f"[startup] Model loaded on {device}. "
          f"Checkpoint found: {loaded} ({CHECKPOINT_PATH})")
    yield


app = FastAPI(
    title="Robot Joint Inspection API",
    description="CV defect detection + digital twin + agentic risk assessment "
                "for robot joint quality inspection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Routes are registered here (import triggers route registration on the app instance).
from routes import router  # noqa: E402
app.include_router(router)