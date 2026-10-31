"""VISERION — Road Accident Forensics Backend Application (FastAPI).

Provides REST APIs for:
1. Incident Creation & Evidence Store (SHA-256 verification, file management).
2. Forensic Pipeline Execution (/api/pipeline/run & /api/pipeline/load fallback).
3. Timestamp Alignment & Unified Chronological Timeline (/api/incidents/{id}/timeline).
4. Contradiction & Corroboration Engine (/api/incidents/{id}/contradictions).
5. Surrogate Safety Metrics (/api/incidents/{id}/safety).
6. Vehicle Tracks & Trajectories (/api/incidents/{id}/tracks).
7. HTML Forensic Report Streaming (/api/incidents/{id}/report).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.backend.evidence_store import EvidenceStore
from src.backend.pipeline import ForensicPipeline
from src.backend.timeline_merger import merge_and_align_timeline
from src.schema import load_json_records

# ---------------------------------------------------------------------------
# App Initialization & Configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VISERION Forensics API",
    description="AI-Powered Timeline Reconstruction for Road Accident Forensics",
    version="1.0.0",
)

# CORS middleware for local frontend dashboards (e.g. React / Vite on port 3000, 5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evidence_store = EvidenceStore()
pipeline_runner = ForensicPipeline()

# Mount processed files & keyframes for direct static serving
data_dir = Path("data").resolve()
if data_dir.exists():
    app.mount("/static/data", StaticFiles(directory=str(data_dir)), name="static_data")

mock_dir = Path("mock").resolve()
if mock_dir.exists():
    app.mount("/static/mock", StaticFiles(directory=str(mock_dir)), name="static_mock")


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------
class IncidentCreateRequest(BaseModel):
    incident_id: Optional[str] = Field(None, description="Custom incident ID, e.g. CASE-2026-T46-01")
    title: str = Field("Traffic Incident Reconstruction", description="Title or summary of incident")
    description: str = Field("", description="Incident background notes")
    location: str = Field("Intersection", description="Geographical or road location")
    incident_date: Optional[str] = Field(None, description="Date/time of incident (ISO-8601)")


class PipelineRunRequest(BaseModel):
    incident_id: str = Field("CASE-2026-T46-01", description="Target incident ID")
    video_path: Optional[str] = Field(None, description="Path to video file")
    witness_statement: Optional[str] = Field(None, description="Witness text or file path")
    sensor_csv: Optional[str] = Field(None, description="Path to sensor CSV file")
    skip_video_cv: bool = Field(False, description="Skip heavy YOLO/DL inference and use cached/mock video events")


class PipelineLoadRequest(BaseModel):
    incident_id: str = Field("CASE-2026-T46-01", description="Target incident ID")
    precomputed_dir: Optional[str] = Field(None, description="Directory containing precomputed JSON results")


# ---------------------------------------------------------------------------
# System & Health Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root_info() -> dict[str, Any]:
    return {
        "service": "VISERION Forensics API",
        "team": "Team 46 — CBIT Hacktoberfest '26",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "incidents": "/api/incidents",
            "pipeline_run": "/api/pipeline/run",
            "pipeline_load": "/api/pipeline/load",
            "timeline": "/api/incidents/{incident_id}/timeline",
            "contradictions": "/api/incidents/{incident_id}/contradictions",
            "safety": "/api/incidents/{incident_id}/safety",
            "tracks": "/api/incidents/{incident_id}/tracks",
            "report": "/api/incidents/{incident_id}/report",
        },
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    # Check subsystem availability
    p1_cv = False
    try:
        import cv2
        import torch
        p1_cv = True
    except ImportError:
        pass

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subsystems": {
            "p1_computer_vision": p1_cv,
            "p2_nlp_and_sensors": True,
            "p3_backend_and_timeline": True,
        },
        "incidents_registered": len(evidence_store.list_incidents()),
    }


# ---------------------------------------------------------------------------
# 1. Incident Creation & Evidence Store Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/incidents", status_code=status.HTTP_201_CREATED)
def create_incident(payload: IncidentCreateRequest) -> dict[str, Any]:
    incident = evidence_store.create_incident(
        incident_id=payload.incident_id,
        title=payload.title,
        description=payload.description,
        location=payload.location,
        incident_date=payload.incident_date,
    )
    return incident


@app.get("/api/incidents")
def list_incidents() -> list[dict[str, Any]]:
    return evidence_store.list_incidents()


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: str) -> dict[str, Any]:
    incident = evidence_store.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return incident


@app.post("/api/incidents/{incident_id}/evidence", status_code=status.HTTP_201_CREATED)
async def upload_evidence(
    incident_id: str,
    file: UploadFile = File(...),
    evidence_type: str = Form(..., description="video | witness | sensor"),
    description: str = Form("", description="Optional description of the evidence item"),
) -> dict[str, Any]:
    """Upload evidence file, verify SHA-256 hash, and register in evidence store."""
    valid_types = {"video", "witness", "sensor"}
    if evidence_type.lower() not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid evidence_type '{evidence_type}'. Must be one of: {valid_types}",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    entry = evidence_store.store_evidence(
        incident_id=incident_id,
        filename=file.filename or "uploaded_evidence",
        content=content,
        evidence_type=evidence_type,
        description=description,
    )
    return entry


@app.get("/api/incidents/{incident_id}/evidence")
def list_incident_evidence(incident_id: str) -> list[dict[str, Any]]:
    return evidence_store.list_evidence(incident_id)


# ---------------------------------------------------------------------------
# 2. Pipeline Execution Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/pipeline/run")
def run_pipeline(payload: PipelineRunRequest) -> dict[str, Any]:
    """Run full forensic reconstruction pipeline on incident evidence."""
    # Discover registered evidence if paths are not directly supplied
    inc = evidence_store.get_incident(payload.incident_id)
    video_path = payload.video_path
    witness_path = payload.witness_statement
    sensor_path = payload.sensor_csv

    if inc:
        for ev in inc.get("evidence", []):
            ev_type = ev.get("evidence_type")
            if not video_path and ev_type == "video":
                video_path = ev.get("file_path")
            elif not witness_path and ev_type == "witness":
                witness_path = ev.get("file_path")
            elif not sensor_path and ev_type == "sensor":
                sensor_path = ev.get("file_path")

    # If evidence still not provided, fallback to demo/ files
    if not video_path and Path("data/raw/tbone.mp4").exists():
        video_path = "data/raw/tbone.mp4"
    elif not video_path and Path("demo/tbone.mp4").exists():
        video_path = "demo/tbone.mp4"

    if not witness_path and Path("demo/witness_statement_1.txt").exists():
        witness_path = "demo/witness_statement_1.txt"

    if not sensor_path and Path("demo/sensor_telemetry.csv").exists():
        sensor_path = "demo/sensor_telemetry.csv"

    out_dir = Path("data/processed") / payload.incident_id
    result = pipeline_runner.run_pipeline(
        incident_id=payload.incident_id,
        video_path=video_path,
        witness_statement=witness_path,
        sensor_csv=sensor_path,
        output_dir=out_dir,
        skip_video_cv=payload.skip_video_cv,
    )
    return result


@app.post("/api/pipeline/load")
def load_precomputed_pipeline(payload: PipelineLoadRequest) -> dict[str, Any]:
    """Instant fallback: loads precomputed / mock results without re-running models."""
    out_dir = Path("data/processed") / payload.incident_id
    result = pipeline_runner.load_precomputed(
        incident_id=payload.incident_id,
        precomputed_dir=payload.precomputed_dir,
        output_dir=out_dir,
    )
    return result


# Convenience alias matching prompt: /pipeline/run & /pipeline/load
@app.post("/pipeline/run")
def alias_pipeline_run(payload: PipelineRunRequest) -> dict[str, Any]:
    return run_pipeline(payload)


@app.post("/pipeline/load")
def alias_pipeline_load(payload: PipelineLoadRequest) -> dict[str, Any]:
    return load_precomputed_pipeline(payload)


# ---------------------------------------------------------------------------
# 3. Forensic Artifact Retrieval Endpoints
# ---------------------------------------------------------------------------
def _find_incident_file(incident_id: str, filename: str) -> Optional[Path]:
    candidates = [
        Path("data/processed") / incident_id / filename,
        Path("data/processed/tbone_demo") / filename,
        Path("mock") / filename,
        Path(filename),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


@app.get("/api/incidents/{incident_id}/timeline")
def get_timeline(incident_id: str) -> dict[str, Any]:
    """Retrieve the unified chronological timeline (timeline.json)."""
    t_file = _find_incident_file(incident_id, "timeline.json")
    if t_file:
        try:
            return json.loads(t_file.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading timeline: {e}")

    # Fallback to dynamic merge
    merged = merge_and_align_timeline(incident_id=incident_id)
    return merged


@app.get("/api/incidents/{incident_id}/contradictions")
def get_contradictions(incident_id: str) -> list[dict[str, Any]]:
    """Retrieve cross-source evidence links (contradictions, support, gaps)."""
    l_file = _find_incident_file(incident_id, "evidence_links.json")
    if l_file:
        return load_json_records(l_file)
    return []


@app.get("/api/incidents/{incident_id}/safety")
def get_safety_indicators(incident_id: str) -> list[dict[str, Any]]:
    """Retrieve surrogate safety indicators (TTC and DRAC values)."""
    s_file = _find_incident_file(incident_id, "safety_indicators.json")
    if s_file:
        return load_json_records(s_file)
    return []


@app.get("/api/incidents/{incident_id}/tracks")
def get_tracks(incident_id: str) -> dict[str, Any]:
    """Retrieve vehicle detection tracks and bounding boxes."""
    t_file = _find_incident_file(incident_id, "tracks.json")
    if t_file:
        try:
            return json.loads(t_file.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading tracks: {e}")
    raise HTTPException(status_code=404, detail="Tracks not found for this incident.")


@app.get("/api/incidents/{incident_id}/report", response_class=HTMLResponse)
def get_report(incident_id: str) -> HTMLResponse:
    """Stream standalone forensic HTML reconstruction report directly to browser."""
    r_file = _find_incident_file(incident_id, "report.html")
    if r_file:
        content = r_file.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    raise HTTPException(status_code=404, detail="Forensic report not yet generated for this incident.")
