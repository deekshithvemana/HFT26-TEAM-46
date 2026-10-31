"""Unit & Integration tests for Person 3 Backend (FastAPI, Pipeline, Evidence Store, Timeline Merge)."""

import asyncio
from pathlib import Path
import pytest

from src.backend.app import (
    app,
    create_incident,
    get_contradictions,
    get_incident,
    get_report,
    get_safety_indicators,
    get_timeline,
    health_check,
    alias_pipeline_load,
    IncidentCreateRequest,
    PipelineLoadRequest,
    root_info,
)
from src.backend.evidence_store import EvidenceStore
from src.backend.pipeline import ForensicPipeline
from src.backend.timeline_merger import TimelineMerger
from src.schema import EventKind, SourceType


def test_evidence_store_lifecycle(tmp_path: Path):
    store = EvidenceStore(base_dir=tmp_path)

    # 1. Create incident
    inc = store.create_incident(
        incident_id="TEST-INC-01",
        title="Test Crash",
        description="Two-vehicle intersection collision",
        location="Main St & 5th Ave",
    )
    assert inc["incident_id"] == "TEST-INC-01"
    assert inc["status"] == "created"

    # 2. Store evidence with SHA-256
    sample_bytes = b"time_s,speed_kmh,accel_m_s2\n0.0,50.0,0.0\n0.5,20.0,-6.0"
    evd = store.store_evidence(
        incident_id="TEST-INC-01",
        filename="telemetry.csv",
        content=sample_bytes,
        evidence_type="sensor",
        description="OBD logger",
    )
    assert evd["evidence_id"].startswith("EVD-SEN-")
    assert evd["size_bytes"] == len(sample_bytes)
    assert len(evd["sha256"]) == 64

    # 3. Retrieve and verify
    retrieved = store.get_incident("TEST-INC-01")
    assert retrieved is not None
    assert retrieved["evidence_count"] == 1
    assert retrieved["status"] == "evidence_uploaded"

    file_p = store.get_evidence_file_path("TEST-INC-01", evd["evidence_id"])
    assert file_p is not None
    assert file_p.exists()
    assert file_p.read_bytes() == sample_bytes


def test_timeline_merger_alignment_and_clustering(tmp_path: Path):
    merger = TimelineMerger(time_zero_offset_s=1.0)

    video_events = [
        {
            "id": "V-01",
            "event_type": "collision",
            "start_time": 0.5,
            "end_time": 1.2,
            "confidence": 0.95,
            "kind": "observation",
            "description": "Vehicle collision",
        }
    ]
    sensor_events = [
        {
            "id": "S-01",
            "event_type": "hard_braking",
            "time": 0.2,
            "value": -7.5,
            "unit": "m/s^2",
            "source_row": 5,
        }
    ]
    witness_claims = [
        {
            "id": "W-01",
            "statement_id": "W1",
            "original_text": "The SUV collided at 0.5s",
            "vehicle_desc": "black SUV",
            "action": "collision",
            "time_interval": [0.5, 1.0],
            "uncertainty": 0.3,
            "confidence": 0.75,
        },
        {
            "id": "W-02",
            "statement_id": "W1",
            "original_text": "It was parked earlier",
            "vehicle_desc": "white SUV",
            "action": "parked",
            "time_interval": None,  # undated claim
        },
    ]

    out_file = tmp_path / "test_timeline.json"
    result = merger.merge(
        incident_id="CASE-MERGE-TEST",
        video_events=video_events,
        sensor_events=sensor_events,
        witness_claims=witness_claims,
        output_path=out_file,
    )

    assert result["incident_id"] == "CASE-MERGE-TEST"
    assert result["total_events"] == 3  # V-01, S-01, and dated W-01
    assert result["total_undated_claims"] == 1  # W-02 is undated
    assert out_file.exists()

    events = result["events"]
    # Check alignment (offset +1.0 added): S-01 (0.2 + 1.0 = 1.2), V-01 (0.5 + 1.0 = 1.5)
    assert events[0]["id"] == "S-01"
    assert events[0]["start_time"] == 1.2
    assert events[0]["kind"] == EventKind.DERIVED_MEASUREMENT.value

    # Check clusters formed
    assert len(result["clusters"]) >= 1
    assert any("Phase" in c["phase"] for c in result["clusters"])


def test_pipeline_load_precomputed(tmp_path: Path):
    pipeline = ForensicPipeline(default_output_base=tmp_path)
    res = pipeline.load_precomputed(incident_id="CASE-PRECOMPUTED-01", output_dir=tmp_path / "run1")

    assert res["status"] == "success"
    assert res["mode"] == "precomputed"
    assert "artifacts" in res
    assert Path(res["artifacts"]["timeline_json"]).exists()
    assert Path(res["artifacts"]["report_html"]).exists()
    assert res["timeline"]["total_events"] > 0


def test_api_root_and_health():
    info = root_info()
    assert info["service"] == "VISERION Forensics API"
    assert "endpoints" in info

    health = health_check()
    assert health["status"] == "healthy"
    assert health["subsystems"]["p3_backend_and_timeline"] is True


def test_api_incident_and_evidence():
    # 1. Create incident via route handler
    payload = IncidentCreateRequest(
        incident_id="API-TEST-INC-01",
        title="API Test Crash",
        description="Integration test",
        location="Test Blvd",
    )
    inc = create_incident(payload)
    assert inc["incident_id"] == "API-TEST-INC-01"

    # 2. Retrieve
    retrieved = get_incident("API-TEST-INC-01")
    assert retrieved["incident_id"] == "API-TEST-INC-01"


def test_api_pipeline_load_and_artifact_retrieval(tmp_path: Path):
    # Call /pipeline/load route handler
    load_payload = PipelineLoadRequest(incident_id="API-LOAD-INC-02")
    pipe_res = alias_pipeline_load(load_payload)
    assert pipe_res["status"] == "success"
    assert pipe_res["mode"] == "precomputed"

    # Verify timeline retrieval
    tl_res = get_timeline("API-LOAD-INC-02")
    assert "events" in tl_res
    assert "summary" in tl_res
    assert tl_res["total_events"] > 0

    # Verify contradictions retrieval
    contras = get_contradictions("API-LOAD-INC-02")
    assert isinstance(contras, list)

    # Verify safety indicators retrieval
    safety = get_safety_indicators("API-LOAD-INC-02")
    assert isinstance(safety, list)

    # Verify HTML report retrieval
    html_res = get_report("API-LOAD-INC-02")
    assert "<!DOCTYPE html>" in html_res.body.decode("utf-8")
