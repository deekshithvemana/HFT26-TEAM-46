import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import featherless_client
from src.intelligence.featherless_client import FeatherlessClient, encode_image_to_base64


def test_root_export_importable() -> None:
    client = featherless_client.FeatherlessClient(mock=True)
    assert client.is_mock() is True


def test_encode_image_to_base64(tmp_path: Path) -> None:
    img_path = tmp_path / "test.jpg"
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    uri = encode_image_to_base64(img_path)
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) > 30


def test_describe_keyframe_inference_contract(tmp_path: Path) -> None:
    img_path = tmp_path / "frame_001.jpg"
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(img_path), img)

    client = FeatherlessClient(mock=True)
    res = client.describe_keyframe(
        img_path,
        event_context={"event_type": "sudden_braking", "time_s": 2.4},
    )

    assert "description" in res
    assert res["kind"] == "inference"
    assert 0.0 <= res["confidence"] <= 1.0
    assert "visual_elements" in res
    assert "uncertainty_factors" in res
    assert len(res["uncertainty_factors"]) > 0


def test_extract_witness_claims_mock() -> None:
    statement = (
        "I was waiting at the corner. The red sedan ran the red light around 4 seconds in. "
        "The driver didn't brake at all before hitting the blue car."
    )
    client = FeatherlessClient(mock=True)
    claims = client.extract_witness_claims(statement, statement_id="W1")

    assert len(claims) >= 2
    claim_types = [c["action"] for c in claims]
    assert "signal_violation" in claim_types or "no_braking" in claim_types or "collision" in claim_types
    for c in claims:
        assert c["statement_id"] == "W1"
        assert c["kind"] == "claim"
        if c["time_interval"] is not None:
            assert c["time_interval"][0] <= c["time_interval"][1]


def test_generate_grounded_explanation_avoids_legal_fault() -> None:
    client = FeatherlessClient(mock=True)
    explanation = client.generate_explanation(
        finding_status="contradict",
        claim_summary="Witness claims vehicle never braked",
        evidence_summary="Accelerometer recorded 0.72g deceleration at t=3.8s",
    )

    assert "Contradiction" in explanation or "contradict" in explanation.lower()
    # Ensure it avoids declaring legal liability or guilt
    forbidden_terms = ["guilty", "liable", "at fault", "negligent", "convicted"]
    for term in forbidden_terms:
        assert term not in explanation.lower()
