import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intelligence.matcher import EvidenceMatcher, generate_evidence_links
from src.schema import LinkStatus


def test_matcher_detects_contradiction_between_witness_and_sensor() -> None:
    claims = [
        {
            "id": "W2-C4",
            "action": "no_braking",
            "original_text": "never braked before the impact",
            "vehicle_desc": "black SUV",
        }
    ]
    sensors = [
        {
            "id": "S-01",
            "event_type": "hard_braking",
            "time": 2.4,
            "value": -7.8,
            "unit": "m/s^2",
        }
    ]

    matcher = EvidenceMatcher()
    links = matcher.match_evidence(claims, sensors)

    contradiction = next(l for l in links if l.status == LinkStatus.CONTRADICT.value)
    assert contradiction.linked_ids == ["W2-C4", "S-01"]
    assert contradiction.confidence > 0.9
    assert "Contradiction" in contradiction.explanation
    # Ensure neutral forensic tone (no legal fault claims)
    for forbidden in ["guilty", "liable", "at fault", "negligent"]:
        assert forbidden not in contradiction.explanation.lower()


def test_matcher_detects_trimodal_cross_evidence() -> None:
    claims = [
        {"id": "W2-C1", "action": "stopping", "vehicle_desc": "white SUV", "original_text": "white SUV was already stopped"},
        {"id": "W1-C2", "action": "approach", "vehicle_desc": "white SUV", "original_text": "white SUV entered from the right"},
    ]
    sensors = []
    videos = [
        {"id": "V-02", "event_type": "approach", "vehicle_ref": "white_SUV", "track_ids": [2], "start_time": 1.6, "end_time": 2.3}
    ]

    matcher = EvidenceMatcher()
    links = matcher.match_evidence(claims, sensors, video_events=videos)

    contradiction = next(l for l in links if l.status == LinkStatus.CONTRADICT.value)
    assert "W2-C1" in contradiction.linked_ids
    assert "V-02" in contradiction.linked_ids
    assert contradiction.confidence >= 0.90


def test_matcher_detects_support_and_gaps() -> None:
    claims = [
        {
            "id": "W1-C1",
            "action": "speeding",
            "original_text": "moving very fast",
            "vehicle_desc": "black SUV",
        },
        {
            "id": "W1-C2",
            "action": "approach",
            "original_text": "entered from the right",
            "vehicle_desc": "white SUV",
        },
    ]
    sensors = [
        {
            "id": "S-03",
            "event_type": "speed_drop",
            "time": 2.8,
            "value": 34.9,
            "unit": "km/h_drop",
        }
    ]

    matcher = EvidenceMatcher()
    links = matcher.match_evidence(claims, sensors)

    support_link = next(l for l in links if l.status == LinkStatus.SUPPORT.value)
    assert "W1-C1" in support_link.linked_ids
    assert support_link.confidence > 0.8

    gap_link = next(l for l in links if l.status == LinkStatus.GAP.value)
    assert gap_link.missing_info is not None


def test_matcher_end_to_end_files(tmp_path: Path) -> None:
    claims_path = ROOT / "witness_claims.json"
    sensors_path = ROOT / "sensor_events.json"
    videos_path = ROOT / "video_events.json"
    out_path = tmp_path / "evidence_links.json"

    links = generate_evidence_links(
        claims_path,
        sensors_path,
        video_events_path=videos_path,
        output_links_path=out_path,
    )
    assert len(links) >= 8
    assert out_path.exists()
    statuses = {l.status for l in links}
    assert "contradict" in statuses
    assert "support" in statuses
    assert "gap" in statuses
