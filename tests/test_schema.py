import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schema import (
    CommonEvent,
    WitnessClaim,
    SensorEvent,
    EvidenceLink,
    EventKind,
    SourceType,
    LinkStatus,
    save_json_records,
    load_json_records,
)


def test_witness_claim_conversion_to_common_event() -> None:
    claim = WitnessClaim(
        id="W1-C1",
        statement_id="W1",
        original_text="The car didn't brake before impact",
        vehicle_desc="red sedan",
        action="no_braking",
        time_interval=[2.0, 4.0],
        uncertainty=1.0,
        confidence=0.75,
    )
    event = claim.to_common_event()
    assert event.id == "W1-C1"
    assert event.source_type == SourceType.WITNESS.value
    assert event.kind == EventKind.CLAIM.value
    assert event.start_time == 2.0
    assert event.end_time == 4.0
    assert event.confidence == 0.75

    # Test unstated timestamps (time_interval is None)
    claim_no_time = WitnessClaim(
        id="W1-C2",
        statement_id="W1",
        original_text="The black SUV rotated and stopped",
        vehicle_desc="black SUV",
        action="stopping",
        time_interval=None,
        uncertainty=None,
    )
    event_no_time = claim_no_time.to_common_event()
    assert event_no_time.start_time is None
    assert event_no_time.end_time is None
    assert event_no_time.time_uncertainty is None


def test_sensor_event_conversion_to_common_event() -> None:
    sensor = SensorEvent(
        id="S-04",
        event_type="hard_braking",
        time=3.5,
        value=-6.8,
        unit="m/s^2",
        source_row=42,
        confidence=0.98,
    )
    event = sensor.to_common_event()
    assert event.id == "S-04"
    assert event.source_type == SourceType.SENSOR.value
    assert event.kind == EventKind.DERIVED_MEASUREMENT.value
    assert event.start_time == 3.5
    assert event.confidence == 0.98


def test_save_and_load_records(tmp_path: Path) -> None:
    link = EvidenceLink(
        id="L-01",
        finding_id="F-01",
        linked_ids=["W1-C1", "S-04"],
        status=LinkStatus.CONTRADICT.value,
        confidence=0.89,
        explanation="Witness claims no braking but telemetry records -6.8 m/s^2 deceleration.",
    )
    target_path = tmp_path / "links.json"
    save_json_records([link], target_path)

    loaded = load_json_records(target_path)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "L-01"
    assert loaded[0]["status"] == "contradict"
