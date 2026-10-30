import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sensors.sensor_processor import SensorProcessor, process_sensor_file


def test_sensor_processor_detects_all_events(tmp_path: Path) -> None:
    csv_file = tmp_path / "telemetry.csv"
    csv_file.write_text(
        "time_s,speed_kmh,accel_long_mps2,accel_lat_mps2,brake_bar,steering_deg\n"
        "0.0,60.0,-0.2,0.1,0.0,0.0\n"
        "1.0,58.0,-0.5,0.2,5.0,0.5\n"
        "2.0,52.0,-4.5,0.3,35.0,1.0\n"  # hard braking starts
        "2.2,38.0,-6.5,0.4,48.0,1.5\n"  # peak braking, speed drop
        "2.4,20.0,-5.0,9.5,45.0,20.0\n"  # impact lateral spike
        "3.0,0.0,0.0,0.0,30.0,0.0\n"     # vehicle stop
        "3.5,0.0,0.0,0.0,30.0,0.0\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "sensor_events.json"
    events = process_sensor_file(csv_file, output_json=out_file, vehicle_ref="test_veh")

    assert len(events) >= 3
    event_types = [e.event_type for e in events]
    assert "hard_braking" in event_types
    assert "impact_spike" in event_types
    assert "speed_drop" in event_types
    assert "vehicle_stop" in event_types

    assert out_file.exists()
    braking = next(e for e in events if e.event_type == "hard_braking")
    assert braking.value <= -4.0
    assert braking.unit == "m/s^2"
    assert braking.kind == "derived_measurement"
    assert braking.confidence > 0.9

    stop = next(e for e in events if e.event_type == "vehicle_stop")
    assert stop.time == 3.0
    assert stop.value == 0.0


def test_sensor_processor_handles_empty_or_normal_driving(tmp_path: Path) -> None:
    csv_file = tmp_path / "normal.csv"
    csv_file.write_text(
        "time_s,speed_kmh,accel_long_mps2,accel_lat_mps2,brake_bar,steering_deg\n"
        "0.0,40.0,0.1,0.0,0.0,0.0\n"
        "1.0,40.5,0.0,0.1,0.0,0.1\n"
        "2.0,39.8,-0.1,0.0,0.0,-0.1\n",
        encoding="utf-8",
    )
    processor = SensorProcessor()
    events = processor.process_file(csv_file)
    assert len(events) == 0  # No anomalous critical kinetic events
