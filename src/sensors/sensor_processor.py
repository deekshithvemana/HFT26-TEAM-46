"""P2 Task 3 — Sensor CSV/JSON processing.

Parses vehicle telemetry / CAN bus / accelerometer records to identify critical kinetic events:
- hard_braking (deceleration spike or high brake pressure)
- speed_drop (rapid kinetic energy loss)
- impact_spike (extreme lateral or longitudinal g-force shock)
- vehicle_stop (vehicle coming to a complete halt)

Exports sensor_events.json strictly following the Hackathon Data Contract.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from src.schema import SensorEvent, save_json_records


class SensorProcessor:
    """Processes vehicular sensor telemetry and derives key kinetic events."""

    def __init__(
        self,
        hard_braking_accel_thresh: float = -4.0,  # m/s^2 (negative for braking)
        hard_braking_pressure_thresh: float = 30.0,  # bar
        speed_drop_kmh_thresh: float = 15.0,  # drop threshold within 0.6s
        impact_lateral_thresh: float = 8.0,  # m/s^2 lateral shock
        vehicle_ref: str = "black_SUV",
    ) -> None:
        self.hard_braking_accel_thresh = hard_braking_accel_thresh
        self.hard_braking_pressure_thresh = hard_braking_pressure_thresh
        self.speed_drop_kmh_thresh = speed_drop_kmh_thresh
        self.impact_lateral_thresh = impact_lateral_thresh
        self.vehicle_ref = vehicle_ref

    def parse_csv(self, file_path: Path | str) -> list[dict[str, Any]]:
        """Read telemetry CSV and normalize column names."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Telemetry CSV not found: {p}")

        rows: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, r in enumerate(reader, start=2):  # row 1 is header
                try:
                    time_s = float(r.get("time_s") or r.get("timestamp") or r.get("time") or 0.0)
                    speed_kmh = float(r.get("speed_kmh") or r.get("speed") or 0.0)
                    accel_long = float(r.get("accel_long_mps2") or r.get("accel_x") or r.get("ax") or 0.0)
                    accel_lat = float(r.get("accel_lat_mps2") or r.get("accel_y") or r.get("ay") or 0.0)
                    brake_bar = float(r.get("brake_bar") or r.get("brake_pressure") or r.get("brake") or 0.0)
                    steering = float(r.get("steering_deg") or r.get("steering") or 0.0)

                    rows.append({
                        "source_row": idx,
                        "time_s": time_s,
                        "speed_kmh": speed_kmh,
                        "accel_long_mps2": accel_long,
                        "accel_lat_mps2": accel_lat,
                        "brake_bar": brake_bar,
                        "steering_deg": steering,
                    })
                except (ValueError, TypeError):
                    continue
        return rows

    def detect_events(self, telemetry_rows: list[dict[str, Any]]) -> list[SensorEvent]:
        """Detect critical kinetic milestones from telemetry timeseries."""
        if not telemetry_rows:
            return []

        events: list[SensorEvent] = []
        event_counter = 1

        # 1. Hard Braking Detection (Cluster continuous braking)
        braking_episodes: list[dict[str, Any]] = []
        current_braking: list[dict[str, Any]] = []

        for row in telemetry_rows:
            is_braking = (
                row["accel_long_mps2"] <= self.hard_braking_accel_thresh
                or row["brake_bar"] >= self.hard_braking_pressure_thresh
            )
            if is_braking:
                current_braking.append(row)
            else:
                if current_braking:
                    braking_episodes.append(list(current_braking))
                    current_braking.clear()
        if current_braking:
            braking_episodes.append(current_braking)

        for episode in braking_episodes:
            # Pick peak deceleration
            peak_row = min(episode, key=lambda r: r["accel_long_mps2"])
            events.append(
                SensorEvent(
                    id=f"S-{event_counter:02d}",
                    event_type="hard_braking",
                    time=round(peak_row["time_s"], 2),
                    value=round(peak_row["accel_long_mps2"], 2),
                    unit="m/s^2",
                    source_row=peak_row["source_row"],
                    confidence=0.96,
                    kind="derived_measurement",
                    vehicle_ref=self.vehicle_ref,
                )
            )
            event_counter += 1

        # 2. Impact Spike Detection (Extreme lateral or longitudinal shock)
        impact_rows = [
            r for r in telemetry_rows
            if abs(r["accel_lat_mps2"]) >= self.impact_lateral_thresh
        ]
        if impact_rows:
            peak_impact = max(impact_rows, key=lambda r: abs(r["accel_lat_mps2"]))
            events.append(
                SensorEvent(
                    id=f"S-{event_counter:02d}",
                    event_type="impact_spike",
                    time=round(peak_impact["time_s"], 2),
                    value=round(peak_impact["accel_lat_mps2"], 2),
                    unit="m/s^2",
                    source_row=peak_impact["source_row"],
                    confidence=0.98,
                    kind="derived_measurement",
                    vehicle_ref=self.vehicle_ref,
                )
            )
            event_counter += 1

        # 3. Speed Drop Detection (Check rolling drop over 3 samples / ~0.6s)
        window_size = 3
        max_drop = 0.0
        max_drop_row = None
        for i in range(len(telemetry_rows) - window_size):
            start = telemetry_rows[i]
            end = telemetry_rows[i + window_size]
            drop = start["speed_kmh"] - end["speed_kmh"]
            if drop >= self.speed_drop_kmh_thresh and drop > max_drop:
                max_drop = drop
                max_drop_row = end

        if max_drop_row is not None:
            events.append(
                SensorEvent(
                    id=f"S-{event_counter:02d}",
                    event_type="speed_drop",
                    time=round(max_drop_row["time_s"], 2),
                    value=round(max_drop, 1),
                    unit="km/h_drop",
                    source_row=max_drop_row["source_row"],
                    confidence=0.94,
                    kind="derived_measurement",
                    vehicle_ref=self.vehicle_ref,
                )
            )
            event_counter += 1

        # 4. Vehicle Stop Detection (First point vehicle reaches speed == 0.0)
        stop_rows = [r for r in telemetry_rows if r["speed_kmh"] <= 0.1]
        if stop_rows:
            first_stop = stop_rows[0]
            events.append(
                SensorEvent(
                    id=f"S-{event_counter:02d}",
                    event_type="vehicle_stop",
                    time=round(first_stop["time_s"], 2),
                    value=0.0,
                    unit="km/h",
                    source_row=first_stop["source_row"],
                    confidence=0.99,
                    kind="derived_measurement",
                    vehicle_ref=self.vehicle_ref,
                )
            )
            event_counter += 1

        # Sort chronologically by timestamp
        events.sort(key=lambda e: e.time)
        return events

    def process_file(
        self,
        input_csv: Path | str,
        output_json: Optional[Path | str] = None,
    ) -> list[SensorEvent]:
        """Load CSV, derive kinetic events, and save output JSON."""
        rows = self.parse_csv(input_csv)
        events = self.detect_events(rows)
        if output_json:
            save_json_records(events, output_json)
        return events


def process_sensor_file(
    input_csv: Path | str,
    output_json: Optional[Path | str] = None,
    vehicle_ref: str = "black_SUV",
) -> list[SensorEvent]:
    """Convenience function to parse a sensor CSV into SensorEvent records."""
    processor = SensorProcessor(vehicle_ref=vehicle_ref)
    return processor.process_file(input_csv, output_json=output_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vehicle Telematics Sensor Processor")
    parser.add_argument("csv_path", type=Path, help="Path to telemetry CSV file")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (e.g. sensor_events.json)")
    parser.add_argument("--vehicle", type=str, default="black_SUV", help="Vehicle reference identifier")
    args = parser.parse_args()

    events = process_sensor_file(args.csv_path, output_json=args.out, vehicle_ref=args.vehicle)
    print(json.dumps([e.to_dict() for e in events], indent=2))


if __name__ == "__main__":
    main()
