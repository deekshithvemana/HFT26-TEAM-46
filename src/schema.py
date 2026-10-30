"""Unified Event and Evidence Schemas for Road Accident Forensics.

Conforms strictly to the 24-Hour Hackathon Data Contract:
- Observation: Direct empirical observation (e.g. video bounding box)
- Derived Measurement: Computed metric (e.g. sensor deceleration, TTC)
- Claim: Witness statement assertion
- Inference: Model reasoning or keyframe description
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class SourceType(str, Enum):
    VIDEO = "video"
    WITNESS = "witness"
    SENSOR = "sensor"


class EventKind(str, Enum):
    OBSERVATION = "observation"
    DERIVED_MEASUREMENT = "derived_measurement"
    CLAIM = "claim"
    INFERENCE = "inference"


class LinkStatus(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    GAP = "gap"


@dataclass
class CommonEvent:
    """Standardized event entry in the timeline."""
    id: str
    source_type: str  # "video" | "witness" | "sensor"
    event_type: str   # e.g., "hard_braking", "collision", "swerve", "stop"
    start_time: Optional[float]
    end_time: Optional[float]
    confidence: float
    kind: str         # "observation" | "derived_measurement" | "claim" | "inference"
    description: str
    time_uncertainty: Optional[float] = None
    evidence_ref: Optional[Any] = None
    vehicle_ref: Optional[str] = None
    target_vehicle: Optional[str] = None
    temporal_anchor: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WitnessClaim:
    """Extracted claim from a written witness account."""
    id: str
    statement_id: str
    original_text: str
    vehicle_desc: str
    action: str
    direction: Optional[str] = None
    target_vehicle: Optional[str] = None
    temporal_anchor: Optional[str] = None
    time_interval: Optional[list[float]] = None
    uncertainty: Optional[float] = None
    confidence: float = 0.7
    kind: str = EventKind.CLAIM.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_common_event(self) -> CommonEvent:
        start_t = self.time_interval[0] if (self.time_interval and len(self.time_interval) > 0) else None
        end_t = self.time_interval[1] if (self.time_interval and len(self.time_interval) > 1) else start_t
        return CommonEvent(
            id=self.id,
            source_type=SourceType.WITNESS.value,
            event_type=self.action.lower().replace(" ", "_"),
            start_time=start_t,
            end_time=end_t,
            time_uncertainty=self.uncertainty,
            confidence=self.confidence,
            kind=EventKind.CLAIM.value,
            evidence_ref={"statement_id": self.statement_id, "quote": self.original_text},
            vehicle_ref=self.vehicle_desc,
            target_vehicle=self.target_vehicle,
            temporal_anchor=self.temporal_anchor,
            description=f"Witness reports {self.vehicle_desc} {self.action} ({self.original_text})",
        )


@dataclass
class SensorEvent:
    """Critical kinetic event derived from sensor telemetry."""
    id: str
    event_type: str  # "hard_braking", "speed_drop", "vehicle_stop", "impact_spike"
    time: float
    value: float
    unit: str
    source_row: int
    confidence: float = 0.95
    kind: str = EventKind.DERIVED_MEASUREMENT.value
    vehicle_ref: Optional[str] = "ego_vehicle"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_common_event(self) -> CommonEvent:
        return CommonEvent(
            id=self.id,
            source_type=SourceType.SENSOR.value,
            event_type=self.event_type,
            start_time=self.time,
            end_time=self.time + 0.2,
            time_uncertainty=0.1,
            confidence=self.confidence,
            kind=self.kind,
            evidence_ref={"source_row": self.source_row, "measurement": f"{self.value} {self.unit}"},
            vehicle_ref=self.vehicle_ref,
            description=f"Sensor registered {self.event_type}: {self.value:.2f} {self.unit}",
        )


@dataclass
class EvidenceLink:
    """Cross-source evidence comparison and contradiction link."""
    id: str
    finding_id: str
    linked_ids: list[str]
    status: str       # "support" | "contradict" | "partial" | "unverified" | "gap"
    confidence: float
    explanation: str
    missing_info: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyIndicator:
    """Time-To-Collision (TTC) or DRAC safety indicator."""
    indicator: str    # "TTC" | "DRAC"
    vehicle_a: int
    vehicle_b: int
    time_s: float
    value: float
    unit: str
    warning_level: str  # "normal" | "caution" | "critical"
    confidence: float = 0.8
    kind: str = EventKind.DERIVED_MEASUREMENT.value
    caveats: str = "Estimated from 2D pixel bounding boxes; indicator only, never proof of collision."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json_records(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        return data if isinstance(data, list) else [data]


def save_json_records(records: list[Any] | dict[str, Any], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        item.to_dict() if hasattr(item, "to_dict") else item
        for item in records
    ] if isinstance(records, list) else records

    with p.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
