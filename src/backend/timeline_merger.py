"""P3 Tasks 3 & 4 — Timestamp Alignment and Multi-Modal Timeline Merge.

Converts multi-source events (Video, Sensors, Witness Claims, Safety Indicators):
1. Normalizes all timestamps relative to incident start clock (t=0).
2. Retains time uncertainty intervals (e.g. ±0.1s or [t_start, t_end]).
3. Strictly adheres to Epistemic Classification:
   - observation (direct visual detections/bounding boxes)
   - derived_measurement (sensor telemetry, calculated deceleration, TTC/DRAC)
   - claim (witness reported statements)
   - inference (deep learning predictions, VLM captions, contradiction links)
4. Groups concurrent and related events into incident phases / clusters (Pre-impact, Collision, Post-impact).
5. Retains cross-modal evidence links and separate undated contextual claims.
6. Exports unified timeline.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional

from src.schema import (
    CommonEvent,
    EventKind,
    EvidenceLink,
    SafetyIndicator,
    SensorEvent,
    SourceType,
    WitnessClaim,
    save_json_records,
)


class TimelineMerger:
    """Merges and aligns multi-source forensic evidence into a chronological master timeline."""

    def __init__(self, time_zero_offset_s: float = 0.0) -> None:
        self.time_zero_offset_s = time_zero_offset_s

    def align_timestamp(self, t: Optional[float]) -> Optional[float]:
        """Align timestamp to the master incident clock."""
        if t is None:
            return None
        return round(float(t) + self.time_zero_offset_s, 3)

    def process_video_events(
        self, video_events: list[dict[str, Any]]
    ) -> list[CommonEvent]:
        """Convert video events into standardized CommonEvents."""
        events: list[CommonEvent] = []
        for ve in video_events:
            start_t = self.align_timestamp(ve.get("start_time"))
            end_t = self.align_timestamp(ve.get("end_time", start_t))
            raw_kind = ve.get("kind", EventKind.OBSERVATION.value)

            # Collision is direct visual observation; braking/swerve are derived measurements or observations
            kind = raw_kind if raw_kind in [e.value for e in EventKind] else EventKind.OBSERVATION.value

            ev = CommonEvent(
                id=str(ve.get("id", f"V-{len(events)+1:02d}")),
                source_type=SourceType.VIDEO.value,
                event_type=str(ve.get("event_type", "video_event")),
                start_time=start_t,
                end_time=end_t,
                time_uncertainty=ve.get("time_uncertainty", 0.08),
                confidence=float(ve.get("confidence", 0.85)),
                kind=kind,
                description=str(ve.get("description", f"Visual observation: {ve.get('event_type')}")),
                evidence_ref=ve.get("evidence_ref"),
                vehicle_ref=str(ve.get("track_ids", [])) if ve.get("track_ids") else None,
                metadata={
                    "frame_start": ve.get("frame_start"),
                    "frame_end": ve.get("frame_end"),
                    "keyframe_frame": ve.get("keyframe_frame"),
                    "track_ids": ve.get("track_ids", []),
                },
            )
            events.append(ev)
        return events

    def process_sensor_events(
        self, sensor_events: list[dict[str, Any]]
    ) -> list[CommonEvent]:
        """Convert sensor telemetry events into standardized CommonEvents."""
        events: list[CommonEvent] = []
        for se in sensor_events:
            t = self.align_timestamp(se.get("time", 0.0))
            ev = CommonEvent(
                id=str(se.get("id", f"S-{len(events)+1:02d}")),
                source_type=SourceType.SENSOR.value,
                event_type=str(se.get("event_type", "sensor_event")),
                start_time=t,
                end_time=t + 0.15 if t is not None else None,
                time_uncertainty=0.05,
                confidence=float(se.get("confidence", 0.95)),
                kind=EventKind.DERIVED_MEASUREMENT.value,
                description=f"Sensor registered {se.get('event_type')}: {se.get('value')} {se.get('unit')}",
                evidence_ref={"source_row": se.get("source_row"), "value": se.get("value"), "unit": se.get("unit")},
                vehicle_ref=str(se.get("vehicle_ref", "ego_vehicle")),
                metadata={
                    "source_row": se.get("source_row"),
                    "value": se.get("value"),
                    "unit": se.get("unit"),
                },
            )
            events.append(ev)
        return events

    def process_witness_claims(
        self, witness_claims: list[dict[str, Any]]
    ) -> tuple[list[CommonEvent], list[dict[str, Any]]]:
        """Convert dated witness claims into CommonEvents and collect undated claims."""
        dated_events: list[CommonEvent] = []
        undated_claims: list[dict[str, Any]] = []

        for wc in witness_claims:
            time_interval = wc.get("time_interval")
            if time_interval and len(time_interval) > 0 and time_interval[0] is not None:
                start_t = self.align_timestamp(time_interval[0])
                end_t = (
                    self.align_timestamp(time_interval[1])
                    if len(time_interval) > 1 and time_interval[1] is not None
                    else start_t
                )
                ev = CommonEvent(
                    id=str(wc.get("id", f"W-{len(dated_events)+1:02d}")),
                    source_type=SourceType.WITNESS.value,
                    event_type=str(wc.get("action", "statement")).lower().replace(" ", "_"),
                    start_time=start_t,
                    end_time=end_t,
                    time_uncertainty=wc.get("uncertainty", 0.5),
                    confidence=float(wc.get("confidence", 0.70)),
                    kind=EventKind.CLAIM.value,
                    description=f"Witness reports {wc.get('vehicle_desc')} {wc.get('action')} — \"{wc.get('original_text')}\"",
                    evidence_ref={"statement_id": wc.get("statement_id"), "quote": wc.get("original_text")},
                    vehicle_ref=str(wc.get("vehicle_desc")),
                    target_vehicle=wc.get("target_vehicle"),
                    temporal_anchor=wc.get("temporal_anchor"),
                    metadata={
                        "statement_id": wc.get("statement_id"),
                        "original_text": wc.get("original_text"),
                        "direction": wc.get("direction"),
                    },
                )
                dated_events.append(ev)
            else:
                undated_claims.append(wc)

        return dated_events, undated_claims

    def cluster_events(self, events: list[CommonEvent]) -> list[dict[str, Any]]:
        """Group temporally proximate events across modalities into incident phases."""
        if not events:
            return []

        clusters: list[dict[str, Any]] = []
        current_cluster_events: list[CommonEvent] = []
        cluster_window_s = 1.0

        for ev in events:
            if not current_cluster_events:
                current_cluster_events.append(ev)
                continue

            last_ev = current_cluster_events[-1]
            t_curr = ev.start_time if ev.start_time is not None else 0.0
            t_last = last_ev.start_time if last_ev.start_time is not None else 0.0

            if abs(t_curr - t_last) <= cluster_window_s:
                current_cluster_events.append(ev)
            else:
                clusters.append(self._build_cluster(current_cluster_events, len(clusters) + 1))
                current_cluster_events = [ev]

        if current_cluster_events:
            clusters.append(self._build_cluster(current_cluster_events, len(clusters) + 1))

        return clusters

    @staticmethod
    def _build_cluster(events: list[CommonEvent], cluster_idx: int) -> dict[str, Any]:
        valid_starts = [e.start_time for e in events if e.start_time is not None]
        valid_ends = [e.end_time for e in events if e.end_time is not None]
        t_min = min(valid_starts) if valid_starts else 0.0
        t_max = max(valid_ends) if valid_ends else t_min

        event_types = set(e.event_type for e in events)
        sources = set(e.source_type for e in events)

        # Categorize incident phase
        if any("collision" in et for et in event_types) or any("impact" in et for et in event_types):
            phase = "Impact Phase"
        elif any("braking" in et for et in event_types) or any("approach" in et for et in event_types):
            phase = "Pre-Impact Approach / Braking"
        elif any("stop" in et for et in event_types) or any("post" in et for et in event_types):
            phase = "Post-Impact Resting Phase"
        else:
            phase = "Transit Phase"

        return {
            "cluster_id": f"CL-{cluster_idx:02d}",
            "phase": phase,
            "start_time": round(t_min, 3),
            "end_time": round(t_max, 3),
            "duration_s": round(t_max - t_min, 3),
            "event_count": len(events),
            "modalities": list(sources),
            "event_ids": [e.id for e in events],
            "primary_types": list(event_types),
        }

    def merge(
        self,
        incident_id: str,
        video_events: list[dict[str, Any]],
        sensor_events: list[dict[str, Any]],
        witness_claims: list[dict[str, Any]],
        evidence_links: Optional[list[dict[str, Any]]] = None,
        safety_indicators: Optional[list[dict[str, Any]]] = None,
        output_path: Optional[Path | str] = None,
    ) -> dict[str, Any]:
        """Merge all sources into the unified timeline."""
        # 1. Process sources
        v_events = self.process_video_events(video_events)
        s_events = self.process_sensor_events(sensor_events)
        w_events, undated_claims = self.process_witness_claims(witness_claims)

        # 2. Combine and sort chronologically
        all_dated = v_events + s_events + w_events
        all_dated.sort(
            key=lambda e: (
                e.start_time if e.start_time is not None else 999.0,
                0 if e.kind == EventKind.OBSERVATION.value else 1,
            )
        )

        # 3. Form incident clusters
        clusters = self.cluster_events(all_dated)

        # 4. Summary counts
        kinds = [e.kind for e in all_dated]
        sources = [e.source_type for e in all_dated]
        links = evidence_links or []
        contras = [l for l in links if l.get("status") == "contradict"]
        supports = [l for l in links if l.get("status") == "support"]

        now_iso = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "incident_id": incident_id,
            "created_at": now_iso,
            "time_zero_offset_s": self.time_zero_offset_s,
            "time_convention": "All timestamps represent seconds elapsed from video start (t=0).",
            "total_events": len(all_dated),
            "total_undated_claims": len(undated_claims),
            "summary": {
                "observations": kinds.count(EventKind.OBSERVATION.value),
                "derived_measurements": kinds.count(EventKind.DERIVED_MEASUREMENT.value),
                "claims": kinds.count(EventKind.CLAIM.value),
                "inferences": kinds.count(EventKind.INFERENCE.value),
                "sources": {
                    "video": sources.count(SourceType.VIDEO.value),
                    "sensor": sources.count(SourceType.SENSOR.value),
                    "witness": sources.count(SourceType.WITNESS.value),
                },
                "contradictions_flagged": len(contras),
                "corroborations_flagged": len(supports),
            },
            "clusters": clusters,
            "events": [e.to_dict() for e in all_dated],
            "undated_claims": undated_claims,
            "evidence_links": links,
            "safety_indicators": safety_indicators or [],
        }

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return payload


def merge_and_align_timeline(
    incident_id: str = "CASE-2026-T46-01",
    video_events_path: Optional[Path | str] = "video_events.json",
    sensor_events_path: Optional[Path | str] = "sensor_events.json",
    witness_claims_path: Optional[Path | str] = "witness_claims.json",
    evidence_links_path: Optional[Path | str] = "evidence_links.json",
    safety_indicators_path: Optional[Path | str] = "safety_indicators.json",
    output_path: Optional[Path | str] = "timeline.json",
    time_zero_offset_s: float = 0.0,
) -> dict[str, Any]:
    """Helper to merge JSON files from disk."""
    def _read_json(p_in: Optional[Path | str], mock_fallback: str) -> list[dict[str, Any]]:
        p = Path(p_in) if p_in and Path(p_in).exists() else Path(mock_fallback)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else [data]
        except Exception:
            return []

    v_events = _read_json(video_events_path, "mock/video_events.json")
    s_events = _read_json(sensor_events_path, "mock/sensor_events.json")
    w_claims = _read_json(witness_claims_path, "mock/witness_claims.json")
    e_links = _read_json(evidence_links_path, "mock/evidence_links.json")
    s_indicators = _read_json(safety_indicators_path, "mock/safety_indicators.json")

    merger = TimelineMerger(time_zero_offset_s=time_zero_offset_s)
    return merger.merge(
        incident_id=incident_id,
        video_events=v_events,
        sensor_events=s_events,
        witness_claims=w_claims,
        evidence_links=e_links,
        safety_indicators=s_indicators,
        output_path=output_path,
    )
