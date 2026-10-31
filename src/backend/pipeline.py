"""P3 Task 2 — Full Forensic Pipeline Runner & Precomputed Fallback.

Orchestrates multi-source forensic ingestion and reconstruction:
- P1: Video preprocessing, YOLO tracking, Deep Learning event recognition
- P2: Witness statement NLP claim extraction, telematics processing, TTC/DRAC safety metrics, evidence contradiction matcher
- P3: Master timestamp alignment, chronological timeline merge, and interactive forensic report generation

Provides two primary execution modes:
1. run_pipeline(): Full live processing across all neural models and parsers.
2. load_precomputed(): Instant fallback loading existing / mock artifacts for demos and testing.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Optional

from src.backend.timeline_merger import TimelineMerger
from src.intelligence.matcher import EvidenceMatcher
from src.nlp.witness_parser import WitnessParser
from src.reporting.report_generator import generate_report
from src.schema import load_json_records, save_json_records
from src.sensors.safety_indicators import compute_safety_indicators
from src.sensors.sensor_processor import SensorProcessor
from src.video.event_recognition import DeepTrafficEventRecognizer
from src.video.preprocess import preprocess_video
from src.video.track import track_video


class ForensicPipeline:
    """End-to-end multi-modal accident timeline reconstruction pipeline."""

    def __init__(self, default_output_base: Path | str = "data/processed") -> None:
        self.output_base = Path(default_output_base).resolve()
        self.output_base.mkdir(parents=True, exist_ok=True)

    def load_precomputed(
        self,
        incident_id: str = "CASE-2026-T46-01",
        precomputed_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ) -> dict[str, Any]:
        """Load precomputed mock or demo results instantly without re-running heavy models."""
        t_start = time.time()
        out_root = Path(output_dir) if output_dir else self.output_base / incident_id
        out_root.mkdir(parents=True, exist_ok=True)

        # Source candidate search paths
        candidate_dirs = [
            Path(precomputed_dir) if precomputed_dir else None,
            Path("mock"),
            Path("data/processed/tbone_demo"),
            Path("."),
        ]

        def _find_file(filename: str) -> Optional[Path]:
            for cd in candidate_dirs:
                if cd and cd.exists():
                    p = cd / filename
                    if p.exists():
                        return p
            return None

        tracks_file = _find_file("tracks.json")
        video_events_file = _find_file("video_events.json")
        claims_file = _find_file("witness_claims.json")
        sensor_file = _find_file("sensor_events.json")
        links_file = _find_file("evidence_links.json")
        safety_file = _find_file("safety_indicators.json")

        tracks_data = json.loads(tracks_file.read_text(encoding="utf-8")) if tracks_file else {}
        video_events = load_json_records(video_events_file) if video_events_file else []
        claims = load_json_records(claims_file) if claims_file else []
        sensor_events = load_json_records(sensor_file) if sensor_file else []
        links = load_json_records(links_file) if links_file else []
        safety = load_json_records(safety_file) if safety_file else []

        # If safety indicators are missing, calculate them from tracks
        if not safety and tracks_file:
            safety_objs = compute_safety_indicators(tracks_file, output_path=out_root / "safety_indicators.json")
            safety = [s.to_dict() for s in safety_objs]

        # If links are missing, run matcher
        if not links:
            matcher = EvidenceMatcher()
            link_objs = matcher.match_evidence(claims, sensor_events, video_events=video_events)
            links = [l.to_dict() for l in link_objs]

        # Merge timeline
        merger = TimelineMerger(time_zero_offset_s=0.0)
        timeline_path = out_root / "timeline.json"
        timeline = merger.merge(
            incident_id=incident_id,
            video_events=video_events,
            sensor_events=sensor_events,
            witness_claims=claims,
            evidence_links=links,
            safety_indicators=safety,
            output_path=timeline_path,
        )

        # Save artifacts to out_root
        save_json_records(video_events, out_root / "video_events.json")
        save_json_records(claims, out_root / "witness_claims.json")
        save_json_records(sensor_events, out_root / "sensor_events.json")
        save_json_records(links, out_root / "evidence_links.json")
        save_json_records(safety, out_root / "safety_indicators.json")
        if tracks_data:
            (out_root / "tracks.json").write_text(json.dumps(tracks_data, indent=2), encoding="utf-8")

        # Generate report
        report_path = out_root / "report.html"
        generate_report(
            witness_path=out_root / "witness_claims.json",
            sensor_path=out_root / "sensor_events.json",
            video_path=out_root / "video_events.json",
            links_path=out_root / "evidence_links.json",
            safety_path=out_root / "safety_indicators.json",
            keyframes_dir="data/processed/tbone_demo/keyframes" if Path("data/processed/tbone_demo/keyframes").exists() else None,
            output_path=report_path,
            case_id=incident_id,
        )

        elapsed = round(time.time() - t_start, 3)
        return {
            "status": "success",
            "mode": "precomputed",
            "incident_id": incident_id,
            "elapsed_seconds": elapsed,
            "output_dir": str(out_root.resolve()),
            "artifacts": {
                "timeline_json": str(timeline_path.resolve()),
                "tracks_json": str((out_root / "tracks.json").resolve()) if tracks_file else None,
                "video_events_json": str((out_root / "video_events.json").resolve()),
                "witness_claims_json": str((out_root / "witness_claims.json").resolve()),
                "sensor_events_json": str((out_root / "sensor_events.json").resolve()),
                "evidence_links_json": str((out_root / "evidence_links.json").resolve()),
                "safety_indicators_json": str((out_root / "safety_indicators.json").resolve()),
                "report_html": str(report_path.resolve()),
            },
            "summary": timeline.get("summary", {}),
            "timeline": timeline,
        }

    def run_pipeline(
        self,
        incident_id: str = "CASE-2026-T46-01",
        video_path: Optional[str | Path] = None,
        witness_statement: Optional[str | Path] = None,
        sensor_csv: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        skip_video_cv: bool = False,
    ) -> dict[str, Any]:
        """Run the multi-source processing pipeline on supplied evidence."""
        t_start = time.time()
        out_root = Path(output_dir) if output_dir else self.output_base / incident_id
        out_root.mkdir(parents=True, exist_ok=True)

        video_events: list[dict[str, Any]] = []
        tracks_data: dict[str, Any] = {}
        safety_indicators: list[dict[str, Any]] = []

        # -------------------------------------------------------------
        # 1. Person 1 — Computer Vision & Deep Learning Event Recognition
        # -------------------------------------------------------------
        if video_path and Path(video_path).exists() and not skip_video_cv:
            v_p = Path(video_path)
            # 1a. Preprocessing
            preprocess_video(v_p, output_dir=out_root, sample_fps=5.0)

            # 1b. YOLO Tracking
            track_res = track_video(v_p, output_dir=out_root, write_annotated=True)
            tracks_data = track_res

            # 1c. Deep Learning Event Recognition
            recognizer = DeepTrafficEventRecognizer()
            rec_res = recognizer.process_video(
                v_p,
                tracks_path=out_root / "tracks.json",
                output_dir=out_root,
            )
            video_events = rec_res.get("events", [])

            # 1d. Safety Indicators (TTC & DRAC)
            safety_objs = compute_safety_indicators(
                out_root / "tracks.json",
                output_path=out_root / "safety_indicators.json",
            )
            safety_indicators = [s.to_dict() for s in safety_objs]
        else:
            # Fallback to existing or mock video tracks and events
            if (out_root / "video_events.json").exists():
                video_events = load_json_records(out_root / "video_events.json")
            elif Path("mock/video_events.json").exists():
                video_events = load_json_records("mock/video_events.json")

            if (out_root / "safety_indicators.json").exists():
                safety_indicators = load_json_records(out_root / "safety_indicators.json")
            elif Path("mock/safety_indicators.json").exists():
                safety_indicators = load_json_records("mock/safety_indicators.json")

        # -------------------------------------------------------------
        # 2. Person 2 — Witness NLP Claim Extraction
        # -------------------------------------------------------------
        claims: list[dict[str, Any]] = []
        witness_parser = WitnessParser()
        if witness_statement:
            w_p = Path(witness_statement)
            if w_p.exists() and w_p.is_file():
                claim_objs = witness_parser.process_file(
                    w_p,
                    output_path=out_root / "witness_claims.json",
                    statement_id="W1",
                )
            else:
                claim_objs = witness_parser.parse_statement(
                    str(witness_statement),
                    statement_id="W1",
                )
                save_json_records(claim_objs, out_root / "witness_claims.json")
            claims = [c.to_dict() for c in claim_objs]
        else:
            if Path("mock/witness_claims.json").exists():
                claims = load_json_records("mock/witness_claims.json")
                save_json_records(claims, out_root / "witness_claims.json")

        # -------------------------------------------------------------
        # 3. Person 2 — Vehicle Sensor Telematics Processing
        # -------------------------------------------------------------
        sensor_events: list[dict[str, Any]] = []
        if sensor_csv and Path(sensor_csv).exists():
            processor = SensorProcessor()
            sensor_objs = processor.process_file(
                sensor_csv,
                output_json=out_root / "sensor_events.json",
            )
            sensor_events = [s.to_dict() for s in sensor_objs]
        else:
            if Path("mock/sensor_events.json").exists():
                sensor_events = load_json_records("mock/sensor_events.json")
                save_json_records(sensor_events, out_root / "sensor_events.json")

        # -------------------------------------------------------------
        # 4. Person 2 — Multi-Source Evidence Matcher & Contradictions
        # -------------------------------------------------------------
        matcher = EvidenceMatcher()
        link_objs = matcher.match_evidence(claims, sensor_events, video_events=video_events)
        links = [l.to_dict() for l in link_objs]
        save_json_records(links, out_root / "evidence_links.json")

        # -------------------------------------------------------------
        # 5. Person 3 — Master Timestamp Alignment & Timeline Merge
        # -------------------------------------------------------------
        merger = TimelineMerger(time_zero_offset_s=0.0)
        timeline_path = out_root / "timeline.json"
        timeline = merger.merge(
            incident_id=incident_id,
            video_events=video_events,
            sensor_events=sensor_events,
            witness_claims=claims,
            evidence_links=links,
            safety_indicators=safety_indicators,
            output_path=timeline_path,
        )

        # -------------------------------------------------------------
        # 6. Person 2/3 — Forensic Report Generation
        # -------------------------------------------------------------
        report_path = out_root / "report.html"
        kf_dir = out_root / "keyframes" if (out_root / "keyframes").exists() else Path("data/processed/tbone_demo/keyframes")
        generate_report(
            witness_path=out_root / "witness_claims.json",
            sensor_path=out_root / "sensor_events.json",
            video_path=out_root / "video_events.json",
            links_path=out_root / "evidence_links.json",
            safety_path=out_root / "safety_indicators.json",
            keyframes_dir=kf_dir if kf_dir.exists() else None,
            output_path=report_path,
            case_id=incident_id,
        )

        elapsed = round(time.time() - t_start, 3)
        return {
            "status": "success",
            "mode": "live_pipeline",
            "incident_id": incident_id,
            "elapsed_seconds": elapsed,
            "output_dir": str(out_root.resolve()),
            "artifacts": {
                "timeline_json": str(timeline_path.resolve()),
                "video_events_json": str((out_root / "video_events.json").resolve()),
                "witness_claims_json": str((out_root / "witness_claims.json").resolve()),
                "sensor_events_json": str((out_root / "sensor_events.json").resolve()),
                "evidence_links_json": str((out_root / "evidence_links.json").resolve()),
                "safety_indicators_json": str((out_root / "safety_indicators.json").resolve()),
                "report_html": str(report_path.resolve()),
            },
            "summary": timeline.get("summary", {}),
            "timeline": timeline,
        }
