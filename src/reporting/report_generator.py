"""P2 Task 7 — Forensic Report Generation (HTML & PDF printable).

Compiles a unified forensic incident reconstruction report incorporating:
- Multi-source chronological timeline (Video, Sensor Telematics, Witness Claims)
- Epistemic classification badges (observation, derived_measurement, claim, inference)
- Cross-modal contradiction & corroboration matrix
- Surrogate safety metrics (TTC & DRAC)
- Keyframe evidence visual previews
- Legal neutrality disclaimers & uncertainty metrics

Conforms to the 24-Hour Hackathon Data Contract.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
from typing import Any, Optional


class ForensicReportGenerator:
    """Generates standalone, self-contained forensic reconstruction HTML reports."""

    def __init__(self, title: str = "Forensic Incident Reconstruction Report") -> None:
        self.title = title

    @staticmethod
    def _load_json(file_path: Optional[Path | str]) -> list[dict[str, Any]]:
        if not file_path:
            return []
        p = Path(file_path)
        if not p.exists():
            return []
        try:
            with p.open("r", encoding="utf-8") as h:
                data = json.load(h)
                return data if isinstance(data, list) else [data]
        except Exception:
            return []

    @staticmethod
    def _image_to_base64(img_path: Path | str) -> Optional[str]:
        p = Path(img_path)
        if not p.exists():
            return None
        try:
            data = p.read_bytes()
            mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return None

    def build_report_html(
        self,
        witness_claims: list[dict[str, Any]],
        sensor_events: list[dict[str, Any]],
        video_events: list[dict[str, Any]],
        evidence_links: list[dict[str, Any]],
        safety_indicators: list[dict[str, Any]],
        keyframes_dir: Optional[Path | str] = None,
        case_id: str = "CASE-2026-T46-01",
    ) -> str:
        """Construct a complete, self-contained interactive HTML document."""

        # 1. Calculate KPI Metrics
        total_claims = len(witness_claims)
        total_sensor = len(sensor_events)
        total_video = len(video_events)
        total_events = total_claims + total_sensor + total_video

        contradictions = [lnk for lnk in evidence_links if lnk.get("status") == "contradict"]
        supports = [lnk for lnk in evidence_links if lnk.get("status") == "support"]
        gaps = [lnk for lnk in evidence_links if lnk.get("status") == "gap"]

        ttc_values = [
            ind["value"] for ind in safety_indicators
            if ind.get("indicator") == "TTC" and isinstance(ind.get("value"), (int, float)) and ind["value"] > 0
        ]
        min_ttc = min(ttc_values) if ttc_values else None

        drac_values = [
            ind["value"] for ind in safety_indicators
            if ind.get("indicator") == "DRAC" and isinstance(ind.get("value"), (int, float))
        ]
        max_drac = max(drac_values) if drac_values else None

        # 2. Build Chronological Timeline Items
        timeline_items: list[dict[str, Any]] = []

        # Video events
        for ve in video_events:
            start_t = ve.get("start_time", 0.0)
            end_t = ve.get("end_time", start_t)
            timeline_items.append({
                "time": start_t,
                "end_time": end_t,
                "source": "Video Footage",
                "source_type": "video",
                "kind": ve.get("kind", "observation"),
                "event_type": ve.get("event_type", "video_event"),
                "desc": ve.get("description", f"Visual observation: {ve.get('event_type')}"),
                "conf": ve.get("confidence", 0.9),
                "id": ve.get("id", "VE-?"),
                "uncertainty": ve.get("time_uncertainty", 0.1),
            })

        # Sensor events
        for se in sensor_events:
            t = se.get("time", 0.0)
            timeline_items.append({
                "time": t,
                "end_time": t + 0.2,
                "source": "Sensor Telematics",
                "source_type": "sensor",
                "kind": se.get("kind", "derived_measurement"),
                "event_type": se.get("event_type", "sensor_event"),
                "desc": f"Registered {se.get('event_type')}: {se.get('value')} {se.get('unit')}",
                "conf": se.get("confidence", 0.95),
                "id": se.get("id", "SE-?"),
                "uncertainty": 0.05,
            })

        # Witness claims with timestamps
        undated_claims: list[dict[str, Any]] = []
        for wc in witness_claims:
            time_interval = wc.get("time_interval")
            if time_interval and len(time_interval) > 0 and time_interval[0] is not None:
                start_t = time_interval[0]
                end_t = time_interval[1] if len(time_interval) > 1 else start_t
                timeline_items.append({
                    "time": start_t,
                    "end_time": end_t,
                    "source": "Witness Statement",
                    "source_type": "witness",
                    "kind": wc.get("kind", "claim"),
                    "event_type": wc.get("action", "statement"),
                    "desc": f"{wc.get('vehicle_desc')} {wc.get('action')} — \"{wc.get('original_text')}\"",
                    "conf": wc.get("confidence", 0.7),
                    "id": wc.get("id", "WC-?"),
                    "uncertainty": wc.get("uncertainty", 0.5),
                })
            else:
                undated_claims.append(wc)

        # Sort chronological items by timestamp
        timeline_items.sort(key=lambda x: (x["time"] if x["time"] is not None else 999.0))

        # 3. Load Keyframes
        keyframe_cards = []
        if keyframes_dir:
            kd = Path(keyframes_dir)
            if kd.exists():
                for img_file in sorted(kd.glob("*.jpg")):
                    b64_uri = self._image_to_base64(img_file)
                    if b64_uri:
                        label = img_file.stem.replace("_", " ").title()
                        keyframe_cards.append({"title": label, "src": b64_uri})

        # Render HTML
        return self._render_template(
            case_id=case_id,
            total_events=total_events,
            total_claims=total_claims,
            total_sensor=total_sensor,
            total_video=total_video,
            contradictions_count=len(contradictions),
            supports_count=len(supports),
            gaps_count=len(gaps),
            min_ttc=min_ttc,
            max_drac=max_drac,
            timeline_items=timeline_items,
            undated_claims=undated_claims,
            evidence_links=evidence_links,
            safety_indicators=safety_indicators,
            keyframe_cards=keyframe_cards,
        )

    def _render_template(
        self,
        case_id: str,
        total_events: int,
        total_claims: int,
        total_sensor: int,
        total_video: int,
        contradictions_count: int,
        supports_count: int,
        gaps_count: int,
        min_ttc: Optional[float],
        max_drac: Optional[float],
        timeline_items: list[dict[str, Any]],
        undated_claims: list[dict[str, Any]],
        evidence_links: list[dict[str, Any]],
        safety_indicators: list[dict[str, Any]],
        keyframe_cards: list[dict[str, str]],
    ) -> str:
        """Assembles HTML string with embedded modern dark/light CSS."""

        # Timeline rows
        timeline_html = ""
        for item in timeline_items:
            kind_badge_class = f"badge-{item['kind']}"
            source_badge_class = f"badge-source-{item['source_type']}"
            t_str = f"t = {item['time']:.2f}s"
            if item.get("end_time") and item["end_time"] > item["time"]:
                t_str += f" – {item['end_time']:.2f}s"
            if item.get("uncertainty"):
                t_str += f" (±{item['uncertainty']:.2f}s)"

            conf_pct = int(item["conf"] * 100)

            timeline_html += f"""
            <div class="timeline-row">
                <div class="timeline-time">
                    <span class="time-stamp">{t_str}</span>
                </div>
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                    <div class="card-header">
                        <span class="badge {kind_badge_class}">{item['kind'].replace('_', ' ').upper()}</span>
                        <span class="badge {source_badge_class}">{item['source']}</span>
                        <span class="badge-id">{item['id']}</span>
                        <span class="conf-badge">Conf: {conf_pct}%</span>
                    </div>
                    <div class="card-body">
                        <div class="event-type">{html.escape(str(item['event_type'])).replace('_', ' ').title()}</div>
                        <p class="event-desc">{html.escape(str(item['desc']))}</p>
                    </div>
                </div>
            </div>
            """

        # Contradiction links
        links_html = ""
        for lnk in evidence_links:
            status = lnk.get("status", "unverified").lower()
            status_class = f"status-{status}"
            linked_ids = ", ".join(lnk.get("linked_ids", []))
            links_html += f"""
            <tr class="{status_class}-row">
                <td><strong>{html.escape(str(lnk.get('finding_id', lnk.get('id', ''))))}</strong></td>
                <td><span class="badge status-{status}">{status.upper()}</span></td>
                <td><code>{html.escape(linked_ids)}</code></td>
                <td>{html.escape(str(lnk.get('explanation', '')))}</td>
                <td>{int(lnk.get('confidence', 0.8) * 100)}%</td>
            </tr>
            """

        # Safety indicators table
        safety_html = ""
        # Filter top 10 most critical/caution indicators
        critical_safety = [
            s for s in safety_indicators
            if s.get("warning_level") in {"critical", "caution"}
        ]
        for s in critical_safety[:12]:
            w_level = s.get("warning_level", "normal")
            safety_html += f"""
            <tr>
                <td><strong>{html.escape(str(s.get('indicator')))}</strong></td>
                <td>Track {s.get('vehicle_a')} & Track {s.get('vehicle_b')}</td>
                <td>t = {s.get('time_s', 0.0):.2f}s</td>
                <td><strong>{s.get('value', 0.0):.2f} {s.get('unit', '')}</strong></td>
                <td><span class="badge warning-{w_level}">{w_level.upper()}</span></td>
                <td>{html.escape(str(s.get('caveats', '')))}</td>
            </tr>
            """

        # Keyframe previews
        keyframes_html = ""
        for kf in keyframe_cards:
            keyframes_html += f"""
            <div class="keyframe-card">
                <img src="{kf['src']}" alt="{html.escape(kf['title'])}" />
                <div class="keyframe-caption">{html.escape(kf['title'])}</div>
            </div>
            """

        # Undated claims
        undated_html = ""
        for uc in undated_claims:
            undated_html += f"""
            <div class="undated-claim-card">
                <div class="claim-header">
                    <span class="badge-id">{uc.get('id')}</span>
                    <span class="badge badge-claim">CLAIM</span>
                    <strong>{html.escape(str(uc.get('vehicle_desc')))} — {html.escape(str(uc.get('action')))}</strong>
                </div>
                <blockquote class="claim-quote">"{html.escape(str(uc.get('original_text')))}"</blockquote>
                <div class="claim-meta">Temporal Anchor: <em>{html.escape(str(uc.get('temporal_anchor', 'unstated')))}</em> | Statement: {uc.get('statement_id')}</div>
            </div>
            """

        min_ttc_display = f"{min_ttc:.2f} s" if min_ttc is not None else "N/A"
        max_drac_display = f"{max_drac:.2f} m/s²" if max_drac is not None else "N/A"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.title)} — {case_id}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            --accent-blue: #38bdf8;
            --accent-purple: #c084fc;
            --accent-green: #4ade80;
            --accent-amber: #fbbf24;
            --accent-red: #f87171;
            --accent-cyan: #22d3ee;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
        .header-top h1 {{ font-size: 24px; font-weight: 700; color: #fff; }}
        .case-tag {{
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            border: 1px solid var(--accent-blue);
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
        }}
        .header-subtitle {{ color: var(--text-secondary); font-size: 14px; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }}
        .kpi-val {{ font-size: 26px; font-weight: 700; margin-bottom: 4px; }}
        .kpi-label {{ color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .val-red {{ color: var(--accent-red); }}
        .val-green {{ color: var(--accent-green); }}
        .val-blue {{ color: var(--accent-blue); }}
        .val-purple {{ color: var(--accent-purple); }}

        .section {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .badge-observation {{ background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
        .badge-derived_measurement {{ background: rgba(192, 132, 252, 0.2); color: var(--accent-purple); border: 1px solid var(--accent-purple); }}
        .badge-claim {{ background: rgba(251, 191, 36, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber); }}
        .badge-inference {{ background: rgba(34, 211, 238, 0.2); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }}

        .badge-source-video {{ background: rgba(56, 189, 248, 0.1); color: #7dd3fc; }}
        .badge-source-sensor {{ background: rgba(192, 132, 252, 0.1); color: #d8b4fe; }}
        .badge-source-witness {{ background: rgba(251, 191, 36, 0.1); color: #fde68a; }}

        .badge-id {{ font-family: monospace; font-size: 11px; color: var(--text-secondary); background: #0f172a; padding: 2px 6px; border-radius: 4px; }}
        .conf-badge {{ font-size: 11px; color: #a7f3d0; background: rgba(74, 222, 128, 0.1); padding: 2px 6px; border-radius: 4px; }}

        /* Timeline styles */
        .timeline {{ position: relative; padding-left: 20px; border-left: 2px solid var(--border-color); }}
        .timeline-row {{ position: relative; margin-bottom: 20px; }}
        .timeline-dot {{
            position: absolute;
            left: -27px;
            top: 14px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--accent-blue);
            border: 2px solid var(--bg-primary);
        }}
        .timeline-content {{
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 16px;
        }}
        .card-header {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
        .time-stamp {{ font-family: monospace; font-size: 12px; color: #38bdf8; font-weight: 600; display: inline-block; margin-bottom: 4px; }}
        .event-type {{ font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 4px; }}
        .event-desc {{ font-size: 13px; color: var(--text-secondary); }}

        /* Contradiction Table */
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background: #0f172a; color: var(--text-secondary); font-weight: 600; }}
        .status-contradict {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }}
        .status-support {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .status-partial {{ background: rgba(251, 191, 36, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber); }}
        .status-gap {{ background: rgba(148, 163, 184, 0.2); color: var(--text-secondary); border: 1px solid var(--border-color); }}

        .status-contradict-row {{ background: rgba(248, 113, 113, 0.05); }}

        .warning-critical {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }}
        .warning-caution {{ background: rgba(251, 191, 36, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber); }}
        .warning-normal {{ background: rgba(74, 222, 128, 0.2); color: var(--accent-green); }}

        /* Keyframe Grid */
        .keyframe-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 16px;
        }}
        .keyframe-card {{
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}
        .keyframe-card img {{ width: 100%; height: auto; display: block; }}
        .keyframe-caption {{ padding: 8px 12px; font-size: 12px; color: var(--text-secondary); font-weight: 600; text-align: center; }}

        /* Undated Claims */
        .undated-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
        .undated-claim-card {{ background: #0f172a; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; }}
        .claim-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 13px; }}
        .claim-quote {{ font-style: italic; color: #fde68a; font-size: 12px; margin-bottom: 6px; border-left: 2px solid var(--accent-amber); padding-left: 8px; }}
        .claim-meta {{ font-size: 11px; color: var(--text-secondary); }}

        .disclaimer-box {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-blue);
            border-radius: 6px;
            padding: 14px 18px;
            margin-top: 24px;
            font-size: 12px;
            color: var(--text-secondary);
        }}
        .disclaimer-box strong {{ color: #f8fafc; }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <div class="header-top">
            <h1>{html.escape(self.title)}</h1>
            <span class="case-tag">{case_id}</span>
        </div>
        <p class="header-subtitle">
            AI-Assisted Multi-Modal Evidence Ingestion, Cross-Verification & Chronological Timeline Reconstruction
        </p>
    </header>

    <!-- KPI Metric Cards -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-val val-blue">{total_events}</div>
            <div class="kpi-label">Ingested Events</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val val-red">{contradictions_count}</div>
            <div class="kpi-label">Contradictions Flagged</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val val-green">{supports_count}</div>
            <div class="kpi-label">Corroborated Facts</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val val-purple">{min_ttc_display}</div>
            <div class="kpi-label">Min Time-To-Collision (TTC)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val val-red">{max_drac_display}</div>
            <div class="kpi-label">Peak DRAC Required</div>
        </div>
    </div>

    <!-- Contradictions & Evidence Verification Matrix -->
    <div class="section">
        <div class="section-title">
            <span>⚖️ Cross-Modal Evidence Verification & Contradiction Matrix</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th style="width: 100px;">Finding ID</th>
                    <th style="width: 110px;">Verification</th>
                    <th style="width: 140px;">Linked Evidence</th>
                    <th>Forensic Comparative Finding</th>
                    <th style="width: 80px;">Confidence</th>
                </tr>
            </thead>
            <tbody>
                {links_html}
            </tbody>
        </table>
    </div>

    <!-- Integrated Chronological Timeline -->
    <div class="section">
        <div class="section-title">
            <span>⏱️ Integrated Chronological Timeline</span>
        </div>
        <div class="timeline">
            {timeline_html}
        </div>
    </div>

    <!-- Keyframe Evidence Gallery -->
    {'<div class="section"><div class="section-title"><span>📸 Keyframe Visual Evidence</span></div><div class="keyframe-grid">' + keyframes_html + '</div></div>' if keyframes_html else ''}

    <!-- Surrogate Safety Indicators (TTC / DRAC) -->
    <div class="section">
        <div class="section-title">
            <span>⚠️ Surrogate Safety Measures (TTC & DRAC Conflict Analysis)</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Measure</th>
                    <th>Vehicles</th>
                    <th>Timestamp</th>
                    <th>Value</th>
                    <th>Warning Level</th>
                    <th>Forensic Caveat</th>
                </tr>
            </thead>
            <tbody>
                {safety_html if safety_html else '<tr><td colspan="6" style="text-align: center; color: var(--text-secondary);">No critical safety warnings detected.</td></tr>'}
            </tbody>
        </table>
    </div>

    <!-- Undated Witness Statements -->
    <div class="section">
        <div class="section-title">
            <span>📝 Contextual Witness Claims (Unstated Numerical Timings)</span>
        </div>
        <div class="undated-grid">
            {undated_html if undated_html else '<p style="color: var(--text-secondary);">All witness claims anchored to numerical timestamps.</p>'}
        </div>
    </div>

    <!-- Forensic Legal Neutrality Disclaimer -->
    <div class="disclaimer-box">
        <strong>Forensic Integrity & Legal Neutrality Notice:</strong><br>
        This document presents an automated algorithmic synthesis of dashcam video tracking, CAN bus vehicular sensor telemetry, and written witness testimony. All observations are categorized according to epistemic confidence levels (<em>observation</em>, <em>derived_measurement</em>, <em>claim</em>, <em>inference</em>). This reconstruction highlights physical measurements, velocity derivatives, and narrative discrepancies without declaring legal liability, traffic negligence, or statutory fault.
    </div>
</div>
</body>
</html>
"""


def generate_report(
    witness_path: Path | str = "witness_claims.json",
    sensor_path: Path | str = "sensor_events.json",
    video_path: Path | str = "video_events.json",
    links_path: Path | str = "evidence_links.json",
    safety_path: Path | str = "safety_indicators.json",
    keyframes_dir: Optional[Path | str] = "data/processed/tbone_demo/keyframes",
    output_path: Path | str = "report.html",
    case_id: str = "CASE-2026-T46-01",
) -> Path:
    """Generate and write a standalone forensic report HTML file."""
    # Fallback to mock/ files if root files are not present
    w_p = Path(witness_path) if Path(witness_path).exists() else Path("mock/witness_claims.json")
    s_p = Path(sensor_path) if Path(sensor_path).exists() else Path("mock/sensor_events.json")
    v_p = Path(video_path) if Path(video_path).exists() else Path("mock/video_events.json")
    l_p = Path(links_path) if Path(links_path).exists() else Path("mock/evidence_links.json")
    safe_p = Path(safety_path) if Path(safety_path).exists() else Path("mock/safety_indicators.json")

    generator = ForensicReportGenerator()
    w_claims = generator._load_json(w_p)
    s_events = generator._load_json(s_p)
    v_events = generator._load_json(v_p)
    e_links = generator._load_json(l_p)
    s_indicators = generator._load_json(safe_p)

    kd = Path(keyframes_dir) if keyframes_dir and Path(keyframes_dir).exists() else None

    html_content = generator.build_report_html(
        witness_claims=w_claims,
        sensor_events=s_events,
        video_events=v_events,
        evidence_links=e_links,
        safety_indicators=s_indicators,
        keyframes_dir=kd,
        case_id=case_id,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 Task 7 — Forensic Report HTML Generator")
    parser.add_argument("--witness", type=str, default="witness_claims.json", help="Path to witness_claims.json")
    parser.add_argument("--sensors", type=str, default="sensor_events.json", help="Path to sensor_events.json")
    parser.add_argument("--video", type=str, default="video_events.json", help="Path to video_events.json")
    parser.add_argument("--links", type=str, default="evidence_links.json", help="Path to evidence_links.json")
    parser.add_argument("--safety", type=str, default="safety_indicators.json", help="Path to safety_indicators.json")
    parser.add_argument("--keyframes", type=str, default="data/processed/tbone_demo/keyframes", help="Keyframes folder")
    parser.add_argument("--out", type=str, default="report.html", help="Path to output HTML report")
    parser.add_argument("--case-id", type=str, default="CASE-2026-T46-01", help="Case identifier")
    args = parser.parse_args()

    out_path = generate_report(
        witness_path=args.witness,
        sensor_path=args.sensors,
        video_path=args.video,
        links_path=args.links,
        safety_path=args.safety,
        keyframes_dir=args.keyframes,
        output_path=args.out,
        case_id=args.case_id,
    )
    print(f"Forensic Report generated successfully: {out_path.resolve()}")


if __name__ == "__main__":
    main()
