# Agent 2 Task & Execution Plan: NLP, Sensors & Evidence Intelligence

> **Role**: Person 2 (P2) — NLP, Sensors and Evidence Intelligence  
> **Team**: VISERION (Team 46) — CBIT Hacktoberfest '26  
> **Branch**: `p2-nlp-sensors`  
> **Core Deliverables**: `witness_claims.json`, `sensor_events.json`, `evidence_links.json`, `safety_indicators.json`, `featherless_client.py`, `report.html`

---

## 1. Role Overview & Design Principles

Person 2 is responsible for extracting evidence from non-video sources (written witness statements and vehicle sensor telemetry), cross-referencing and contradiction matching across all evidence modalities (video tracks, sensor data, witness statements), estimating safety metrics (Time-To-Collision - TTC), and generating evidence-grounded forensic reports.

### Forensic Design Principles:
1. **Schema Compliance**: All generated items strictly follow the unified event schema from Hour 0-2.
2. **Epistemic Classification**: Every data item must be explicitly categorized:
   - `observation`: Direct empirical recording (e.g., video frame bounding box).
   - `derived_measurement`: Direct mathematical computation from sensor or tracking data (e.g., speed drop, deceleration, TTC).
   - `claim`: Reported assertion from human witness (subjective, requires verification).
   - `inference`: Model prediction or reasoning (e.g., VLM keyframe caption, contradiction hypothesis).
3. **Explicit Uncertainty**: Every record communicates confidence $[0.0, 1.0]$ and time uncertainty intervals $[\pm \Delta t]$, never claiming absolute certainty.
4. **No Legal Fault Claims**: The tool highlights physical observations, measurement deltas, and inconsistencies; it strictly avoids declaring legal liability or guilt.

---

## 2. Handoff Contract Matrix

### 2.1 Person 2 Receives
| From | Deliverable | Format / Schema | Usage by P2 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | `tracks.json` | `frames[{frame, time_s, objects[{track_id, class, bbox, conf}]}]` | Compute TTC / DRAC; match witness descriptions to track IDs | In Progress |
| **P1** | `video_events.json` | Common event schema (`id`, `event_type`, `start_time`, `end_time`, `track_ids`, `confidence`, `evidence_ref`) | Contradiction & support matching against claims and sensors | Pending |
| **P1** | Frame-to-time mapping | FPS and timestamp offset | Synchronize sensor timestamps to video seconds | In Progress |
| **P3** | Common Event Schema | `schema.py` / `schema.md` (Pydantic / dataclass) | Foundation for all JSON exports | Ready |
| **P3** | Mock Scenario Data | `/demo/` dashcam, witness statement, sensor CSV | Development and automated test harness | Ready |

### 2.2 Person 2 Provides
| To | Deliverable | Format / Schema | Target Hour | Status |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | `featherless_client.py` | Shared vision & LLM client wrapper with prompt templates | Hour 3 | **COMPLETED** |
| **P1** | Event Type Feedback | Guidance on which event categories to extract for contradiction testing | Hour 4 | **COMPLETED** |
| **P3** | `witness_claims.json` | Structured claims with original wording, time interval, uncertainty | Hour 8 | **COMPLETED** |
| **P3** | `sensor_events.json` | Timestamped sensor events (hard braking, stop, speed drop) | Hour 8 | **COMPLETED** |
| **P3** | `evidence_links.json` | Support / contradict / partial / unverified / gap findings | Hour 12 | **COMPLETED** |
| **P3** | `safety_indicators.json` | TTC / DRAC values with method, units, and caveats | Hour 16 | Pending |
| **P3** | `report.html` / PDF | Forensic incident reconstruction summary report | Hour 18 | Pending |

---

## 3. Person 2 Task Breakdown

### Task 1: Featherless Client Wrapper (Shared with P1 for vision calls)
- **Deliverable**: `featherless_client.py` & `src/intelligence/featherless_client.py`
- **Lead**: Person 2 (consumed by P1 and P2)
- **Status**: **COMPLETED**
- **Requirements**:
  - Unified OpenAI-compatible wrapper for Featherless API (`https://api.featherless.ai/v1`).
  - Vision model support for frame analysis (`describe_keyframe`) with base64 image encoding.
  - Text model support for witness extraction (`extract_witness_claims`) and explanation generation (`generate_explanation`).
  - Robust offline / mock fallback mode when `FEATHERLESS_API_KEY` is not present or offline.
  - Built-in forensic guardrails: never declare legal fault; tag outputs as `inference` with confidence metrics.
  - Comprehensive unit tests verifying both mock and live interface modes.

### Task 2: Witness Extraction into Structured Claims
- **Deliverable**: `src/nlp/witness_parser.py` -> `witness_claims.json`
- **Status**: **COMPLETED**
- **Contract Fields**: `id`, `statement_id`, `original_text`, `vehicle_desc`, `action`, `direction`, `target_vehicle`, `temporal_anchor`, `time_interval`, `uncertainty`, `kind="claim"`
- **Requirements**:
  - Ingest raw written witness accounts (e.g. from `/demo/witness_statement.txt`).
  - Extract structured claims: vehicle description, movement action (speeding, approach, collision, rotation, stopping, braking, swerve, signal violation).
  - Handle coreference across sentences (e.g. resolving pronouns like "it" to antecedent vehicles).
  - Preserve exact verbatim quote snippets for each claim.
  - Strictly enforce no invented timestamps when none are provided (`time_interval=None`).
  - Handle time uncertainty when explicitly stated (e.g. "around 4 seconds in" -> $[3.5s, 4.5s]$).

### Task 3: Sensor CSV/JSON Processing
- **Deliverable**: `src/sensors/sensor_processor.py` -> `sensor_events.json`
- **Status**: **COMPLETED**
- **Contract Fields**: `id`, `event_type`, `time`, `value`, `unit`, `source_row`, `confidence`, `kind="derived_measurement"`
- **Requirements**:
  - Parse telematics / CAN bus / accelerometer CSV data (timestamps, speed, acceleration, braking status, steering).
  - Detect critical kinetic events: hard braking ($> 0.4g$ or $> 4.0 m/s^2$), abrupt deceleration / speed drop, sudden stop ($speed = 0$), and lateral impact shocks.
  - Align sensor timestamps with incident timeline zero-point ($t=0$ at incident start / collision).

### Task 4: Matching and Contradiction Engine
- **Deliverable**: `src/intelligence/matcher.py` -> `evidence_links.json`
- **Status**: **COMPLETED**
- **Contract Fields**: `id`, `finding_id`, `linked_ids[]`, `status` (`support` | `contradict` | `partial` | `unverified` | `gap`), `confidence`, `explanation`, `missing_info`
- **Requirements**:
  - Align witness claims, sensor events, and video events along temporal overlap windows $[\text{start} - \Delta, \text{end} + \Delta]$.
  - Cross-check claims against objective data (e.g. witness claims "car never braked" vs. sensor shows $0.65g$ deceleration or video shows vehicle stopping).
  - Flag timeline coverage gaps (e.g. sensor recording missing during interval, video occlusion).

### Task 5: Per-Finding Grounded Explanations & Confidence
- **Deliverable**: `src/intelligence/matcher.py`
- **Status**: **COMPLETED**
- **Requirements**:
  - Generate clear, factual forensic explanations citing strictly linked evidence IDs and timestamps.
  - Assign calibrated confidence scores based on source reliability (sensor/video > witness memory).
  - Strictly maintain neutral, factual tone without attributing legal blame.

### Task 6: Simple TTC from P1's Tracks (Optional DRAC)
- **Deliverable**: `src/sensors/safety_indicators.py` -> `safety_indicators.json`
- **Status**: Pending
- **Requirements**:
  - Compute Time-To-Collision (TTC) using bounding box closing speed and centroid distances from `tracks.json`.
  - Compute Deceleration Rate to Avoid Crash (DRAC) where applicable.
  - Mark outputs clearly as heuristic safety indicators, never definitive proof of collision.

### Task 7: Forensic Report Generation (HTML / PDF)
- **Deliverable**: `src/reporting/report_generator.py` -> `report.html`
- **Status**: Pending
- **Requirements**:
  - Render an interactive standalone HTML forensic timeline report.
  - Include summary dashboard: incident metadata, evidence breakdown, contradiction matrix, and chronological event sequence with confidence badges.

---

## 4. Person 2 Execution & Progress Log

| Date / Time | Task | Change Summary | Artifacts | Verification |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-31 03:12 | **Task 1: Featherless Client** | Created `featherless_client.py` and `src/intelligence/featherless_client.py` with vision & LLM prompt styles, mock offline engine, and strict uncertainty tagging. | `featherless_client.py`, `tests/test_featherless_client.py` | Pytest passed (5 tests) |
| 2026-10-31 03:48 | **Task 2: Witness Extraction** | Upgraded claim extraction engine to extract multiple atomic claims, correctly associate vehicles, resolve cross-sentence coreferences ("it"), recognize actions (collision, approach, direction, rotation, stopping, speeding), preserve exact quote snippets, and avoid inventing timestamps when unstated. Added live API tester. | `src/nlp/witness_parser.py`, `tests/test_witness_claims.py`, `scripts/test_featherless_live.py` | Pytest passed (18 passed, 1 skipped live) |
| 2026-10-31 03:54 | **Task 3: Sensor Processing** | Implemented sensor telemetry processor detecting critical kinetic events: hard braking, lateral impact shock, speed drop, and vehicle stop. Generated sensor_events.json and mock data. | `src/sensors/sensor_processor.py`, `demo/sensor_telemetry.csv`, `sensor_events.json`, `tests/test_sensor_processor.py` | Pytest passed (20 passed, 1 skipped live) |
| 2026-10-31 04:12 | **Task 4 & 5: Contradiction Engine** | Implemented multi-source evidence cross-referencing engine detecting contradictions (e.g. no braking claim vs -7.8 m/s^2 sensor reading and brake lights; stationary claim vs video approach in motion), support, and coverage gaps with calibrated confidence and grounded explanations across witness, sensor, and video tracks. | `src/intelligence/matcher.py`, `evidence_links.json`, `mock/video_events.json`, `mock/tracks.json`, `tests/test_matcher.py` | Pytest passed (24 passed, 1 skipped live) |
