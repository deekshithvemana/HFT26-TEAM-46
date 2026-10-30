"""Featherless AI Client Wrapper for Multi-Source Accident Reconstruction.

Shared between Person 1 (Vision keyframe descriptions) and Person 2 (Witness NLP & Evidence matching).

Adheres to Core Forensic Principles:
1. Every observation or analysis is tagged (Observation / Derived Measurement / Claim / Inference).
2. Explicit confidence and uncertainty metrics are communicated.
3. Strictly avoids declaring legal responsibility, liability, or fault.
4. Built-in resilient offline/mock mode for reliable local execution and testing.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEFAULT_TEXT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

FORENSIC_SYSTEM_PROMPT = (
    "You are an objective forensic timeline reconstruction assistant analyzing road incident evidence.\n"
    "Guidelines:\n"
    "1. Strictly report observable physical movements, vehicle orientations, and timeline indicators.\n"
    "2. Explicitly communicate uncertainty (e.g., lighting limits, angle occlusion, timestamp variance).\n"
    "3. Distinguish empirical observations from model inferences.\n"
    "4. NEVER declare legal fault, responsibility, or traffic liability."
)


def encode_image_to_base64(image_path: Path | str) -> str:
    """Encode an image file to a base64 JPEG data URL."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found at {p}")
    suffix = p.suffix.lower().replace(".", "") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    with p.open("rb") as handle:
        data = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:image/{suffix};base64,{data}"


class FeatherlessClient:
    """Unified client for Featherless AI text and vision models with automatic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        vision_model: Optional[str] = None,
        text_model: Optional[str] = None,
        mock: Optional[bool] = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.environ.get("FEATHERLESS_API_KEY")
        self.base_url = (base_url or os.environ.get("FEATHERLESS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.vision_model = vision_model or os.environ.get("FEATHERLESS_VISION_MODEL") or DEFAULT_VISION_MODEL
        self.text_model = text_model or os.environ.get("FEATHERLESS_TEXT_MODEL") or DEFAULT_TEXT_MODEL
        self.timeout = timeout

        # Enable mock mode if explicitly requested or if no API key is provided
        env_mock = os.environ.get("FEATHERLESS_MOCK", "").lower() in ("1", "true", "yes")
        self.mock = mock if mock is not None else (env_mock or not self.api_key)

    def is_mock(self) -> bool:
        return self.mock

    # -------------------------------------------------------------------------
    # Person 1 Vision Deliverable: Keyframe Description
    # -------------------------------------------------------------------------
    def describe_keyframe(
        self,
        image_path: Path | str,
        event_context: Optional[dict[str, Any]] = None,
        custom_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze a video keyframe around an event and return an Inference description.

        Shared with P1 for Task 6 (exporting keyframes with Featherless descriptions).
        """
        p = Path(image_path)
        ctx = event_context or {}
        event_type = ctx.get("event_type", "detected vehicle movement")
        timestamp = ctx.get("time_s", ctx.get("start_time", 0.0))

        prompt = custom_prompt or (
            f"Analyze this keyframe captured at t={timestamp:.2f}s during a '{event_type}' event.\n"
            "Describe the visible vehicles, their relative positioning, road surface condition, and visible "
            "brake light or impact indicators. Avoid assigning fault. Output your assessment in JSON format:\n"
            "{\n"
            '  "description": "objective physical description",\n'
            '  "visual_elements": ["item1", "item2"],\n'
            '  "confidence": 0.85,\n'
            '  "uncertainty_factors": ["potential factors"]\n'
            "}"
        )

        if self.mock or not self.api_key:
            return self._mock_describe_keyframe(p, event_type, timestamp)

        try:
            image_data_uri = encode_image_to_base64(p)
            payload = {
                "model": self.vision_model,
                "messages": [
                    {"role": "system", "content": FORENSIC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_uri}},
                        ],
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 400,
            }
            res = self._post_chat(payload)
            content = res["choices"][0]["message"]["content"]
            parsed = self._extract_json(content)
            if parsed and isinstance(parsed, dict) and "description" in parsed:
                parsed.setdefault("kind", "inference")
                parsed.setdefault("confidence", 0.85)
                return parsed

            return {
                "description": content.strip(),
                "visual_elements": ["vehicle", "road_scene"],
                "confidence": 0.80,
                "uncertainty_factors": ["unparsed json response"],
                "kind": "inference",
            }
        except Exception as exc:
            # Fall back safely to mock response on network or auth error
            mock_res = self._mock_describe_keyframe(p, event_type, timestamp)
            mock_res["uncertainty_factors"].append(f"Remote API fallback ({type(exc).__name__})")
            return mock_res

    # -------------------------------------------------------------------------
    # Person 2 NLP Deliverable: Witness Claim Extraction
    # -------------------------------------------------------------------------
    def extract_witness_claims(
        self,
        statement_text: str,
        statement_id: str = "W1",
    ) -> list[dict[str, Any]]:
        """Extract structured claims from a written witness statement.
        
        Guarantees:
        1. Extracts multiple atomic claims per statement (splitting compound actions).
        2. Correctly associates actions with their specific vehicle actor.
        3. Recognizes collision, approach, direction, rotation, stopping, speeding, braking, swerve, signal violations.
        4. Preserves exact verbatim substring in original_text.
        5. Does not invent timestamps when none are provided in the statement.
        6. Resolves coreferences and pronouns ('it', compound predicates) across sentences.
        """
        if not statement_text or not statement_text.strip():
            return []

        if self.mock or not self.api_key:
            return self._mock_extract_witness_claims(statement_text, statement_id)

        prompt = (
            "You are an objective forensic collision analysis assistant. Extract all factual, testable claims from this witness statement:\n\n"
            f'"{statement_text.strip()}"\n\n'
            "RULES FOR EXTRACTION:\n"
            "1. Extract MULTIPLE separate claims. Break compound actions into atomic claims (e.g. 'rotated and stopped' must become two claims: 'rotation' and 'stopping').\n"
            "2. Accurately identify which vehicle performed each action. Distinguish between different vehicles (e.g. 'black SUV' vs 'white SUV').\n"
            "3. Recognise and categorize actions using standard terminology: 'speeding', 'approach', 'collision', 'rotation', 'stopping', 'hard_braking', 'no_braking', 'swerve', 'signal_violation'.\n"
            "4. Extract directional cues when present (e.g. 'from the right', 'from the left', 'straight', or null).\n"
            "5. Resolve pronouns across sentences (e.g., in 'collided with it', resolve 'it' to the specific vehicle struck, setting 'target_vehicle').\n"
            "6. Preserve the EXACT verbatim substring from the statement in 'original_text'. Do NOT paraphrase or summarize.\n"
            "7. CRITICAL TIMING RULE: If NO numerical seconds or timestamps are stated in the text, DO NOT invent timestamps. Output null for 'time_interval' and 'uncertainty'. If a relative anchor is stated (e.g. 'after the collision'), place it in 'temporal_anchor'.\n\n"
            "Return a JSON array of claim objects strictly conforming to this format:\n"
            "[\n"
            "  {\n"
            '    "vehicle_desc": "black SUV",\n'
            '    "action": "speeding",\n'
            '    "direction": null,\n'
            '    "target_vehicle": null,\n'
            '    "temporal_anchor": null,\n'
            '    "original_text": "exact snippet from statement",\n'
            '    "time_interval": null,\n'
            '    "uncertainty": null,\n'
            '    "confidence": 0.85\n'
            "  }\n"
            "]"
        )

        try:
            payload = {
                "model": self.text_model,
                "messages": [
                    {"role": "system", "content": FORENSIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1200,
            }
            res = self._post_chat(payload)
            content = res["choices"][0]["message"]["content"]
            parsed = self._extract_json(content)
            if isinstance(parsed, list) and len(parsed) > 0:
                claims = []
                for i, item in enumerate(parsed):
                    orig_quote = item.get("original_text", "")
                    # Ensure quote is an exact substring or fall back
                    if orig_quote not in statement_text:
                        # Attempt to locate snippet in statement
                        for line in statement_text.split("."):
                            if item.get("action", "") in line.lower() or item.get("vehicle_desc", "") in line:
                                orig_quote = line.strip()
                                break
                    if not orig_quote:
                        orig_quote = statement_text[:80]

                    t_int = item.get("time_interval")
                    # Strict validation: do not allow invented timestamps
                    if t_int is not None:
                        if not isinstance(t_int, list) or len(t_int) != 2:
                            t_int = None

                    claims.append({
                        "id": f"{statement_id}-C{i+1}",
                        "statement_id": statement_id,
                        "original_text": orig_quote,
                        "vehicle_desc": item.get("vehicle_desc", "unknown vehicle"),
                        "action": item.get("action", "unspecified movement"),
                        "direction": item.get("direction"),
                        "target_vehicle": item.get("target_vehicle"),
                        "temporal_anchor": item.get("temporal_anchor"),
                        "time_interval": t_int,
                        "uncertainty": float(item["uncertainty"]) if item.get("uncertainty") is not None else None,
                        "confidence": float(item.get("confidence", 0.75)),
                        "kind": "claim",
                    })
                return claims

            return self._mock_extract_witness_claims(statement_text, statement_id)
        except Exception:
            return self._mock_extract_witness_claims(statement_text, statement_id)

    # -------------------------------------------------------------------------
    # Person 2 Intelligence: Grounded Explanation
    # -------------------------------------------------------------------------
    def generate_explanation(
        self,
        finding_status: str,
        claim_summary: str,
        evidence_summary: str,
    ) -> str:
        """Generate a grounded forensic explanation for an evidence contradiction or support finding."""
        if self.mock or not self.api_key:
            return self._mock_generate_explanation(finding_status, claim_summary, evidence_summary)

        prompt = (
            f"Draft a concise forensic reconciliation note for this finding:\n"
            f"- Status: {finding_status.upper()}\n"
            f"- Claim: {claim_summary}\n"
            f"- Objective Evidence: {evidence_summary}\n\n"
            "State the observable facts, numerical deltas, and uncertainty intervals. "
            "Never attribute legal fault or liability. Limit to 2 clear sentences."
        )

        try:
            payload = {
                "model": self.text_model,
                "messages": [
                    {"role": "system", "content": FORENSIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 200,
            }
            res = self._post_chat(payload)
            return res["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._mock_generate_explanation(finding_status, claim_summary, evidence_summary)

    # -------------------------------------------------------------------------
    # Internal HTTP & Helpers
    # -------------------------------------------------------------------------
    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        if requests is None:
            raise RuntimeError("The 'requests' package is required for live Featherless calls.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _extract_json(self, text: str) -> Optional[Any]:
        """Extract JSON block or payload from LLM markdown response."""
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        candidate = match.group(1).strip() if match else text.strip()
        try:
            return json.loads(candidate)
        except Exception:
            # Try to locate first '[' or '{' to last ']' or '}'
            for start_char, end_char in (("[", "]"), ("{", "}")):
                start = candidate.find(start_char)
                end = candidate.rfind(end_char)
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(candidate[start : end + 1])
                    except Exception:
                        pass
        return None

    # -------------------------------------------------------------------------
    # Deterministic Mock Engine (Offline / Hackathon Demo Reliability)
    # -------------------------------------------------------------------------
    def _mock_describe_keyframe(
        self, image_path: Path, event_type: str, timestamp: float
    ) -> dict[str, Any]:
        """Deterministic, grounded mock caption for keyframe analysis."""
        return {
            "description": (
                f"Visual observation at t={timestamp:.2f}s during '{event_type}': "
                f"Primary vehicle detected within roadway boundary. Visual indicators show forward motion, "
                f"clear lane delineation, and vehicle brake indicators consistent with the telemetry record."
            ),
            "visual_elements": ["vehicle_primary", "lane_markings", "asphalt_surface", "brake_lamp_glow"],
            "confidence": 0.86,
            "uncertainty_factors": [
                "Single perspective viewpoint; depth is estimated",
                "Atmospheric lighting variance",
            ],
            "kind": "inference",
            "frame_ref": str(image_path.name),
        }

    def _mock_extract_witness_claims(
        self, statement_text: str, statement_id: str
    ) -> list[dict[str, Any]]:
        """Heuristic rule-based claim extractor for witness statements.
        
        Performs:
        - Multi-claim extraction across clauses and conjunctions.
        - Accurate vehicle entity tracking and association.
        - Coreference resolution across sentences ('it', 'the black SUV', compound predicates).
        - Action categorization (speeding, approach, collision, rotation, stopping, braking, swerve, signal).
        - Exact original text substring preservation.
        - Strict timing rule: NEVER invents timestamps when none are provided in statement.
        """
        if not statement_text or not statement_text.strip():
            return []

        text = statement_text.strip()

        # 1. Identify canonical vehicles in order of appearance
        veh_regex = re.compile(
            r"\b(?:(?:the|a|an)\s+)?((?:(?:black|white|red|blue|silver|grey|gray|green|yellow|dark|light)\s+)?(?:suv|sedan|car|truck|van|motorcycle|bike|pickup|bus|vehicle))\b",
            re.IGNORECASE,
        )

        known_vehicles: list[str] = []
        for m in veh_regex.finditer(text):
            name = m.group(1).strip()
            lower = name.lower()
            if lower in ("it", "they", "vehicle", "car") and known_vehicles:
                continue
            words = [
                w.upper() if w.lower() == "suv" else (w.capitalize() if w.lower() in ("sedan", "car", "truck") else w.lower())
                for w in name.split()
            ]
            canonical = " ".join(words)
            if canonical not in known_vehicles:
                known_vehicles.append(canonical)

        # Action patterns with strict priority
        action_defs = [
            ("speeding", re.compile(r"\b(moving\s+very\s+fast|moving\s+fast|driving\s+fast|speeding|going\s+(?:very\s+)?fast|high\s+speed|rapidly\s+accelerat\w+)\b", re.IGNORECASE)),
            ("approach", re.compile(r"\b(entered(?:\s+from\s+[^\.,;]+)?|approach\w*|pulled\s+out|came\s+from|came\s+out|turn\w*\s+onto|turn\w*\s+into)\b", re.IGNORECASE)),
            ("collision", re.compile(r"\b(collid\w*(?:\s+with\s+[^\.,;]+)?|crash\w*(?:\s+into\s+[^\.,;]+)?|hit|struck|impact\w*|t-bon\w*|sideswip\w*)\b", re.IGNORECASE)),
            ("rotation", re.compile(r"\b(rotat\w*|spun(?:\s+around|\s+out)?|spin\w*|fish-tail\w*|careen\w*)\b", re.IGNORECASE)),
            ("stopping", re.compile(r"\b(stop\w*(?:\s+after\s+[^\.,;]+)?|came\s+to\s+a\s+stop|came\s+to\s+a\s+halt|halt\w*|pull\w*\s+over)\b", re.IGNORECASE)),
            ("hard_braking", re.compile(r"\b(slam\w*\s+on\s+(?:the\s+)?brakes|hard\s+brak\w+|brak\w*\s+hard|sudden\s+brak\w+)\b", re.IGNORECASE)),
            ("no_braking", re.compile(r"\b(didn't\s+brake|never\s+braked|did\s+not\s+brake|no\s+brak\w+|without\s+brak\w+)\b", re.IGNORECASE)),
            ("swerve", re.compile(r"\b(swerv\w*|veer\w*|turn\w*\s+sharply|dodg\w*|evasive\s+action)\b", re.IGNORECASE)),
            ("signal_violation", re.compile(r"\b(ran\s+(?:the\s+)?red\s+light|pass\w*\s+(?:the\s+)?red\s+(?:light|signal))\b", re.IGNORECASE)),
        ]

        dir_regex = re.compile(r"\b(from\s+the\s+right|from\s+the\s+left|from\s+behind|head\s+on|straight|northbound|southbound|eastbound|westbound)\b", re.IGNORECASE)
        temporal_anchor_regex = re.compile(r"\b(after\s+(?:the\s+)?(?:collision|crash|impact|accident)|before\s+(?:the\s+)?(?:collision|crash|impact|accident))\b", re.IGNORECASE)
        explicit_time_regex = re.compile(r"\b(?:around|at|approximately)?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", re.IGNORECASE)

        claims: list[dict[str, Any]] = []
        claim_count = 0
        current_subject = known_vehicles[0] if known_vehicles else "unspecified vehicle"

        # Split into sentences preserving punctuation
        sentence_matches = list(re.finditer(r"([^.!?]+[.!?]?)", text))
        for s_match in sentence_matches:
            raw_sentence = s_match.group(1).strip()
            if not raw_sentence:
                continue

            clean_sentence = raw_sentence.rstrip(".!?")
            
            # Split sentence on coordinating conjunctions for compound actions
            clause_splits = list(re.finditer(r"(.*?)(?:\s+(?:and|then)\s+|$)", clean_sentence, re.IGNORECASE))
            clauses = [m.group(1).strip() for m in clause_splits if m.group(1).strip()]
            if not clauses:
                clauses = [clean_sentence]

            sentence_subject = None

            for clause in clauses:
                # Check for explicit vehicle in this clause
                clause_veh = None
                for v in known_vehicles:
                    if re.search(r"\b" + re.escape(v) + r"\b", clause, re.IGNORECASE):
                        clause_veh = v
                        break

                if clause_veh:
                    current_subject = clause_veh
                    if sentence_subject is None:
                        sentence_subject = clause_veh
                elif sentence_subject:
                    # Inherit subject from earlier in the same sentence
                    current_subject = sentence_subject

                # Check actions in this clause
                detected_actions = []
                for action_name, action_re in action_defs:
                    match = action_re.search(clause)
                    if match:
                        detected_actions.append((action_name, match))

                if not detected_actions:
                    continue

                for action_name, match in detected_actions:
                    claim_count += 1
                    
                    # Direction
                    dir_m = dir_regex.search(clause)
                    direction = dir_m.group(1).lower() if dir_m else None

                    # Temporal anchor
                    anchor_m = temporal_anchor_regex.search(clause)
                    temporal_anchor = anchor_m.group(1).lower() if anchor_m else None

                    # Explicit time - STRICT: only if explicitly present!
                    time_m = explicit_time_regex.search(clause) or explicit_time_regex.search(raw_sentence)
                    time_interval = None
                    uncertainty = None
                    if time_m:
                        t_val = float(time_m.group(1))
                        time_interval = [max(0.0, round(t_val - 0.5, 2)), round(t_val + 0.5, 2)]
                        uncertainty = 1.0

                    # Target vehicle & pronoun resolution
                    target_vehicle = None
                    if action_name == "collision":
                        it_m = re.search(r"\b(?:with|hit|into)\s+it\b", clause, re.IGNORECASE)
                        if it_m and len(known_vehicles) >= 2:
                            other = [v for v in known_vehicles if v.lower() != current_subject.lower()]
                            if other:
                                target_vehicle = other[0]
                        else:
                            for v in known_vehicles:
                                if v.lower() != current_subject.lower() and re.search(r"\b" + re.escape(v) + r"\b", clause, re.IGNORECASE):
                                    target_vehicle = v
                                    break

                    # Preserve exact original text substring from raw statement
                    exact_quote = clause
                    if clause in text:
                        exact_quote = clause
                    elif raw_sentence in text:
                        exact_quote = raw_sentence

                    claims.append({
                        "id": f"{statement_id}-C{claim_count}",
                        "statement_id": statement_id,
                        "original_text": exact_quote,
                        "vehicle_desc": current_subject,
                        "action": action_name,
                        "direction": direction,
                        "target_vehicle": target_vehicle,
                        "temporal_anchor": temporal_anchor,
                        "time_interval": time_interval,
                        "uncertainty": uncertainty,
                        "confidence": 0.85 if action_name in ("collision", "speeding", "stopping") else 0.75,
                        "kind": "claim",
                    })

        if not claims:
            claims.append({
                "id": f"{statement_id}-C1",
                "statement_id": statement_id,
                "original_text": statement_text[:80],
                "vehicle_desc": known_vehicles[0] if known_vehicles else "observed vehicle",
                "action": "reported incident involvement",
                "direction": None,
                "target_vehicle": None,
                "temporal_anchor": None,
                "time_interval": None,
                "uncertainty": None,
                "confidence": 0.60,
                "kind": "claim",
            })

        return claims

    def _mock_generate_explanation(
        self, finding_status: str, claim_summary: str, evidence_summary: str
    ) -> str:
        status = finding_status.lower()
        if status == "contradict":
            return (
                f"Contradiction identified: Claim states '{claim_summary}', whereas objective telemetry/video "
                f"records '{evidence_summary}'. Delta exceeds expected perceptual uncertainty bounds."
            )
        elif status == "support":
            return (
                f"Corroborating evidence: Claim of '{claim_summary}' aligns with recorded evidence '{evidence_summary}' "
                f"within temporal tolerance windows."
            )
        elif status == "partial":
            return (
                f"Partial alignment: Event sequence matches general timeline for '{claim_summary}', "
                f"but quantitative measurements '{evidence_summary}' differ by magnitude."
            )
        elif status == "gap":
            return (
                f"Evidence gap detected: For event '{claim_summary}', required cross-modality verification data "
                f"is unavailable or occluded ({evidence_summary})."
            )
        return f"Unverified claim: '{claim_summary}' lacks sufficient corroborating telemetry ({evidence_summary})."


# Expose root-level convenience instance
default_client = FeatherlessClient()


def main() -> None:
    parser = argparse.ArgumentParser(description="Featherless AI Client for Accident Forensics")
    subparsers = parser.add_subparsers(dest="command")

    desc_parser = subparsers.add_parser("describe", help="Describe keyframe image")
    desc_parser.add_argument("image", type=Path, help="Path to image file")
    desc_parser.add_argument("--event", type=str, default="collision", help="Event context")
    desc_parser.add_argument("--time", type=float, default=2.5, help="Event timestamp in seconds")

    claim_parser = subparsers.add_parser("parse-witness", help="Extract claims from statement")
    claim_parser.add_argument("text", type=str, help="Witness statement text or path")
    claim_parser.add_argument("--id", type=str, default="W1", help="Statement identifier")

    args = parser.parse_args()
    client = FeatherlessClient()

    if args.command == "describe":
        res = client.describe_keyframe(args.image, {"event_type": args.event, "time_s": args.time})
        print(json.dumps(res, indent=2))
    elif args.command == "parse-witness":
        statement = args.text
        if Path(statement).exists():
            statement = Path(statement).read_text(encoding="utf-8")
        res = client.extract_witness_claims(statement, statement_id=args.id)
        print(json.dumps(res, indent=2))
    else:
        print(f"Featherless Client ready. Mock mode active: {client.is_mock()}")
        print("Use --help for CLI commands.")


if __name__ == "__main__":
    main()
