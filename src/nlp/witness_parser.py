"""P2 Task 2 — Witness extraction into structured claims.

Extracts testable forensic claims from raw written witness accounts:
- vehicle, action, direction, target_vehicle, temporal_anchor, original wording
- strictly preserves verbatim quote snippets
- does NOT invent timestamps when none are provided
- exports to witness_claims.json adhering to common schema
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from src.intelligence.featherless_client import FeatherlessClient, default_client
from src.schema import WitnessClaim, save_json_records


class WitnessParser:
    """Parser that ingests witness statements and outputs validated WitnessClaim records."""

    def __init__(self, client: Optional[FeatherlessClient] = None) -> None:
        self.client = client or default_client

    def parse_statement(
        self,
        statement_text: str,
        statement_id: str = "W1",
    ) -> list[WitnessClaim]:
        """Extract claims from text into strongly-typed WitnessClaim objects."""
        raw_claims = self.client.extract_witness_claims(statement_text, statement_id=statement_id)
        claims: list[WitnessClaim] = []
        for raw in raw_claims:
            claims.append(
                WitnessClaim(
                    id=raw["id"],
                    statement_id=raw.get("statement_id", statement_id),
                    original_text=raw.get("original_text", ""),
                    vehicle_desc=raw.get("vehicle_desc", "unknown vehicle"),
                    action=raw.get("action", "unspecified"),
                    direction=raw.get("direction"),
                    target_vehicle=raw.get("target_vehicle"),
                    temporal_anchor=raw.get("temporal_anchor"),
                    time_interval=raw.get("time_interval"),
                    uncertainty=raw.get("uncertainty"),
                    confidence=float(raw.get("confidence", 0.75)),
                    kind=raw.get("kind", "claim"),
                )
            )
        return claims

    def process_file(
        self,
        input_path: Path | str,
        output_path: Optional[Path | str] = None,
        statement_id: str = "W1",
    ) -> list[WitnessClaim]:
        """Process a witness statement file and optionally save to witness_claims.json."""
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Witness statement file not found: {p}")
        text = p.read_text(encoding="utf-8")
        claims = self.parse_statement(text, statement_id=statement_id)
        if output_path:
            out = Path(output_path)
            save_json_records(claims, out)
        return claims


def parse_witness_statement(
    statement_text: str,
    statement_id: str = "W1",
    client: Optional[FeatherlessClient] = None,
) -> list[WitnessClaim]:
    """Convenience functional API for parsing a witness statement."""
    parser = WitnessParser(client=client)
    return parser.parse_statement(statement_text, statement_id=statement_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Witness Statement Claim Extractor")
    parser.add_argument("statement", type=str, help="Witness statement text or file path")
    parser.add_argument("--id", type=str, default="W1", help="Statement identifier")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (e.g. witness_claims.json)")
    parser.add_argument("--mock", action="store_true", help="Force mock / offline mode")
    args = parser.parse_args()

    client = FeatherlessClient(mock=args.mock) if args.mock else default_client
    witness_parser = WitnessParser(client=client)

    if Path(args.statement).exists():
        claims = witness_parser.process_file(args.statement, output_path=args.out, statement_id=args.id)
    else:
        claims = witness_parser.parse_statement(args.statement, statement_id=args.id)
        if args.out:
            save_json_records(claims, args.out)

    print(json.dumps([c.to_dict() for c in claims], indent=2))


if __name__ == "__main__":
    main()
