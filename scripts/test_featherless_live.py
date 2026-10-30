"""Live Featherless AI witness claim extraction tester.

Usage:
  python scripts/test_featherless_live.py --api-key <YOUR_FEATHERLESS_API_KEY>
  or
  set FEATHERLESS_API_KEY=<KEY>
  python scripts/test_featherless_live.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intelligence.featherless_client import FeatherlessClient

SAMPLE_STATEMENT = (
    "I saw a black SUV moving very fast. A white SUV entered from the right and collided with it. "
    "The black SUV rotated and stopped after the collision."
)


def run_live_test(api_key: str, model: str | None = None) -> bool:
    print("=" * 60)
    print("FEATHERLESS LIVE API TEST")
    print("=" * 60)
    print(f"Endpoint: https://api.featherless.ai/v1")
    print(f"Model: {model or 'default'}")
    print(f"Test Statement: \"{SAMPLE_STATEMENT}\"\n")

    client = FeatherlessClient(api_key=api_key, text_model=model, mock=False)
    if client.is_mock():
        print("[ERROR] Client fell back to mock mode. API key missing.")
        return False

    try:
        print("[INFO] Sending extraction request to Featherless AI...")
        claims = client.extract_witness_claims(SAMPLE_STATEMENT, statement_id="W-LIVE")
        print(f"[SUCCESS] Received {len(claims)} extracted claims from live model:\n")
        print(json.dumps(claims, indent=2))

        # Verification checks
        print("\n--- Verification Checklist ---")
        print(f"1. Multiple claims extracted: {len(claims) >= 3} ({len(claims)} claims)")
        actions = [c.get("action") for c in claims]
        print(f"2. Actions detected: {actions}")
        vehicles = [c.get("vehicle_desc") for c in claims]
        print(f"3. Vehicles associated: {set(vehicles)}")
        no_invented_times = all(c.get("time_interval") is None for c in claims)
        print(f"4. No invented timestamps (time_interval is None): {no_invented_times}")
        verbatim_quotes = all(c.get("original_text") in SAMPLE_STATEMENT for c in claims if c.get("original_text"))
        print(f"5. Quotes present in statement: {verbatim_quotes}")
        return True
    except Exception as e:
        print(f"[ERROR] Live call failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Featherless Extraction Test")
    parser.add_argument("--api-key", type=str, default=None, help="Featherless API Key")
    parser.add_argument("--model", type=str, default=None, help="Text model override")
    args = parser.parse_args()

    key = args.api_key or os.environ.get("FEATHERLESS_API_KEY")
    if not key:
        print("[NOTICE] No FEATHERLESS_API_KEY provided.")
        print("To run live test, pass --api-key <YOUR_KEY> or set FEATHERLESS_API_KEY.")
        print("\nRunning in MOCK mode to demonstrate extraction logic:")
        client = FeatherlessClient(mock=True)
        claims = client.extract_witness_claims(SAMPLE_STATEMENT)
        print(json.dumps(claims, indent=2))
        return

    success = run_live_test(key, model=args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
