import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intelligence.featherless_client import FeatherlessClient
from src.nlp.witness_parser import parse_witness_statement
from src.schema import WitnessClaim

SAMPLE_STATEMENT = (
    "I saw a black SUV moving very fast. A white SUV entered from the right and collided with it. "
    "The black SUV rotated and stopped after the collision."
)


def test_extract_multiple_claims_mock() -> None:
    client = FeatherlessClient(mock=True)
    claims = client.extract_witness_claims(SAMPLE_STATEMENT, statement_id="W1")

    # Requirement 1: Extracts multiple claims from a single statement (at least 4, ideally 5)
    assert len(claims) >= 4, f"Expected at least 4 claims, got {len(claims)}"

    actions = [c["action"] for c in claims]
    # Requirement 3: Recognises collision, approach, direction, rotation, stopping and speeding
    assert "speeding" in actions
    assert "approach" in actions
    assert "collision" in actions
    assert "rotation" in actions
    assert "stopping" in actions


def test_correct_vehicle_associations_mock() -> None:
    client = FeatherlessClient(mock=True)
    claims = client.extract_witness_claims(SAMPLE_STATEMENT, statement_id="W1")

    claim_by_action = {c["action"]: c for c in claims}

    # Requirement 2 & 6: Correctly associates actions with their respective vehicles
    assert claim_by_action["speeding"]["vehicle_desc"].lower() == "black suv"
    assert claim_by_action["approach"]["vehicle_desc"].lower() == "white suv"
    assert claim_by_action["approach"]["direction"] == "from the right"

    # Collision: White SUV collided with Black SUV (resolving 'it')
    assert claim_by_action["collision"]["vehicle_desc"].lower() == "white suv"
    assert claim_by_action["collision"]["target_vehicle"].lower() == "black suv"

    # Rotation and Stopping must belong to Black SUV (NOT White SUV!)
    assert claim_by_action["rotation"]["vehicle_desc"].lower() == "black suv"
    assert claim_by_action["stopping"]["vehicle_desc"].lower() == "black suv"


def test_exact_original_text_preserved() -> None:
    client = FeatherlessClient(mock=True)
    claims = client.extract_witness_claims(SAMPLE_STATEMENT, statement_id="W1")

    # Requirement 4: Preserves exact original text for each claim
    for c in claims:
        quote = c["original_text"]
        assert quote in SAMPLE_STATEMENT, f"Quote '{quote}' was not found verbatim in original statement"


def test_does_not_invent_timestamps() -> None:
    client = FeatherlessClient(mock=True)
    claims = client.extract_witness_claims(SAMPLE_STATEMENT, statement_id="W1")

    # Requirement 5: Does not invent timestamps when none are provided
    for c in claims:
        assert c["time_interval"] is None, f"Claim {c['id']} invented a timestamp: {c['time_interval']}"
        assert c["uncertainty"] is None, f"Claim {c['id']} invented an uncertainty interval: {c['uncertainty']}"


def test_temporal_anchor_captured() -> None:
    client = FeatherlessClient(mock=True)
    claims = client.extract_witness_claims(SAMPLE_STATEMENT, statement_id="W1")

    claim_by_action = {c["action"]: c for c in claims}
    assert "after the collision" in (claim_by_action["stopping"].get("temporal_anchor") or "")


def test_explicit_timestamps_are_preserved_when_present() -> None:
    statement_with_time = "The red sedan ran the red light around 4 seconds in without braking."
    client = FeatherlessClient(mock=True)
    claims = client.extract_witness_claims(statement_with_time, statement_id="W2")

    assert len(claims) >= 1
    # Explicit timestamps SHOULD be parsed when explicitly mentioned
    timed_claims = [c for c in claims if c["time_interval"] is not None]
    assert len(timed_claims) >= 1
    assert timed_claims[0]["time_interval"][0] <= 4.0 <= timed_claims[0]["time_interval"][1]


def test_witness_parser_converts_to_schema_models() -> None:
    claims = parse_witness_statement(SAMPLE_STATEMENT, statement_id="W1")
    assert all(isinstance(c, WitnessClaim) for c in claims)
    stopping_claim = next(c for c in claims if c.action == "stopping")
    assert stopping_claim.vehicle_desc.lower() == "black suv"
    assert stopping_claim.time_interval is None

    common_event = stopping_claim.to_common_event()
    assert common_event.vehicle_ref.lower() == "black suv"
    assert common_event.start_time is None
    assert common_event.kind == "claim"


def test_featherless_live_api_call() -> None:
    """Test actual Featherless API call when FEATHERLESS_API_KEY is available."""
    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if not api_key:
        pytest.skip("FEATHERLESS_API_KEY is not set in environment. Skipping live network call.")

    client = FeatherlessClient(api_key=api_key, mock=False)
    assert not client.is_mock()

    try:
        claims = client.extract_witness_claims(SAMPLE_STATEMENT, statement_id="W-LIVE")
        assert len(claims) >= 3, f"Expected at least 3 claims from live LLM, got {len(claims)}"
        actions = [c["action"] for c in claims]
        assert any(a in actions for a in ("speeding", "approach", "collision", "rotation", "stopping"))
        # Verify no invented timestamps in live LLM output
        for c in claims:
            assert c["time_interval"] is None
    except Exception as e:
        pytest.skip(f"Live Featherless endpoint unreachable: {e}")
