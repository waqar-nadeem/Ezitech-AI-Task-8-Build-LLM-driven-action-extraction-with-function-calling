"""
Unit tests for the logistics action extraction module.
Uses the offline mock path so tests run without an OpenAI API key.
"""

import json
import os
import sys

import pytest

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from action_extractor import (
    ActionExtractor,
    ExtractionResult,
    LogisticsAction,
    extract_actions,
)


# ---------------------------------------------------------------------------
# Synthetic transcripts (three required samples)
# ---------------------------------------------------------------------------

TRANSCRIPT_1 = (
    "Hey dispatch, can we schedule a pickup for shipment CL-7842 at the "
    "downtown warehouse tomorrow morning around 9 AM? Priority is high."
)

TRANSCRIPT_2 = (
    "This is driver of Truck 12. We are stuck on the highway due to a major "
    "accident. We will be at least 45 minutes late for the delivery to client Apex."
)

TRANSCRIPT_3 = (
    "Just checking in. Nothing special today, traffic is normal, all good."
)


def test_extraction_returns_list_of_action_objects():
    """Extraction function returns a list of action objects with type, entity, and details."""
    result = extract_actions(TRANSCRIPT_1)

    assert isinstance(result, ExtractionResult)
    assert isinstance(result.actions, list)
    assert len(result.actions) >= 1

    action = result.actions[0]
    assert isinstance(action, LogisticsAction)
    assert hasattr(action, "type")
    assert hasattr(action, "entity")
    assert hasattr(action, "details")
    assert isinstance(action.details, dict)
    assert action.type == "schedule_pickup"
    assert "CL-7842" in action.entity or action.entity == "CL-7842"


def test_low_confidence_or_empty_sets_manual_review_flag():
    """When confidence is low / no actions, the function returns a flag for manual review."""
    result = extract_actions(TRANSCRIPT_3)

    assert isinstance(result, ExtractionResult)
    assert result.needs_manual_review is True
    assert result.review_reason is not None
    assert len(result.actions) == 0


def test_three_sample_transcripts_produce_correct_json_shape():
    """Unit test with three sample transcripts verifies correct JSON output structure."""
    samples = [TRANSCRIPT_1, TRANSCRIPT_2, TRANSCRIPT_3]
    expected_types = [
        {"schedule_pickup"},          # transcript 1
        {"report_delay"},             # transcript 2
        set(),                        # transcript 3 – empty
    ]

    for transcript, expected in zip(samples, expected_types):
        result = extract_actions(transcript)
        data = result.to_dict()

        # Top-level keys
        assert "actions" in data
        assert "needs_manual_review" in data
        assert isinstance(data["actions"], list)
        assert isinstance(data["needs_manual_review"], bool)

        # Each action has required fields
        for act in data["actions"]:
            assert "type" in act
            assert "entity" in act
            assert "details" in act
            assert "confidence" in act
            assert 0.0 <= act["confidence"] <= 1.0

        # Type check against expected
        got_types = {a["type"] for a in data["actions"]}
        assert got_types == expected, f"For transcript '{transcript[:40]}...' expected {expected}, got {got_types}"


def test_delay_transcript_extracts_report_delay():
    result = extract_actions(TRANSCRIPT_2)
    assert len(result.actions) >= 1
    assert result.actions[0].type == "report_delay"
    assert "Truck 12" in result.actions[0].entity or "truck" in result.actions[0].entity.lower()


def test_empty_transcript_flags_review():
    result = extract_actions("")
    assert result.needs_manual_review is True
    assert len(result.actions) == 0


def test_to_dict_is_json_serializable():
    result = extract_actions(TRANSCRIPT_1)
    payload = result.to_dict()
    # Must not raise
    serialized = json.dumps(payload, indent=2)
    assert "actions" in serialized
