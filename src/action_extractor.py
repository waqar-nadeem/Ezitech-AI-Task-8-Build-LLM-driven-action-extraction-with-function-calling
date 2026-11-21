"""
LLM-driven logistics action extraction using OpenAI function calling.
Designed for CargoLink Logistics voice-note assistant (Task 2).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    SCHEDULE_PICKUP = "schedule_pickup"
    UPDATE_STATUS = "update_status"
    REPORT_DELAY = "report_delay"
    CONFIRM_DELIVERY = "confirm_delivery"
    REQUEST_SUPPORT = "request_support"
    RESCHEDULE = "reschedule"
    OTHER = "other"


# JSON Schema for a single logistics action (used inside the function parameters)
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": [t.value for t in ActionType],
            "description": "The category of logistics action.",
        },
        "entity": {
            "type": "string",
            "description": "Primary entity involved (shipment ID, vehicle ID, driver name, customer name, location, etc.).",
        },
        "details": {
            "type": "object",
            "description": "Additional structured details specific to the action type.",
            "properties": {
                "location": {"type": ["string", "null"]},
                "time": {"type": ["string", "null"]},
                "status": {"type": ["string", "null"]},
                "reason": {"type": ["string", "null"]},
                "priority": {
                    "type": ["string", "null"],
                    "enum": ["low", "medium", "high", "urgent", None],
                },
                "notes": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Model confidence in this extraction (0.0-1.0).",
        },
        "source_span": {
            "type": ["string", "null"],
            "description": "Short quote from the transcript that supports this action.",
        },
    },
    "required": ["type", "entity", "details", "confidence"],
    "additionalProperties": False,
}


# Tool / function definition for OpenAI function calling
EXTRACT_ACTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_logistics_actions",
        "description": (
            "Extract zero or more logistics actions from a driver/coordinator voice-note transcript. "
            "Return only actions that are clearly actionable. If the transcript is ambiguous or "
            "contains no clear actions, return an empty list and set needs_manual_review=true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": ACTION_SCHEMA,
                    "description": "List of extracted logistics actions.",
                },
                "needs_manual_review": {
                    "type": "boolean",
                    "description": (
                        "True when overall confidence is low, the transcript is noisy/ambiguous, "
                        "or the model is unsure about any critical field."
                    ),
                },
                "review_reason": {
                    "type": ["string", "null"],
                    "description": "Short explanation why manual review is recommended.",
                },
            },
            "required": ["actions", "needs_manual_review"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


SYSTEM_PROMPT = """You are an expert logistics operations assistant for CargoLink Logistics.
Your job is to read a short transcript of a voice note from a driver or coordinator and extract
clear, actionable logistics tasks.

Supported action types:
- schedule_pickup: arrange or confirm a pickup
- update_status: change shipment / vehicle status
- report_delay: notify of a delay with reason
- confirm_delivery: mark delivery as completed
- request_support: ask for help / roadside assistance / extra resources
- reschedule: change a previously planned time/location
- other: anything else that is still an actionable logistics task

Rules:
1. Extract only actions that are explicit or strongly implied.
2. Prefer concrete entity identifiers (shipment IDs, plate numbers, customer names, warehouse codes).
3. Always assign a confidence score between 0.0 and 1.0 for each action.
4. If confidence for any action is below 0.65, or the whole transcript is unclear, set needs_manual_review=true.
5. Keep details concise. Use null for unknown fields.
6. Never invent shipment IDs or times that are not present in the transcript.
7. Return valid JSON that strictly matches the provided function schema.

Few-shot examples:

Transcript: "Hey, can we schedule a pickup for shipment CL-7842 at the downtown warehouse tomorrow morning around 9?"
→ actions: [{type: "schedule_pickup", entity: "CL-7842", details: {location: "downtown warehouse", time: "tomorrow morning around 9", priority: "medium"}, confidence: 0.92, source_span: "schedule a pickup for shipment CL-7842..."}]

Transcript: "Truck 12 is stuck on the highway, traffic jam, will be at least 45 minutes late for the delivery to client Apex."
→ actions: [{type: "report_delay", entity: "Truck 12", details: {reason: "traffic jam", time: "at least 45 minutes late", notes: "delivery to client Apex"}, confidence: 0.88, source_span: "Truck 12 is stuck..."}]

Transcript: "um yeah just checking in nothing special today"
→ actions: [], needs_manual_review: true, review_reason: "No actionable logistics content detected."
"""


@dataclass
class LogisticsAction:
    type: str
    entity: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_span: Optional[str] = None
    needs_review: bool = False  # per-action flag derived from confidence

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ExtractionResult:
    actions: List[LogisticsAction]
    needs_manual_review: bool
    review_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "needs_manual_review": self.needs_manual_review,
            "review_reason": self.review_reason,
            "error": self.error,
        }


class ActionExtractor:
    """
    Extracts structured logistics actions from a transcript using OpenAI function calling.
    Includes low-confidence fallback that flags items for manual review.
    """

    LOW_CONFIDENCE_THRESHOLD = 0.65

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self._client: Optional[OpenAI] = None

        if self.api_key:
            self._client = OpenAI(api_key=self.api_key)
        else:
            logger.warning(
                "OPENAI_API_KEY not set. extract() will return a deterministic mock result "
                "suitable for offline testing."
            )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_openai(self, transcript: str) -> Dict[str, Any]:
        assert self._client is not None
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Transcript:\n\n{transcript.strip()}\n\nExtract the logistics actions.",
                },
            ],
            tools=[EXTRACT_ACTIONS_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "extract_logistics_actions"},
            },
        )
        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError("Model did not return a function call.")

        tool_call = message.tool_calls[0]
        if tool_call.function.name != "extract_logistics_actions":
            raise ValueError(f"Unexpected function: {tool_call.function.name}")

        return json.loads(tool_call.function.arguments)

    def _mock_extract(self, transcript: str) -> Dict[str, Any]:
        """Deterministic offline fallback used when no API key is present."""
        text = transcript.lower()
        actions = []

        # Simple keyword heuristics for demo / tests
        if "pickup" in text or "schedule" in text:
            entity = "unknown"
            m = re.search(r"(CL-\d+)", transcript, re.I)
            if m:
                entity = m.group(1)
            else:
                m = re.search(r"shipment\s+([A-Za-z0-9\-]+)", transcript, re.I)
                if m:
                    entity = m.group(1)
            actions.append(
                {
                    "type": "schedule_pickup",
                    "entity": entity,
                    "details": {"notes": "Detected via mock heuristic"},
                    "confidence": 0.75,
                    "source_span": transcript[:80],
                }
            )
        if "delay" in text or "late" in text or "stuck" in text:
            entity = "vehicle"
            m = re.search(r"(truck\s+\d+|vehicle\s+\w+)", transcript, re.I)
            if m:
                entity = m.group(0)
            actions.append(
                {
                    "type": "report_delay",
                    "entity": entity,
                    "details": {"reason": "detected delay language"},
                    "confidence": 0.80,
                    "source_span": transcript[:80],
                }
            )
        if "delivered" in text or "delivery complete" in text:
            actions.append(
                {
                    "type": "confirm_delivery",
                    "entity": "shipment",
                    "details": {},
                    "confidence": 0.85,
                    "source_span": transcript[:80],
                }
            )

        needs_review = len(actions) == 0 or any(a["confidence"] < self.LOW_CONFIDENCE_THRESHOLD for a in actions)
        reason = None
        if not actions:
            reason = "No clear logistics actions detected by mock extractor."
        elif needs_review:
            reason = "One or more actions have confidence below threshold."

        return {
            "actions": actions,
            "needs_manual_review": needs_review,
            "review_reason": reason,
        }

    def extract(self, transcript: str) -> ExtractionResult:
        """
        Main entry point.
        Returns ExtractionResult with list of LogisticsAction objects and review flags.
        """
        if not transcript or not transcript.strip():
            return ExtractionResult(
                actions=[],
                needs_manual_review=True,
                review_reason="Empty transcript.",
            )

        try:
            if self._client is None:
                raw = self._mock_extract(transcript)
            else:
                raw = self._call_openai(transcript)
        except Exception as exc:
            logger.exception("Extraction failed")
            return ExtractionResult(
                actions=[],
                needs_manual_review=True,
                review_reason=f"Extraction error: {exc}",
                error=str(exc),
            )

        actions: List[LogisticsAction] = []
        for item in raw.get("actions", []):
            conf = float(item.get("confidence", 0.5))
            action = LogisticsAction(
                type=item.get("type", ActionType.OTHER.value),
                entity=item.get("entity", "unknown"),
                details=item.get("details") or {},
                confidence=conf,
                source_span=item.get("source_span"),
                needs_review=conf < self.LOW_CONFIDENCE_THRESHOLD,
            )
            actions.append(action)

        # Aggregate review flag
        needs_review = bool(raw.get("needs_manual_review", False))
        if any(a.needs_review for a in actions):
            needs_review = True

        return ExtractionResult(
            actions=actions,
            needs_manual_review=needs_review,
            review_reason=raw.get("review_reason"),
            raw_response=raw,
        )


def extract_actions(transcript: str, **kwargs) -> ExtractionResult:
    """Convenience function used by Streamlit UI and tests."""
    extractor = ActionExtractor(**kwargs)
    return extractor.extract(transcript)
