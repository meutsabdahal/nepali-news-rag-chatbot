from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    blocked: bool
    guardrail_type: str | None
    message: str | None


_PREDICTION_PATTERNS = [
    "predict",
    "forecast",
    "future price",
    "next year",
    "will happen",
    "भविष्यवाणी",
    "अर्को वर्ष",
    "हुन्छ कि",
]


def evaluate_guardrails(query: str, language: str) -> GuardrailResult:
    lowered = query.lower()
    if any(token in lowered for token in _PREDICTION_PATTERNS):
        if language == "Nepali":
            return GuardrailResult(
                blocked=True,
                guardrail_type="prediction",
                message="म भविष्य सम्बन्धी अड्कलबाजी दिन सक्दिन। म उपलब्ध समाचारको आधारमा मात्र उत्तर दिन्छु।",
            )
        return GuardrailResult(
            blocked=True,
            guardrail_type="prediction",
            message="I cannot provide speculative predictions. I can answer only from available news evidence.",
        )

    return GuardrailResult(blocked=False, guardrail_type=None, message=None)
