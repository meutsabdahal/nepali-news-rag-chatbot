from __future__ import annotations

from dataclasses import dataclass
import re

from .guardrails import GuardrailResult, evaluate_guardrails


@dataclass(frozen=True)
class RouteDecision:
    route: str
    guardrail: GuardrailResult


_DIRECT_TOKENS = {
    "hi",
    "hello",
    "hey",
    "who are you",
    "what can you do",
    "नमस्ते",
    "तिमी को हौ",
}

_GENERAL_PATTERNS = [
    r"^who\s+is\b",
    r"^who\s+was\b",
    r"^where\s+is\b",
    r"^नेपालको\s+प्रधानमन्त्री\b",
    r"प्रधानमन्त्री\s+को\s+हो",
    r"प्रधानमन्त्री\s+को\s+हुन्",
]

_NEWS_HINTS = {
    "news",
    "reported",
    "recent",
    "recently",
    "article",
    "समाचार",
    "हाल",
    "रिपोर्ट",
}


def _is_general_query(lowered: str) -> bool:
    if any(hint in lowered for hint in _NEWS_HINTS):
        return False
    return any(re.search(pattern, lowered) for pattern in _GENERAL_PATTERNS)


def route_query(query: str, language: str) -> RouteDecision:
    guardrail = evaluate_guardrails(query, language)
    if guardrail.blocked:
        return RouteDecision(route="OOS", guardrail=guardrail)

    lowered = query.strip().lower()
    if lowered in _DIRECT_TOKENS:
        return RouteDecision(route="DIRECT", guardrail=guardrail)

    if _is_general_query(lowered):
        return RouteDecision(route="DIRECT", guardrail=guardrail)

    return RouteDecision(route="RAG", guardrail=guardrail)
