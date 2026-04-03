from __future__ import annotations

from dataclasses import dataclass

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


def route_query(query: str, language: str) -> RouteDecision:
    guardrail = evaluate_guardrails(query, language)
    if guardrail.blocked:
        return RouteDecision(route="OOS", guardrail=guardrail)

    lowered = query.strip().lower()
    if lowered in _DIRECT_TOKENS:
        return RouteDecision(route="DIRECT", guardrail=guardrail)

    return RouteDecision(route="RAG", guardrail=guardrail)
