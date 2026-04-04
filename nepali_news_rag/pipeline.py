from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import pandas as pd

from .config import get_settings
from .language_detector import detect_language
from .llm import get_llm_client
from .prompts import build_direct_prompt, build_rag_prompt
from .retriever import Retriever
from .router import route_query


@dataclass
class PipelineResponse:
    answer: str
    success: bool
    route: str
    guardrail_type: str | None
    sources: list[str]


class NepaliNewsPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retriever = Retriever(self.settings)
        self.llm = get_llm_client(self.settings)

        # Prompt context budgets by language family.
        self._max_prompt_chars = {"Nepali": 1400, "English": 3200}

    def run(self, query: str, language: str | None = None) -> dict:
        if not query or not query.strip():
            return asdict(
                PipelineResponse(
                    answer="Please provide a question.",
                    success=False,
                    route="INVALID",
                    guardrail_type=None,
                    sources=[],
                )
            )

        target_language = language or detect_language(query)
        decision = route_query(query, target_language)

        if decision.route == "OOS":
            return asdict(
                PipelineResponse(
                    answer=decision.guardrail.message or "Request blocked.",
                    success=True,
                    route="OOS",
                    guardrail_type=decision.guardrail.guardrail_type,
                    sources=[],
                )
            )

        if decision.route == "DIRECT":
            prompt = build_direct_prompt(query, target_language)
            try:
                answer = self._clean_response(self.llm.generate(prompt))
            except Exception as exc:
                return asdict(
                    PipelineResponse(
                        answer=self._provider_error_message(exc, target_language),
                        success=False,
                        route="DIRECT",
                        guardrail_type="provider_error",
                        sources=[],
                    )
                )
            return asdict(
                PipelineResponse(
                    answer=answer,
                    success=True,
                    route="DIRECT",
                    guardrail_type=None,
                    sources=[],
                )
            )

        docs = self.retriever.retrieve(query, k=self.settings.retriever_k)
        if not docs:
            return asdict(
                PipelineResponse(
                    answer="I cannot find the answer in the provided news.",
                    success=True,
                    route="RAG",
                    guardrail_type=None,
                    sources=[],
                )
            )

        context, used_docs = self._build_context(query, target_language, docs)
        prompt = build_rag_prompt(
            context=context, question=query, target_language=target_language
        )
        try:
            answer = self._clean_response(self.llm.generate(prompt))
        except Exception as exc:
            return asdict(
                PipelineResponse(
                    answer=self._provider_error_message(exc, target_language),
                    success=False,
                    route="RAG",
                    guardrail_type="provider_error",
                    sources=[],
                )
            )

        if self._looks_unresolved(answer):
            fallback = self._keyword_fallback_answer(query, target_language)
            if fallback is not None:
                answer, sources = fallback
                return asdict(
                    PipelineResponse(
                        answer=answer,
                        success=True,
                        route="RAG",
                        guardrail_type=None,
                        sources=sources,
                    )
                )

            answer = "I cannot find the answer in the provided news."

        sources = [
            f"{d.metadata.get('source', 'Unknown')} - {d.metadata.get('heading', 'N/A')}"
            for d in used_docs
        ]

        return asdict(
            PipelineResponse(
                answer=answer,
                success=True,
                route="RAG",
                guardrail_type=None,
                sources=sources,
            )
        )

    def _build_context(self, query: str, language: str, docs: list):
        budget = self._max_prompt_chars.get(language, 1400)
        context_parts: list[str] = []
        char_count = 0

        for doc in docs:
            remaining = budget - (len(query) + char_count)
            if remaining <= 0:
                break

            snippet = doc.page_content
            if len(snippet) > remaining:
                snippet = snippet[:remaining]

            context_parts.append(snippet)
            char_count += len(snippet) + 2

        if not context_parts and docs:
            fallback_chars = max(300, budget // 2)
            context_parts.append(docs[0].page_content[:fallback_chars])

        used_docs = docs[: len(context_parts)]
        return "\n\n".join(context_parts), used_docs

    def _keyword_fallback_answer(
        self, query: str, language: str
    ) -> tuple[str, list[str]] | None:
        try:
            raw_path = self.settings.raw_csv_path
        except Exception:
            return None

        if not raw_path.exists():
            return None

        try:
            df = pd.read_csv(
                raw_path, usecols=["source", "heading", "content", "category"]
            )
        except Exception:
            return None

        merged = (
            df["heading"].fillna("").astype(str)
            + " "
            + df["content"].fillna("").astype(str)
        ).str.lower()

        tokens = [
            t
            for t in re.findall(r"[\w\u0900-\u097F]+", query.lower())
            if len(t) >= 3 or ("\u0900" <= t[0] <= "\u097f")
        ]
        stop_tokens = {
            "what",
            "has",
            "been",
            "reported",
            "recently",
            "about",
            "who",
            "the",
            "is",
            "of",
            "in",
            "के",
            "छ",
            "को",
            "बारे",
            "भन्छ",
        }
        tokens = [t for t in tokens if t not in stop_tokens]

        expansions = {
            "pokhara": ["पोखरा"],
            "kathmandu": ["काठमाडौं", "काठमाडौँ"],
            "tourism": ["पर्यटन"],
            "pollution": ["प्रदूषण"],
            "education": ["शिक्षा"],
            "nepal": ["नेपाल"],
            "prime": ["प्रधानमन्त्री"],
            "minister": ["प्रधानमन्त्री"],
        }
        expanded: list[str] = []
        for t in tokens:
            expanded.append(t)
            expanded.extend(expansions.get(t, []))
        tokens = list(dict.fromkeys(expanded))

        if not tokens:
            return None

        score = pd.Series(0, index=df.index, dtype="int64")
        for token in tokens:
            score += merged.str.contains(re.escape(token), regex=True).astype("int64")

        candidates = df[score > 0].copy()
        if candidates.empty:
            return None

        candidates["score"] = score[score > 0]
        candidates = candidates.sort_values("score", ascending=False).head(3)

        context_parts = []
        sources: list[str] = []
        for row in candidates.itertuples(index=False):
            text = f"Category: {getattr(row, 'category', 'General')} | Title: {getattr(row, 'heading', '')}\n{getattr(row, 'content', '')}"
            context_parts.append(text[:1200])
            sources.append(
                f"{getattr(row, 'source', 'Unknown')} - {getattr(row, 'heading', 'N/A')}"
            )

        prompt = build_rag_prompt(
            context="\n\n".join(context_parts),
            question=query,
            target_language=language,
        )

        try:
            answer = self._clean_response(self.llm.generate(prompt))
        except Exception:
            return None

        if not answer.strip():
            return None
        return answer, sources

    @staticmethod
    def _looks_unresolved(answer: str) -> bool:
        normalized = answer.strip().lower()
        unresolved_signals = [
            "i cannot find the answer in the provided news",
            "i don't have information",
            "i do not have information",
            "does not mention",
            "not available in the context",
            "कुनै जानकारी",
            "जानकारी छैन",
            "प्रदान गरिएको छैन",
            "प्रस्तुत गरिएको छैन",
            "उल्लेख गरिएको छैन",
        ]
        return any(sig in normalized for sig in unresolved_signals)

    @staticmethod
    def _clean_response(response: str) -> str:
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        return response.replace("<|im_end|>", "").strip()

    @staticmethod
    def _provider_error_message(error: Exception, language: str) -> str:
        message = str(error).lower()

        if "system memory" in message or "out of memory" in message:
            if language == "Nepali":
                return (
                    "हालको मेशिनमा Llama 3.1 8B चलाउन पर्याप्त मेमोरी छैन। "
                    "कृपया रिमोट प्रदायक (Groq/Hugging Face) प्रयोग गर्नुहोस् वा RAM बढाउनुहोस्।"
                )
            return (
                "This machine does not have enough memory to run Llama 3.1 8B locally. "
                "Use a remote provider (Groq/Hugging Face) or run on a higher-RAM system."
            )

        if language == "Nepali":
            return "LLM प्रदायकबाट उत्तर प्राप्त गर्न सकिएन। कृपया प्रदायक सेटिङ र API key जाँच गर्नुहोस्।"
        return "Could not get a response from the configured LLM provider. Check provider settings and credentials."
