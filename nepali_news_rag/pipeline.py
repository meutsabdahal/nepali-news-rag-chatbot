from __future__ import annotations

from dataclasses import asdict, dataclass

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

        if "cannot find" in answer.lower() or "don't have" in answer.lower():
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

        used_docs = docs[: len(context_parts)]
        return "\n\n".join(context_parts), used_docs

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
