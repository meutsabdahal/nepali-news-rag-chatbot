from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp

from .config import Settings


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class LlamaCppClient(BaseLLMClient):
    def __init__(self, settings: Settings) -> None:
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = settings.model_cache_dir / settings.llama_gguf_file

        if not model_path.exists():
            hf_hub_download(
                repo_id=settings.llama_gguf_repo,
                filename=settings.llama_gguf_file,
                local_dir=str(settings.model_cache_dir),
            )

        self._llm = LlamaCpp(
            model_path=str(model_path),
            n_ctx=2048,
            n_batch=512,
            n_gpu_layers=0,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            max_tokens=settings.response_max_tokens,
            verbose=False,
        )

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt)


def _not_ready(provider_name: str) -> BaseLLMClient:
    class _ProviderNotReady(BaseLLMClient):
        def generate(self, prompt: str) -> str:
            raise RuntimeError(
                f"Provider '{provider_name}' is not implemented yet for this workspace."
            )

    return _ProviderNotReady()


def get_llm_client(settings: Settings) -> BaseLLMClient:
    provider = settings.llm_provider
    if provider == "llama_cpp":
        return LlamaCppClient(settings)
    if provider == "ollama":
        return _not_ready("ollama")
    if provider == "groq":
        return _not_ready("groq")
    if provider in {"hf", "huggingface"}:
        return _not_ready("huggingface")
    raise ValueError(f"Unsupported LLM provider: {provider}")
