from __future__ import annotations

from abc import ABC, abstractmethod

from groq import Groq
from huggingface_hub import InferenceClient
from ollama import Client as OllamaClientSDK

from .config import Settings


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    def __init__(self, settings: Settings) -> None:
        self._client = OllamaClientSDK(host=settings.ollama_host)
        self._model = settings.ollama_model

    def generate(self, prompt: str) -> str:
        response = self._client.generate(model=self._model, prompt=prompt)
        return response.get("response", "").strip()


class GroqClient(BaseLLMClient):
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_model
        self._max_tokens = settings.response_max_tokens

    def generate(self, prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=self._max_tokens,
        )
        message = completion.choices[0].message.content
        return (message or "").strip()


class HuggingFaceClientLLM(BaseLLMClient):
    def __init__(self, settings: Settings) -> None:
        self._client = InferenceClient(
            model=settings.hf_model, token=settings.hf_api_key
        )
        self._max_tokens = settings.response_max_tokens

    def generate(self, prompt: str) -> str:
        # Prefer chat completion for instruct/chat-tuned models.
        try:
            chat_output = self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=0.1,
            )
            choices = getattr(chat_output, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content.strip()
        except Exception:
            pass

        try:
            output = self._client.text_generation(
                prompt=prompt,
                max_new_tokens=self._max_tokens,
                temperature=0.1,
                return_full_text=False,
            )
        except TypeError:
            output = self._client.text_generation(
                prompt, max_new_tokens=self._max_tokens
            )

        if isinstance(output, str):
            return output.strip()
        return str(output).strip()


def get_llm_client(settings: Settings) -> BaseLLMClient:
    provider = settings.llm_provider
    if provider == "ollama":
        return OllamaClient(settings)
    if provider == "groq":
        return GroqClient(settings)
    if provider in {"hf", "huggingface"}:
        return HuggingFaceClientLLM(settings)
    raise ValueError(f"Unsupported LLM provider: {provider}")
