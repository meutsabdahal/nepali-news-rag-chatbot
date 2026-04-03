import unittest
from unittest.mock import patch

from nepali_news_rag.config import get_settings
from nepali_news_rag.llm import get_llm_client


class LLMProviderFactoryTest(unittest.TestCase):
    def test_unsupported_provider_raises(self) -> None:
        with patch.dict("os.environ", {"LLM_PROVIDER": "unknown"}, clear=False):
            settings = get_settings()
            with self.assertRaises(ValueError):
                get_llm_client(settings)

    def test_groq_without_key_raises(self) -> None:
        with patch.dict(
            "os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": ""}, clear=False
        ):
            settings = get_settings()
            with self.assertRaises(RuntimeError):
                get_llm_client(settings)


if __name__ == "__main__":
    unittest.main()
