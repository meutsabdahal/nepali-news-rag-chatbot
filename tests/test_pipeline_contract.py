from types import SimpleNamespace
import unittest

from langchain_core.documents import Document

from nepali_news_rag.pipeline import NepaliNewsPipeline


class _StubRetriever:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def retrieve(self, query: str, k: int) -> list[Document]:
        return self._docs[:k]


class _StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


class PipelineContractTest(unittest.TestCase):
    def _build_pipeline(
        self, docs: list[Document], response: str
    ) -> NepaliNewsPipeline:
        pipeline = NepaliNewsPipeline.__new__(NepaliNewsPipeline)
        pipeline.settings = SimpleNamespace(retriever_k=3)
        pipeline.retriever = _StubRetriever(docs)
        pipeline.llm = _StubLLM(response)
        pipeline._max_prompt_chars = {"Nepali": 1400, "English": 3200}
        return pipeline

    def test_response_contract_has_required_fields(self) -> None:
        docs = [
            Document(
                page_content="Tourism increased this season.",
                metadata={"source": "Kantipur", "heading": "Tourism Update"},
            )
        ]
        pipeline = self._build_pipeline(docs, "Tourism increased this season.")
        output = pipeline.run("What is the tourism update?", language="English")

        self.assertIn("answer", output)
        self.assertIn("success", output)
        self.assertIn("route", output)
        self.assertIn("guardrail_type", output)
        self.assertIn("sources", output)
        self.assertEqual(output["route"], "RAG")
        self.assertTrue(len(output["sources"]) >= 1)

    def test_prediction_query_routes_to_oos(self) -> None:
        pipeline = self._build_pipeline([], "unused")
        output = pipeline.run(
            "Can you predict next year political outcome?", language="English"
        )
        self.assertEqual(output["route"], "OOS")
        self.assertTrue(output["success"])


if __name__ == "__main__":
    unittest.main()
