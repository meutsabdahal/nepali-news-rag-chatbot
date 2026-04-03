import unittest

from nepali_news_rag.guardrails import evaluate_guardrails


class GuardrailsTest(unittest.TestCase):
    def test_prediction_is_blocked(self) -> None:
        result = evaluate_guardrails(
            "Can you predict what will happen next year?", "English"
        )
        self.assertTrue(result.blocked)
        self.assertEqual(result.guardrail_type, "prediction")

    def test_regular_news_query_is_allowed(self) -> None:
        result = evaluate_guardrails("What happened in tourism this week?", "English")
        self.assertFalse(result.blocked)
        self.assertIsNone(result.guardrail_type)


if __name__ == "__main__":
    unittest.main()
