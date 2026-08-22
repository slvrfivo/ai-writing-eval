from __future__ import annotations

import json
import unittest

from src.llm.parsing import parse_model_output


def valid_payload() -> dict:
    return {
        "content": {"score": 3, "rationale": "주장과 근거를 함께 제시한다."},
        "organization": {"score": 4, "rationale": "서론과 본론, 결론이 드러난다."},
        "expression": {"score": 2, "rationale": "어색한 문장이 반복된다."},
    }


class ModelOutputParsingTests(unittest.TestCase):
    def assert_failure(self, payload: str, code: str, path: str | None = None) -> None:
        result = parse_model_output(payload)
        self.assertFalse(result.ok)
        self.assertIsNone(result.value)
        self.assertIsNotNone(result.failure)
        assert result.failure is not None
        self.assertEqual(result.failure.code, code)
        if path is not None:
            self.assertEqual(result.failure.path, path)

    def test_valid_json_passes(self) -> None:
        payload = valid_payload()
        result = parse_model_output(json.dumps(payload, ensure_ascii=False))
        self.assertTrue(result.ok)
        self.assertEqual(result.value, payload)
        self.assertIsNone(result.failure)

    def test_float_score_is_rejected(self) -> None:
        payload = valid_payload()
        payload["content"]["score"] = 3.0
        self.assert_failure(
            json.dumps(payload, ensure_ascii=False),
            "invalid_score_type",
            "content.score",
        )

    def test_bool_score_is_rejected(self) -> None:
        payload = valid_payload()
        payload["organization"]["score"] = True
        self.assert_failure(
            json.dumps(payload, ensure_ascii=False),
            "invalid_score_type",
            "organization.score",
        )

    def test_out_of_range_score_is_rejected(self) -> None:
        for score in (0, 6):
            with self.subTest(score=score):
                payload = valid_payload()
                payload["expression"]["score"] = score
                self.assert_failure(
                    json.dumps(payload, ensure_ascii=False),
                    "score_out_of_range",
                    "expression.score",
                )

    def test_empty_rationale_is_rejected_after_strip(self) -> None:
        payload = valid_payload()
        payload["content"]["rationale"] = " \t\n "
        self.assert_failure(
            json.dumps(payload, ensure_ascii=False),
            "empty_rationale",
            "content.rationale",
        )

    def test_markdown_code_fence_is_rejected(self) -> None:
        fenced = "```json\n" + json.dumps(valid_payload(), ensure_ascii=False) + "\n```"
        self.assert_failure(fenced, "markdown_code_fence")

    def test_missing_dimension_is_rejected(self) -> None:
        payload = valid_payload()
        del payload["organization"]
        self.assert_failure(
            json.dumps(payload, ensure_ascii=False),
            "missing_dimension",
            "organization",
        )

    def test_missing_field_is_rejected(self) -> None:
        payload = valid_payload()
        del payload["expression"]["rationale"]
        self.assert_failure(
            json.dumps(payload, ensure_ascii=False),
            "missing_field",
            "expression.rationale",
        )


if __name__ == "__main__":
    unittest.main()
