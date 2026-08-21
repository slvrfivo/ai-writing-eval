from __future__ import annotations

import math
import unittest

from src.evaluate import (
    PredictionValidationError,
    evaluate_predictions,
    rmse,
    round_half_up_score,
    spearman_correlation,
    validate_prediction,
)


class RoundHalfUpTests(unittest.TestCase):
    def test_required_half_values_round_up(self) -> None:
        cases = {1.5: 2, 2.5: 3, 3.5: 4, 4.5: 5}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(round_half_up_score(value), expected)

    def test_values_outside_official_range_raise_before_rounding(self) -> None:
        for value in (0.9, 5.1):
            with self.subTest(value=value):
                with self.assertRaises(PredictionValidationError):
                    round_half_up_score(value)

    def test_python_bankers_rounding_is_not_used(self) -> None:
        self.assertEqual(round_half_up_score(2.5), 3)
        self.assertNotEqual(round_half_up_score(2.5), round(2.5))


class PredictionValidationTests(unittest.TestCase):
    def valid_prediction(self) -> dict:
        return {
            "content": {"score": 2.5, "rationale": "내용 근거"},
            "organization": {"score": 3, "rationale": "구성 근거"},
            "expression": {"score": 4.49, "rationale": "표현 근거"},
        }

    def test_validation_returns_rounded_scores(self) -> None:
        self.assertEqual(
            validate_prediction(self.valid_prediction(), require_rationale=True),
            {"content": 3, "organization": 3, "expression": 4},
        )

    def test_missing_dimension_raises(self) -> None:
        prediction = self.valid_prediction()
        del prediction["expression"]
        with self.assertRaises(PredictionValidationError):
            validate_prediction(prediction)

    def test_non_numeric_score_raises(self) -> None:
        prediction = self.valid_prediction()
        prediction["content"]["score"] = "3"
        with self.assertRaises(PredictionValidationError):
            validate_prediction(prediction)

    def test_required_rationale_must_exist_and_be_string(self) -> None:
        missing = self.valid_prediction()
        del missing["content"]["rationale"]
        with self.assertRaises(PredictionValidationError):
            validate_prediction(missing, require_rationale=True)

        wrong_type = self.valid_prediction()
        wrong_type["content"]["rationale"] = 123
        with self.assertRaises(PredictionValidationError):
            validate_prediction(wrong_type, require_rationale=True)


class MetricTests(unittest.TestCase):
    def test_rmse(self) -> None:
        self.assertAlmostEqual(rmse([1, 2, 3], [1, 2, 4]), math.sqrt(1 / 3))

    def test_spearman_with_and_without_ties(self) -> None:
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)

    def test_constant_values_have_undefined_spearman(self) -> None:
        self.assertIsNone(spearman_correlation([1, 2, 3], [2, 2, 2]))

    def test_evaluator_ignores_average_and_supports_official_wrapper(self) -> None:
        ground_truth = [
            {
                "id": f"id-{index}",
                "document_id": f"doc-{index}",
                "score": {
                    "content": score,
                    "organization": score,
                    "expression": score,
                    "average": 999,
                },
            }
            for index, score in enumerate((1, 3, 5), start=1)
        ]
        predictions = [
            {
                "essay_id": f"doc-{index}",
                "judge": {
                    dimension: {"score": score, "rationale": "근거"}
                    for dimension in ("content", "organization", "expression")
                },
            }
            for index, score in enumerate((1, 3, 5), start=1)
        ]

        result = evaluate_predictions(
            ground_truth, predictions, require_rationale=True
        )
        self.assertEqual(result["n_samples"], 3)
        for dimension in ("content", "organization", "expression"):
            self.assertEqual(result["dimensions"][dimension]["rmse"], 0.0)
            self.assertAlmostEqual(result["dimensions"][dimension]["spearman"], 1.0)
        self.assertEqual(result["mean"]["rmse"], 0.0)
        self.assertAlmostEqual(result["mean"]["spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
