from __future__ import annotations

import unittest

from src.baseline import (
    build_global_predictions,
    build_prompt_predictions,
    calculate_train_statistics,
    run_baselines,
)


def train_record(identifier: str, prompt_num: str, score: float) -> dict:
    return {
        "id": identifier,
        "prompt_num": prompt_num,
        "score": {
            "content": score,
            "organization": score + 0.1,
            "expression": score + 0.2,
            "average": 999,
        },
    }


class TrainStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = [
            train_record("t1", "Q1", 2.0),
            train_record("t2", "Q1", 4.0),
            train_record("t3", "Q2", 3.0),
        ]

    def test_global_and_prompt_means_use_train_scores(self) -> None:
        statistics = calculate_train_statistics(self.train)
        self.assertAlmostEqual(statistics.global_means["content"], 3.0)
        self.assertAlmostEqual(statistics.global_means["organization"], 3.1)
        self.assertAlmostEqual(statistics.global_means["expression"], 3.2)
        self.assertAlmostEqual(statistics.prompt_means["Q1"]["content"], 3.0)
        self.assertAlmostEqual(statistics.prompt_means["Q2"]["content"], 3.0)
        self.assertEqual(statistics.prompt_counts, {"Q1": 2, "Q2": 1})

    def test_predictions_do_not_read_validation_scores(self) -> None:
        statistics = calculate_train_statistics(self.train)
        without_scores = [{"id": "v1", "prompt_num": "Q1"}]
        with_changed_scores = [
            {
                "id": "v1",
                "prompt_num": "Q1",
                "score": {
                    "content": -999,
                    "organization": 999,
                    "expression": 123,
                },
            }
        ]
        self.assertEqual(
            build_global_predictions(without_scores, statistics.global_means),
            build_global_predictions(with_changed_scores, statistics.global_means),
        )
        self.assertEqual(
            build_prompt_predictions(without_scores, statistics),
            build_prompt_predictions(with_changed_scores, statistics),
        )

    def test_prompt_predictions_fall_back_to_global_mean(self) -> None:
        statistics = calculate_train_statistics(self.train)
        validation = [
            {"id": "v1", "prompt_num": "Q1"},
            {"id": "v2", "prompt_num": "Q999"},
        ]
        predictions, fallback_count = build_prompt_predictions(validation, statistics)
        self.assertEqual(fallback_count, 1)
        self.assertEqual(
            predictions[1]["content"]["score"],
            statistics.global_means["content"],
        )

    def test_baseline_reports_undefined_constant_spearman(self) -> None:
        validation = [
            {
                "id": f"v{index}",
                "prompt_num": "Q1",
                "score": {
                    "content": score,
                    "organization": score,
                    "expression": score,
                    "average": 999,
                },
            }
            for index, score in enumerate((1.0, 3.0, 5.0), start=1)
        ]
        results = run_baselines(self.train, validation)
        global_metrics = results["baselines"]["global_mean"]["metrics"]
        self.assertIsNone(global_metrics["mean"]["spearman"])
        for dimension in ("content", "organization", "expression"):
            self.assertIsNone(global_metrics["dimensions"][dimension]["spearman"])


if __name__ == "__main__":
    unittest.main()
