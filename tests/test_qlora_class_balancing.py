from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.qlora.class_balancing import calculate_class_balance
from src.qlora.data import (
    TrainingDataError,
    TrainingSample,
    build_training_sample,
    iter_training_samples,
)


SETTINGS = {
    "enabled": True,
    "method": "bounded_inverse_sqrt_v1",
    "clip_min": 0.75,
    "clip_max": 2.0,
}


def sample(
    index: int,
    content: int,
    organization: int,
    expression: int,
) -> TrainingSample:
    return TrainingSample(
        sample_id=f"sample-{index}",
        document_id=None,
        prompt="prompt",
        essay="essay",
        gold_scores={
            "content": content,
            "organization": organization,
            "expression": expression,
        },
    )


class ClassBalancingTests(unittest.TestCase):
    def test_round_half_up_labels_are_reused_for_counts(self) -> None:
        records = [
            {
                "id": "one",
                "prompt": "p",
                "essay": "e",
                "score": {
                    "content": 1.5,
                    "organization": 2.5,
                    "expression": 4.5,
                },
            },
            {
                "id": "two",
                "prompt": "p",
                "essay": "e",
                "score": {
                    "content": 1.49,
                    "organization": 2.49,
                    "expression": 4.49,
                },
            },
        ]
        result = calculate_class_balance(
            [build_training_sample(record) for record in records], SETTINGS
        )
        self.assertEqual(result.counts["content"], {1: 1, 2: 1, 3: 0, 4: 0, 5: 0})
        self.assertEqual(
            result.counts["organization"], {1: 0, 2: 1, 3: 1, 4: 0, 5: 0}
        )
        self.assertEqual(
            result.counts["expression"], {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
        )

    def test_inverse_sqrt_clipping_and_bounded_normalization(self) -> None:
        samples = [
            sample(index, content, organization, expression)
            for index, (content, organization, expression) in enumerate(
                [(1, 1, 1)]
                + [(2, 2, 2)] * 9
                + [(3, 3, 3)] * 20
                + [(4, 4, 4)] * 30
                + [(5, 5, 5)] * 40
            )
        ]
        result = calculate_class_balance(samples, SETTINGS)
        raw = result.raw_weights["content"]
        self.assertAlmostEqual(raw[1], math.sqrt(20 / 1))
        self.assertAlmostEqual(raw[2], math.sqrt(20 / 9))
        self.assertEqual(result.clipped_weights["content"][1], 2.0)
        self.assertEqual(result.clipped_weights["content"][5], 0.75)
        final = result.final_weights["content"]
        self.assertTrue(all(0.75 <= value <= 2.0 for value in final.values()))
        weighted_mean = sum(
            result.counts["content"][label] * final[label]
            for label in range(1, 6)
        ) / len(samples)
        self.assertAlmostEqual(weighted_mean, 1.0)
        self.assertEqual(final[1], 2.0)

    def test_weights_are_calculated_per_dimension(self) -> None:
        samples = [
            sample(index, 1 if index == 0 else 3, 5 if index < 5 else 3, 4)
            for index in range(20)
        ]
        result = calculate_class_balance(samples, SETTINGS)
        self.assertNotEqual(
            result.final_weights["content"],
            result.final_weights["organization"],
        )
        self.assertNotEqual(
            result.counts["content"], result.counts["expression"]
        )

    def test_disabled_balancing_preserves_unit_multipliers(self) -> None:
        disabled = {**SETTINGS, "enabled": False}
        result = calculate_class_balance([sample(0, 1, 2, 5)], disabled)
        for weights in result.final_weights.values():
            self.assertEqual(weights, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})

    def test_validation_file_cannot_supply_class_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "competition_validation.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "validation",
                        "prompt": "p",
                        "essay": "e",
                        "score": {
                            "content": 1,
                            "organization": 1,
                            "expression": 1,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TrainingDataError, "validation JSONL"):
                list(iter_training_samples(path))


if __name__ == "__main__":
    unittest.main()
