from __future__ import annotations

import unittest

import numpy as np

import json
import tempfile
from pathlib import Path

from src.qlora.config import QLoRAConfig, QLoRAConfigError
from src.qlora.loss import (
    WeightedLossError,
    WeightedLossTrainerMixin,
    mean_supervised_weight,
    shift_causal_labels_and_weights,
    supervised_weight_sum,
    weighted_token_mean,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QLoRALossTests(unittest.TestCase):
    def test_trainer_disables_num_items_loss_kwargs(self) -> None:
        class FakeTrainerBase:
            def __init__(self) -> None:
                self.model_accepts_loss_kwargs = True

        class FakeWeightedTrainer(WeightedLossTrainerMixin, FakeTrainerBase):
            pass

        trainer = FakeWeightedTrainer()
        self.assertFalse(trainer.model_accepts_loss_kwargs)

    def test_causal_shift_uses_next_token_roles(self) -> None:
        labels = np.asarray([[10, 11, 12, 13]])
        weights = np.asarray([[0.0, 1.0, 10.0, 0.1]])
        shifted_labels, shifted_weights = shift_causal_labels_and_weights(
            labels, weights
        )
        np.testing.assert_array_equal(shifted_labels, [[11, 12, 13]])
        np.testing.assert_array_equal(shifted_weights, [[1.0, 10.0, 0.1]])

    def test_weighted_loss_calculation_and_ignored_labels(self) -> None:
        token_losses = np.asarray([[2.0, 100.0, 6.0]])
        labels = np.asarray([[11, -100, 13]])
        weights = np.asarray([[1.0, 10.0, 3.0]])
        loss = weighted_token_mean(token_losses, labels, weights)
        self.assertAlmostEqual(float(loss), 5.0)

    def test_trainer_defaults_to_per_sample_normalization(self) -> None:
        class FakeTrainerBase:
            def __init__(self) -> None:
                self.model_accepts_loss_kwargs = True

        class FakeWeightedTrainer(WeightedLossTrainerMixin, FakeTrainerBase):
            pass

        self.assertIsNone(FakeWeightedTrainer().loss_denominator)

    def test_fixed_denominator(self) -> None:
        token_losses = np.asarray([[2.0, 100.0, 6.0]])
        labels = np.asarray([[11, -100, 13]])
        weights = np.asarray([[1.0, 10.0, 3.0]])
        loss = weighted_token_mean(token_losses, labels, weights, denominator=10.0)
        self.assertAlmostEqual(float(loss), 2.0)
        with self.assertRaises(WeightedLossError):
            weighted_token_mean(token_losses, labels, weights, denominator=0.0)

    def test_fixed_denominator_scales_with_batch_size(self) -> None:
        token_losses = np.asarray([[2.0, 6.0], [4.0, 0.0]])
        labels = np.asarray([[11, 13], [12, -100]])
        weights = np.asarray([[1.0, 3.0], [2.0, 5.0]])
        loss = weighted_token_mean(token_losses, labels, weights, denominator=5.0)
        self.assertAlmostEqual(float(loss), (2.0 + 18.0 + 8.0) / 10.0)

    def test_supervised_weight_sum_uses_shifted_labels(self) -> None:
        labels = [-100, -100, 21, 22, -100]
        weights = [0.0, 0.0, 10.0, 0.25, 0.0]
        self.assertAlmostEqual(supervised_weight_sum(labels, weights), 10.25)
        self.assertAlmostEqual(
            mean_supervised_weight([(labels, weights), ([1, 2], [0.0, 4.0])]),
            (10.25 + 4.0) / 2,
        )
        with self.assertRaises(WeightedLossError):
            mean_supervised_weight([([1, -100], [0.0, 5.0])])

    def test_per_sample_normalization_dampens_sample_level_weights(self) -> None:
        # One score token (weight 10 * class weight) plus fixed non-score weight 15.
        # Doubling the class weight doubles the score token's share only when the
        # denominator is fixed; per-sample normalization divides most of it out.
        def score_coefficient(class_weight: float, denominator: float | None) -> float:
            token_losses = np.asarray([[1.0, 0.0]])
            labels = np.asarray([[1, 2]])
            weights = np.asarray([[10.0 * class_weight, 15.0]])
            return float(
                weighted_token_mean(token_losses, labels, weights, denominator=denominator)
            )

        per_sample_ratio = score_coefficient(2.0, None) / score_coefficient(1.0, None)
        fixed_ratio = score_coefficient(2.0, 25.0) / score_coefficient(1.0, 25.0)
        self.assertAlmostEqual(fixed_ratio, 2.0)
        self.assertAlmostEqual(per_sample_ratio, (20 / 35) / (10 / 25))
        self.assertLess(per_sample_ratio, 1.5)

    def test_config_loss_normalization(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "qwen3_4b_qlora_v1.json"
        config = QLoRAConfig.from_json(config_path)
        self.assertEqual(config.loss_normalization, "per_sample")
        updated = config.with_loss_normalization("dataset_mean")
        self.assertEqual(updated.loss_normalization, "dataset_mean")
        self.assertEqual(
            updated.with_max_seq_length(None).loss_normalization, "dataset_mean"
        )
        self.assertEqual(
            updated.with_class_balancing(True).loss_normalization, "dataset_mean"
        )
        with self.assertRaises(QLoRAConfigError):
            config.with_loss_normalization("global")

        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["loss_normalization"] = "dataset_mean"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                QLoRAConfig.from_json(path).loss_normalization, "dataset_mean"
            )


if __name__ == "__main__":
    unittest.main()
