from __future__ import annotations

import unittest

import numpy as np

from src.qlora.loss import (
    WeightedLossTrainerMixin,
    shift_causal_labels_and_weights,
    weighted_token_mean,
)


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


if __name__ == "__main__":
    unittest.main()
