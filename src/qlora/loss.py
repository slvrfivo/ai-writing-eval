"""Causally shifted token-weighted cross entropy for score-focused SFT."""

from __future__ import annotations

from typing import Any, Sequence


class WeightedLossError(ValueError):
    """Raised when a batch has no supervised token weight."""


def shift_causal_labels_and_weights(labels: Any, loss_weights: Any) -> tuple[Any, Any]:
    """Align each next-token label and its semantic weight with prior logits."""
    return labels[..., 1:], loss_weights[..., 1:]


def weighted_token_mean(
    token_losses: Any,
    labels: Any,
    loss_weights: Any,
    *,
    denominator: float | None = None,
) -> Any:
    """Return sum(loss * weight) / denominator, excluding ignored labels.

    With ``denominator=None`` the denominator is the batch's own supervised
    weight sum. Because the recorded runs use one sample per micro-batch, that
    normalizes every sample separately ("per_sample"). A fixed per-example
    ``denominator`` ("dataset_mean") is multiplied by the number of examples in
    the batch, so per-sample weight differences such as class-balanced score
    weights are not divided back out.
    """
    valid = (labels != -100) & (loss_weights > 0)
    effective_weights = loss_weights * valid
    if denominator is not None:
        if isinstance(denominator, bool) or not denominator > 0:
            raise WeightedLossError("fixed loss denominator must be positive")
        if float(effective_weights.sum()) <= 0:
            raise WeightedLossError("batch has no positive supervised token weight")
        batch_size = labels.shape[0] if len(labels.shape) > 1 else 1
        return (token_losses * effective_weights).sum() / (
            float(denominator) * batch_size
        )
    denominator = effective_weights.sum()
    scalar_denominator = (
        float(denominator.detach().item())
        if hasattr(denominator, "detach")
        else float(denominator)
    )
    if scalar_denominator <= 0:
        raise WeightedLossError("batch has no positive supervised token weight")
    return (token_losses * effective_weights).sum() / denominator


def supervised_weight_sum(labels: Sequence[int], loss_weights: Sequence[float]) -> float:
    """Total weight the causal loss sees for one example (after the shift)."""
    return float(
        sum(
            weight
            for label, weight in zip(labels[1:], loss_weights[1:])
            if label != -100 and weight > 0
        )
    )


def mean_supervised_weight(
    examples: Sequence[tuple[Sequence[int], Sequence[float]]]
) -> float:
    """Average per-example supervised weight, used as the dataset_mean denominator."""
    totals = [supervised_weight_sum(labels, weights) for labels, weights in examples]
    if not totals or not all(total > 0 for total in totals):
        raise WeightedLossError("every example needs positive supervised token weight")
    return sum(totals) / len(totals)


def weighted_causal_cross_entropy(
    logits: Any, labels: Any, loss_weights: Any, *, denominator: float | None = None
) -> Any:
    """Compute unreduced causal CE followed by the configured weighted mean."""
    import torch.nn.functional as functional

    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels, shifted_weights = shift_causal_labels_and_weights(
        labels, loss_weights
    )
    shifted_labels = shifted_labels.to(shifted_logits.device)
    shifted_weights = shifted_weights.to(shifted_logits.device)
    valid = (shifted_labels != -100) & (shifted_weights > 0)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    vocabulary_size = shifted_logits.shape[-1]
    token_losses = functional.cross_entropy(
        shifted_logits.view(-1, vocabulary_size),
        safe_labels.reshape(-1),
        reduction="none",
    ).view_as(safe_labels)
    return weighted_token_mean(
        token_losses, shifted_labels, shifted_weights, denominator=denominator
    )


class WeightedLossTrainerMixin:
    """Transformers Trainer mixin implementing weighted token loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Transformers 4.55 explicitly requires this when a custom loss does not
        # normalize with num_items_in_batch. Trainer then scales gradient
        # accumulation in training_step instead of passing the value to the model.
        self.model_accepts_loss_kwargs = False
        # None = per-sample normalization (the recorded v1/v1b runs).
        self.loss_denominator: float | None = None

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> Any:
        del num_items_in_batch
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = weighted_causal_cross_entropy(
            logits, labels, loss_weights, denominator=self.loss_denominator
        )
        return (loss, outputs) if return_outputs else loss


def weighted_trainer_class() -> type:
    """Create the Trainer subclass lazily so unit tests need no ML packages."""
    from transformers import Trainer

    class WeightedTokenTrainer(WeightedLossTrainerMixin, Trainer):
        pass

    return WeightedTokenTrainer
