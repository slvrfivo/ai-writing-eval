"""Causally shifted token-weighted cross entropy for score-focused SFT."""

from __future__ import annotations

from typing import Any


class WeightedLossError(ValueError):
    """Raised when a batch has no supervised token weight."""


def shift_causal_labels_and_weights(labels: Any, loss_weights: Any) -> tuple[Any, Any]:
    """Align each next-token label and its semantic weight with prior logits."""
    return labels[..., 1:], loss_weights[..., 1:]


def weighted_token_mean(token_losses: Any, labels: Any, loss_weights: Any) -> Any:
    """Return sum(loss * weight) / sum(weight), excluding ignored labels."""
    valid = (labels != -100) & (loss_weights > 0)
    effective_weights = loss_weights * valid
    denominator = effective_weights.sum()
    scalar_denominator = (
        float(denominator.detach().item())
        if hasattr(denominator, "detach")
        else float(denominator)
    )
    if scalar_denominator <= 0:
        raise WeightedLossError("batch has no positive supervised token weight")
    return (token_losses * effective_weights).sum() / denominator


def weighted_causal_cross_entropy(
    logits: Any, labels: Any, loss_weights: Any
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
    return weighted_token_mean(token_losses, shifted_labels, shifted_weights)


class WeightedLossTrainerMixin:
    """Transformers Trainer mixin implementing weighted token loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Transformers 4.55 explicitly requires this when a custom loss does not
        # normalize with num_items_in_batch. Trainer then scales gradient
        # accumulation in training_step instead of passing the value to the model.
        self.model_accepts_loss_kwargs = False

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
        loss = weighted_causal_cross_entropy(logits, labels, loss_weights)
        return (loss, outputs) if return_outputs else loss


def weighted_trainer_class() -> type:
    """Create the Trainer subclass lazily so unit tests need no ML packages."""
    from transformers import Trainer

    class WeightedTokenTrainer(WeightedLossTrainerMixin, Trainer):
        pass

    return WeightedTokenTrainer
