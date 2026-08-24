"""Offset-mapped token roles, length inspection, and no-truncation datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

try:
    from ..llm.prompting import build_messages
except ImportError:  # python src/train_qlora.py
    from llm.prompting import build_messages

from .data import TrainingSample
from .targets import AssistantTarget, CharacterSpan, build_assistant_target


TOKEN_ROLES = ("prompt", "structure", "score", "rationale")
MIXED_BOUNDARY_POLICY = "rationale_ending_to_rationale_v1"


class TokenizationError(ValueError):
    """Raised when chat-template token boundaries cannot be mapped exactly."""


class SequenceLengthError(TokenizationError):
    def __init__(self, sample_id: str, actual: int, maximum: int) -> None:
        super().__init__(
            f"sample {sample_id!r} has {actual} tokens, exceeding max_seq_length={maximum}; "
            "training refuses to truncate"
        )
        self.sample_id = sample_id
        self.actual = actual
        self.maximum = maximum


@dataclass(frozen=True)
class MixedBoundaryToken:
    token_index: int
    overlapping_roles: tuple[str, ...]
    assigned_role: str
    dimension: str


@dataclass(frozen=True)
class TokenizedExample:
    sample_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    loss_weights: list[float]
    token_roles: list[str]
    token_dimensions: list[str | None]
    token_offsets: list[tuple[int, int]]
    prompt_token_count: int
    target_start: int
    target: AssistantTarget
    mixed_boundary_tokens: tuple[MixedBoundaryToken, ...]

    @property
    def token_length(self) -> int:
        return len(self.input_ids)

    def model_inputs(self) -> dict[str, list[int] | list[float]]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
            "loss_weights": list(self.loss_weights),
        }


@dataclass(frozen=True)
class TokenLengthStats:
    count: int
    minimum: int
    p50: int
    p90: int
    p95: int
    p99: int
    maximum: int
    over_2048: int
    over_3072: int
    over_4096: int

    def as_dict(self) -> dict[str, int]:
        return {
            "count": self.count,
            "min": self.minimum,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.maximum,
            ">2048": self.over_2048,
            ">3072": self.over_3072,
            ">4096": self.over_4096,
        }


def _flat_list(value: Any, name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TokenizationError(f"tokenizer {name} must be a list")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise TokenizationError("only single-sample tokenization is supported")
        value = value[0]
    return list(value)


def _overlaps(offset: tuple[int, int], span: CharacterSpan, shift: int) -> bool:
    start, end = offset
    span_start = shift + span.start
    span_end = shift + span.end
    return start < span_end and end > span_start


def _assistant_assignment(
    offset: tuple[int, int],
    target: AssistantTarget,
    target_start: int,
    token_index: int,
) -> tuple[str, str | None, tuple[str, ...]]:
    """Assign one role, allowing only rationale-end/structure mixed tokens.

    The tokenizer may merge the final rationale character or punctuation with
    the following JSON closing quote. That mixed token deterministically uses
    the rationale role and weight. Every other cross-role overlap is rejected.
    """
    if offset[0] == offset[1]:
        return "structure", None, ()

    overlapping = [
        span for span in target.spans if _overlaps(offset, span, target_start)
    ]
    roles = {span.role for span in overlapping}
    if len(roles) > 1:
        rationale_spans = [span for span in overlapping if span.role == "rationale"]
        is_expected_rationale_end = False
        rationale_span: CharacterSpan | None = None
        if roles == {"rationale", "structure"} and len(rationale_spans) == 1:
            rationale_span = rationale_spans[0]
            span_index = target.spans.index(rationale_span)
            following_span = (
                target.spans[span_index + 1]
                if span_index + 1 < len(target.spans)
                else None
            )
            boundary = target_start + rationale_span.end
            is_expected_rationale_end = (
                following_span is not None
                and following_span.role == "structure"
                and following_span.start == rationale_span.end
                and offset[0] < boundary < offset[1]
                and all(
                    span.role == "rationale" or span.start >= rationale_span.end
                    for span in overlapping
                )
            )
        if not is_expected_rationale_end or rationale_span is None:
            raise TokenizationError(
                "unexpected cross-role token overlap at token "
                f"{token_index}: offset={offset}, roles={sorted(roles)}"
            )
        return (
            "rationale",
            rationale_span.dimension,
            ("rationale", "structure"),
        )

    if roles == {"score"}:
        return "score", overlapping[0].dimension, ()
    if roles == {"rationale"}:
        return "rationale", overlapping[0].dimension, ()
    return "structure", None, ()


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def encode_training_example(
    sample: TrainingSample,
    *,
    tokenizer: Any,
    prompt_version: str,
    loss_weights: Mapping[str, float],
    score_class_weights: Mapping[str, Mapping[int, float]] | None = None,
    max_seq_length: int | None = None,
) -> TokenizedExample:
    """Apply the official chat template and map every token to an explicit role."""
    if set(loss_weights) != set(TOKEN_ROLES):
        raise TokenizationError(f"loss_weights must contain exactly {TOKEN_ROLES}")
    if score_class_weights is not None:
        if set(score_class_weights) != set(sample.gold_scores):
            raise TokenizationError(
                "score_class_weights dimensions must match gold score dimensions"
            )
        for dimension, class_weights in score_class_weights.items():
            if set(class_weights) != {1, 2, 3, 4, 5}:
                raise TokenizationError(
                    f"score_class_weights[{dimension}] must contain classes 1..5"
                )

    messages = build_messages(sample.prompt, sample.essay, version=prompt_version)
    target = build_assistant_target(sample.gold_scores)
    prompt_rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_rendered = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": target.text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_rendered.startswith(prompt_rendered):
        raise TokenizationError(
            "chat template with assistant target does not preserve the generation prefix"
        )
    target_start = len(prompt_rendered)
    if full_rendered[target_start : target_start + len(target.text)] != target.text:
        raise TokenizationError("assistant target is not at the explicit chat boundary")

    full_encoding = tokenizer(
        full_rendered,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    prompt_encoding = tokenizer(
        prompt_rendered,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=False,
    )
    input_ids = [int(value) for value in _flat_list(full_encoding["input_ids"], "input_ids")]
    prompt_ids = [int(value) for value in _flat_list(prompt_encoding["input_ids"], "input_ids")]
    offsets_raw = _flat_list(full_encoding["offset_mapping"], "offset_mapping")
    offsets = [(int(start), int(end)) for start, end in offsets_raw]
    if len(offsets) != len(input_ids):
        raise TokenizationError("offset mapping length does not match input_ids")
    prompt_token_count = _common_prefix_length(input_ids, prompt_ids)
    if prompt_token_count < max(0, len(prompt_ids) - 1):
        raise TokenizationError(
            "chat tokenization differs by more than one token at the explicit "
            "prompt/assistant boundary"
        )

    attention_value = full_encoding.get("attention_mask", [1] * len(input_ids))
    attention_mask = [
        int(value) for value in _flat_list(attention_value, "attention_mask")
    ]
    if len(attention_mask) != len(input_ids):
        raise TokenizationError("attention_mask length does not match input_ids")

    assistant_assignments = [
        _assistant_assignment(offset, target, target_start, token_index)
        for token_index, offset in enumerate(
            offsets[prompt_token_count:], start=prompt_token_count
        )
    ]
    roles = ["prompt"] * prompt_token_count + [
        role for role, _, _ in assistant_assignments
    ]
    dimensions = [None] * prompt_token_count + [
        dimension for _, dimension, _ in assistant_assignments
    ]
    if len(roles) != len(input_ids):
        raise TokenizationError("token role count does not match input_ids")
    weights = []
    for role, dimension in zip(roles, dimensions):
        weight = float(loss_weights[role])
        if role == "score" and score_class_weights is not None:
            if dimension is None:
                raise TokenizationError("score token is missing its dimension")
            gold_class = sample.gold_scores[dimension]
            weight *= float(score_class_weights[dimension][gold_class])
        weights.append(weight)

    example = TokenizedExample(
        sample_id=sample.sample_id,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=list(input_ids),
        loss_weights=weights,
        token_roles=roles,
        token_dimensions=dimensions,
        token_offsets=offsets,
        prompt_token_count=prompt_token_count,
        target_start=target_start,
        target=target,
        mixed_boundary_tokens=tuple(
            MixedBoundaryToken(
                token_index=token_index,
                overlapping_roles=mixed_roles,
                assigned_role=role,
                dimension=dimension,
            )
            for token_index, (role, dimension, mixed_roles) in enumerate(
                assistant_assignments, start=prompt_token_count
            )
            if mixed_roles and dimension is not None
        ),
    )
    if max_seq_length is not None and example.token_length > max_seq_length:
        raise SequenceLengthError(sample.sample_id, example.token_length, max_seq_length)
    return example


def percentile_nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        raise ValueError("cannot calculate a percentile for an empty sequence")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(int(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def calculate_token_length_stats(lengths: Sequence[int]) -> TokenLengthStats:
    if not lengths:
        raise ValueError("token length statistics require at least one sample")
    normalized = [int(value) for value in lengths]
    if any(value <= 0 for value in normalized):
        raise ValueError("token lengths must be positive")
    return TokenLengthStats(
        count=len(normalized),
        minimum=min(normalized),
        p50=percentile_nearest_rank(normalized, 0.50),
        p90=percentile_nearest_rank(normalized, 0.90),
        p95=percentile_nearest_rank(normalized, 0.95),
        p99=percentile_nearest_rank(normalized, 0.99),
        maximum=max(normalized),
        over_2048=sum(value > 2048 for value in normalized),
        over_3072=sum(value > 3072 for value in normalized),
        over_4096=sum(value > 4096 for value in normalized),
    )


class TokenizedTrainingDataset:
    def __init__(self, examples: Iterable[TokenizedExample]) -> None:
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("training dataset is empty")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int] | list[float]]:
        return self.examples[index].model_inputs()


def pad_batch_python(
    features: Sequence[Mapping[str, Sequence[int] | Sequence[float]]],
    *,
    pad_token_id: int,
) -> dict[str, list[list[int]] | list[list[float]]]:
    if not features:
        raise ValueError("cannot collate an empty batch")
    maximum = max(len(feature["input_ids"]) for feature in features)
    batch: dict[str, list[list[Any]]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "loss_weights": [],
    }
    padding_values = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
        "labels": -100,
        "loss_weights": 0.0,
    }
    for feature in features:
        length = len(feature["input_ids"])
        for key, padding_value in padding_values.items():
            values = list(feature[key])
            if len(values) != length:
                raise ValueError(f"feature {key} length does not match input_ids")
            batch[key].append(values + [padding_value] * (maximum - length))
    return batch


class WeightedDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        padded = pad_batch_python(features, pad_token_id=self.pad_token_id)
        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                padded["attention_mask"], dtype=torch.long
            ),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(
                padded["loss_weights"], dtype=torch.float32
            ),
        }
