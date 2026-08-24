"""Train-only rounded-score class counts and bounded inverse-sqrt weights."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from ..evaluate import DIMENSIONS
except ImportError:  # python src/train_qlora.py
    from evaluate import DIMENSIONS

from .data import TrainingSample


SCORE_CLASSES = (1, 2, 3, 4, 5)
CLASS_BALANCE_METHOD = "bounded_inverse_sqrt_v1"


@dataclass(frozen=True)
class ClassBalanceResult:
    enabled: bool
    counts: dict[str, dict[int, int]]
    raw_weights: dict[str, dict[int, float | None]]
    clipped_weights: dict[str, dict[int, float]]
    normalization_scales: dict[str, float]
    final_weights: dict[str, dict[int, float]]
    clip_min: float
    clip_max: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": CLASS_BALANCE_METHOD,
            "source": "train rounded labels only",
            "score_classes": list(SCORE_CLASSES),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "normalization": "sample-weighted mean 1.0 with bounds retained",
            "counts": self.counts,
            "raw_weights": self.raw_weights,
            "clipped_weights": self.clipped_weights,
            "normalization_scales": self.normalization_scales,
            "final_weights": self.final_weights,
        }


def _validate_settings(settings: Mapping[str, Any]) -> tuple[bool, float, float]:
    expected_keys = {"enabled", "method", "clip_min", "clip_max"}
    if set(settings) != expected_keys:
        raise ValueError(f"class balancing settings must contain {expected_keys}")
    enabled = settings["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("class balancing enabled must be boolean")
    if settings["method"] != CLASS_BALANCE_METHOD:
        raise ValueError(f"class balancing method must be {CLASS_BALANCE_METHOD}")
    clip_min = settings["clip_min"]
    clip_max = settings["clip_max"]
    if (
        isinstance(clip_min, bool)
        or isinstance(clip_max, bool)
        or not isinstance(clip_min, (int, float))
        or not isinstance(clip_max, (int, float))
        or not 0 < clip_min <= 1.0 <= clip_max
    ):
        raise ValueError("class balancing clip bounds must contain 1.0")
    return enabled, float(clip_min), float(clip_max)


def _bounded_normalize(
    clipped: Mapping[int, float],
    counts: Mapping[int, int],
    *,
    clip_min: float,
    clip_max: float,
) -> tuple[float, dict[int, float]]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("class balancing requires at least one training sample")

    def weighted_mean(scale: float) -> float:
        return sum(
            counts[label]
            * min(clip_max, max(clip_min, clipped[label] * scale))
            for label in SCORE_CLASSES
        ) / total

    lower = 0.0
    upper = 1.0
    while weighted_mean(upper) < 1.0:
        upper *= 2.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if weighted_mean(middle) < 1.0:
            lower = middle
        else:
            upper = middle
    scale = (lower + upper) / 2.0
    normalized = {
        label: min(clip_max, max(clip_min, clipped[label] * scale))
        for label in SCORE_CLASSES
    }
    return scale, normalized


def calculate_class_balance(
    samples: Sequence[TrainingSample],
    settings: Mapping[str, Any],
) -> ClassBalanceResult:
    """Calculate weights exclusively from already rounded training labels."""
    enabled, clip_min, clip_max = _validate_settings(settings)
    if not samples:
        raise ValueError("class balancing requires non-empty training samples")

    counts: dict[str, dict[int, int]] = {}
    raw_weights: dict[str, dict[int, float | None]] = {}
    clipped_weights: dict[str, dict[int, float]] = {}
    normalization_scales: dict[str, float] = {}
    final_weights: dict[str, dict[int, float]] = {}

    for dimension in DIMENSIONS:
        counter = Counter(sample.gold_scores[dimension] for sample in samples)
        dimension_counts = {label: counter[label] for label in SCORE_CLASSES}
        if sum(dimension_counts.values()) != len(samples):
            raise ValueError(f"invalid rounded class labels for {dimension}")
        counts[dimension] = dimension_counts

        if not enabled:
            raw_weights[dimension] = {label: 1.0 for label in SCORE_CLASSES}
            clipped_weights[dimension] = {
                label: 1.0 for label in SCORE_CLASSES
            }
            normalization_scales[dimension] = 1.0
            final_weights[dimension] = {label: 1.0 for label in SCORE_CLASSES}
            continue

        mean_count = len(samples) / len(SCORE_CLASSES)
        raw = {
            label: (
                math.sqrt(mean_count / count) if count > 0 else None
            )
            for label, count in dimension_counts.items()
        }
        clipped = {
            label: (
                clip_max
                if raw[label] is None
                else min(clip_max, max(clip_min, raw[label]))
            )
            for label in SCORE_CLASSES
        }
        scale, normalized = _bounded_normalize(
            clipped,
            dimension_counts,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        raw_weights[dimension] = raw
        clipped_weights[dimension] = clipped
        normalization_scales[dimension] = scale
        final_weights[dimension] = normalized

    return ClassBalanceResult(
        enabled=enabled,
        counts=counts,
        raw_weights=raw_weights,
        clipped_weights=clipped_weights,
        normalization_scales=normalization_scales,
        final_weights=final_weights,
        clip_min=clip_min,
        clip_max=clip_max,
    )
