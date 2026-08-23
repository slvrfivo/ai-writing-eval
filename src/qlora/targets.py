"""Construct complete assistant JSON with explicit semantic character spans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

try:
    from ..evaluate import DIMENSIONS
except ImportError:  # python src/train_qlora.py
    from evaluate import DIMENSIONS

from .rationale_templates import TARGET_CONSTRUCTION_VERSION, rationale_for


@dataclass(frozen=True)
class CharacterSpan:
    start: int
    end: int
    role: str
    dimension: str | None = None


@dataclass(frozen=True)
class AssistantTarget:
    text: str
    spans: tuple[CharacterSpan, ...]
    version: str


def validate_character_spans(target: AssistantTarget) -> None:
    """Require a gap-free, non-overlapping partition of the target text."""
    position = 0
    score_dimensions: list[str | None] = []
    for index, span in enumerate(target.spans):
        if span.start != position or span.end <= span.start:
            raise ValueError(
                f"target span {index} overlaps, has a gap, or is empty: {span}"
            )
        if span.role not in {"structure", "score", "rationale"}:
            raise ValueError(f"target span {index} has an invalid role: {span.role}")
        if span.role == "score":
            score_dimensions.append(span.dimension)
        position = span.end
    if position != len(target.text):
        raise ValueError("target spans do not cover the complete assistant target")
    if score_dimensions != list(DIMENSIONS):
        raise ValueError(
            f"target must contain exactly three ordered score spans: {score_dimensions}"
        )


class _TargetBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[CharacterSpan] = []
        self.position = 0

    def append(
        self, text: str, role: str, dimension: str | None = None
    ) -> None:
        if not text:
            return
        start = self.position
        self.parts.append(text)
        self.position += len(text)
        self.spans.append(CharacterSpan(start, self.position, role, dimension))

    def finish(self) -> AssistantTarget:
        return AssistantTarget(
            text="".join(self.parts),
            spans=tuple(self.spans),
            version=TARGET_CONSTRUCTION_VERSION,
        )


def build_assistant_target(scores: Mapping[str, int]) -> AssistantTarget:
    if set(scores) != set(DIMENSIONS):
        raise ValueError(f"scores must contain exactly {DIMENSIONS}")

    builder = _TargetBuilder()
    builder.append("{", "structure")
    for index, dimension in enumerate(DIMENSIONS):
        if index:
            builder.append(",", "structure")
        score = scores[dimension]
        rationale = rationale_for(dimension, score)
        escaped_rationale = json.dumps(rationale, ensure_ascii=False)[1:-1]

        builder.append(json.dumps(dimension) + ":{\"score\":", "structure")
        builder.append(str(score), "score", dimension)
        builder.append(",\"rationale\":\"", "structure")
        builder.append(escaped_rationale, "rationale", dimension)
        builder.append("\"}", "structure")
    builder.append("}", "structure")
    target = builder.finish()
    validate_character_spans(target)

    parsed = json.loads(target.text)
    if any(parsed[name]["score"] != scores[name] for name in DIMENSIONS):
        raise AssertionError("assistant target score construction failed")
    return target
