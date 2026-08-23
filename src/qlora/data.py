"""Leakage-safe projection of JSONL records into QLoRA training samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    from ..evaluate import DIMENSIONS, round_half_up_score
except ImportError:  # python src/train_qlora.py
    from evaluate import DIMENSIONS, round_half_up_score


class TrainingDataError(ValueError):
    """Raised when a record cannot be used as supervised training data."""


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    document_id: str | None
    prompt: str
    essay: str
    gold_scores: dict[str, int]


def reject_validation_path(path: Path) -> None:
    if "validation" in path.name.lower():
        raise TrainingDataError(
            f"validation JSONL is forbidden for supervised training: {path}"
        )


def build_training_sample(
    record: Mapping[str, Any], location: str = "record"
) -> TrainingSample:
    if not isinstance(record, Mapping):
        raise TrainingDataError(f"{location}: record must be a JSON object")

    identifier = record.get("id")
    document_id = record.get("document_id")
    if not isinstance(identifier, str) or not identifier:
        raise TrainingDataError(f"{location}.id: non-empty string required")
    if document_id is not None and (
        not isinstance(document_id, str) or not document_id
    ):
        raise TrainingDataError(f"{location}.document_id: non-empty string required")

    prompt = record.get("prompt")
    essay = record.get("essay")
    if not isinstance(prompt, str) or not prompt:
        raise TrainingDataError(f"{location}.prompt: non-empty string required")
    if not isinstance(essay, str) or not essay:
        raise TrainingDataError(f"{location}.essay: non-empty string required")

    score = record.get("score")
    if not isinstance(score, Mapping):
        raise TrainingDataError(f"{location}.score: JSON object required")
    gold_scores = {
        dimension: round_half_up_score(
            score.get(dimension), f"{location}.score.{dimension}"
        )
        for dimension in DIMENSIONS
    }
    return TrainingSample(
        sample_id=identifier,
        document_id=document_id,
        prompt=prompt,
        essay=essay,
        gold_scores=gold_scores,
    )


def iter_training_samples(path: Path) -> Iterator[TrainingSample]:
    reject_validation_path(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            yield build_training_sample(payload, f"{path}:{line_number}")
