"""Strictly parse an official writing-score JSON object from model text."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


DIMENSIONS = ("content", "organization", "expression")


@dataclass(frozen=True)
class ParseFailure:
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ParseResult:
    value: dict[str, dict[str, int | str]] | None = None
    failure: ParseFailure | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.failure is None):
            raise ValueError("exactly one of value and failure must be set")

    @property
    def ok(self) -> bool:
        return self.failure is None


def _failure(code: str, message: str, path: str | None = None) -> ParseResult:
    return ParseResult(failure=ParseFailure(code=code, message=message, path=path))


def parse_model_output(output: str) -> ParseResult:
    """Return either a validated prediction or a structured parse failure.

    The parser deliberately performs no score coercion, rounding, clamping, or
    recovery from Markdown fences or surrounding prose.
    """
    if not isinstance(output, str):
        return _failure("invalid_output_type", "model output must be a string")
    if not output.strip():
        return _failure("empty_output", "model output is empty")
    if "```" in output:
        return _failure(
            "markdown_code_fence",
            "Markdown code fences are not valid official output",
        )

    try:
        payload: Any = json.loads(output)
    except json.JSONDecodeError as exc:
        return _failure(
            "invalid_json",
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        )

    if not isinstance(payload, dict):
        return _failure("root_not_object", "model output must be one JSON object")

    parsed: dict[str, dict[str, int | str]] = {}
    for dimension in DIMENSIONS:
        if dimension not in payload:
            return _failure(
                "missing_dimension",
                f"required dimension '{dimension}' is missing",
                dimension,
            )

        entry = payload[dimension]
        if not isinstance(entry, dict):
            return _failure(
                "dimension_not_object",
                f"'{dimension}' must be a JSON object",
                dimension,
            )

        for field in ("score", "rationale"):
            if field not in entry:
                path = f"{dimension}.{field}"
                return _failure(
                    "missing_field",
                    f"required field '{path}' is missing",
                    path,
                )

        score = entry["score"]
        score_path = f"{dimension}.score"
        if isinstance(score, bool) or not isinstance(score, int):
            return _failure(
                "invalid_score_type",
                f"'{score_path}' must be a Python int and must not be bool",
                score_path,
            )
        if score < 1 or score > 5:
            return _failure(
                "score_out_of_range",
                f"'{score_path}' must be between 1 and 5",
                score_path,
            )

        rationale = entry["rationale"]
        rationale_path = f"{dimension}.rationale"
        if not isinstance(rationale, str):
            return _failure(
                "invalid_rationale_type",
                f"'{rationale_path}' must be a string",
                rationale_path,
            )
        if not rationale.strip():
            return _failure(
                "empty_rationale",
                f"'{rationale_path}' must not be empty",
                rationale_path,
            )

        parsed[dimension] = {"score": score, "rationale": rationale}

    return ParseResult(value=parsed)
