"""2026 AI말평 글쓰기 채점 능력 평가용 재사용 evaluator.

2026-08-06 공식 공지에 따라 예측 점수는 ROUND_HALF_UP으로 정수화한 뒤
content, organization, expression의 RMSE와 Spearman 상관계수를 계산한다.
"""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DIMENSIONS = ("content", "organization", "expression")
MIN_SCORE = Decimal("1")
MAX_SCORE = Decimal("5")
INTEGER_QUANTUM = Decimal("1")
IDENTIFIER_FIELDS = ("id", "essay_id", "document_id")


class EvaluationError(ValueError):
    """평가 입력이나 지표 계산을 진행할 수 없을 때 발생한다."""


class PredictionValidationError(EvaluationError):
    """예측 JSON이 공식 출력 조건을 만족하지 않을 때 발생한다."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _numeric_decimal(value: Any, location: str) -> Decimal:
    """bool·문자열·NaN·무한대를 거부하고 유한한 숫자를 Decimal로 바꾼다."""
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise PredictionValidationError(f"{location}: score는 숫자여야 합니다.")

    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PredictionValidationError(f"{location}: 유효하지 않은 숫자입니다.") from exc

    if not numeric.is_finite():
        raise PredictionValidationError(f"{location}: NaN 또는 무한대는 허용되지 않습니다.")
    return numeric


def round_half_up_score(value: Any, location: str = "score") -> int:
    """1~5 범위를 확인한 뒤 사사오입(ROUND_HALF_UP)으로 정수화한다.

    범위 검사는 반올림 전에 수행한다. 따라서 0.9나 5.1을 1 또는 5로
    보정하지 않고 공식 출력 범위 위반으로 처리한다.
    """
    numeric = _numeric_decimal(value, location)
    if numeric < MIN_SCORE or numeric > MAX_SCORE:
        raise PredictionValidationError(
            f"{location}: 예측 점수 {numeric}은 공식 범위 1~5를 벗어났습니다."
        )
    return int(numeric.quantize(INTEGER_QUANTUM, rounding=ROUND_HALF_UP))


def validate_prediction(
    prediction: Mapping[str, Any],
    *,
    require_rationale: bool = False,
    location: str = "prediction",
) -> dict[str, int]:
    """세 영역 예측 구조를 검증하고 ROUND_HALF_UP 정수 점수를 반환한다."""
    if not isinstance(prediction, Mapping):
        raise PredictionValidationError(f"{location}: JSON 객체여야 합니다.")

    rounded: dict[str, int] = {}
    for dimension in DIMENSIONS:
        dimension_location = f"{location}.{dimension}"
        if dimension not in prediction:
            raise PredictionValidationError(f"{dimension_location}: 필수 영역이 없습니다.")

        entry = prediction[dimension]
        if not isinstance(entry, Mapping):
            raise PredictionValidationError(
                f"{dimension_location}: score를 포함한 JSON 객체여야 합니다."
            )
        if "score" not in entry:
            raise PredictionValidationError(f"{dimension_location}.score: 필수 필드가 없습니다.")

        rounded[dimension] = round_half_up_score(
            entry["score"], f"{dimension_location}.score"
        )

        if require_rationale and "rationale" not in entry:
            raise PredictionValidationError(
                f"{dimension_location}.rationale: 필수 필드가 없습니다."
            )
        if "rationale" in entry and not isinstance(entry["rationale"], str):
            raise PredictionValidationError(
                f"{dimension_location}.rationale: 문자열이어야 합니다."
            )

    return rounded


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Square Error를 계산한다."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
        raise EvaluationError("RMSE 입력은 길이가 같은 1차원 배열이어야 합니다.")
    if truth.size == 0:
        raise EvaluationError("RMSE를 계산할 데이터가 없습니다.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise EvaluationError("RMSE 입력에 NaN 또는 무한대가 있습니다.")
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """동점에 평균 순위를 부여한다. 순위는 1부터 시작한다."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise EvaluationError("순위 입력은 1차원 배열이어야 합니다.")
    if not np.isfinite(array).all():
        raise EvaluationError("순위 입력에 NaN 또는 무한대가 있습니다.")

    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """동점 평균 순위를 사용해 Spearman 상관계수를 계산한다."""
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape:
        raise EvaluationError("Spearman 입력은 길이가 같은 1차원 배열이어야 합니다.")
    if truth.size < 2:
        raise EvaluationError("Spearman 계산에는 최소 2개 표본이 필요합니다.")

    truth_ranks = _average_ranks(truth)
    prediction_ranks = _average_ranks(prediction)
    truth_centered = truth_ranks - truth_ranks.mean()
    prediction_centered = prediction_ranks - prediction_ranks.mean()
    denominator = math.sqrt(
        float(np.dot(truth_centered, truth_centered))
        * float(np.dot(prediction_centered, prediction_centered))
    )
    if denominator == 0:
        raise EvaluationError("모든 값이 같아 Spearman 상관계수를 정의할 수 없습니다.")
    return float(np.dot(truth_centered, prediction_centered) / denominator)


def _ground_truth_score(record: Mapping[str, Any], dimension: str, location: str) -> float:
    score_object = record.get("score")
    if not isinstance(score_object, Mapping):
        raise EvaluationError(f"{location}.score: JSON 객체가 필요합니다.")
    if dimension not in score_object:
        raise EvaluationError(f"{location}.score.{dimension}: 정답 점수가 없습니다.")

    value = score_object[dimension]
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise EvaluationError(f"{location}.score.{dimension}: 숫자여야 합니다.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EvaluationError(f"{location}.score.{dimension}: 유한한 숫자여야 합니다.")
    if numeric < float(MIN_SCORE) or numeric > float(MAX_SCORE):
        raise EvaluationError(f"{location}.score.{dimension}: 1~5 범위를 벗어났습니다.")
    return numeric


def _prediction_payload(record: Mapping[str, Any], location: str) -> Mapping[str, Any]:
    """공식 제출 예시의 judge 래퍼와 영역이 직접 있는 형식을 모두 지원한다."""
    if "judge" not in record:
        return record
    payload = record["judge"]
    if not isinstance(payload, Mapping):
        raise PredictionValidationError(f"{location}.judge: JSON 객체여야 합니다.")
    return payload


def _truth_index(
    ground_truth: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, str], dict[str, Mapping[str, Any]]]:
    canonical_order: list[str] = []
    aliases: dict[str, str] = {}
    records_by_id: dict[str, Mapping[str, Any]] = {}

    for index, record in enumerate(ground_truth):
        location = f"ground_truth[{index}]"
        if not isinstance(record, Mapping):
            raise EvaluationError(f"{location}: JSON 객체여야 합니다.")
        canonical = record.get("id")
        if not isinstance(canonical, str) or not canonical:
            raise EvaluationError(f"{location}.id: 비어 있지 않은 문자열이어야 합니다.")
        if canonical in records_by_id:
            raise EvaluationError(f"{location}.id: 중복 id '{canonical}'입니다.")

        canonical_order.append(canonical)
        records_by_id[canonical] = record
        for field in ("id", "document_id"):
            alias = record.get(field)
            if alias is None:
                continue
            if not isinstance(alias, str) or not alias:
                raise EvaluationError(f"{location}.{field}: 비어 있지 않은 문자열이어야 합니다.")
            if alias in aliases and aliases[alias] != canonical:
                raise EvaluationError(f"{location}.{field}: 중복 식별자 '{alias}'입니다.")
            aliases[alias] = canonical

        for dimension in DIMENSIONS:
            _ground_truth_score(record, dimension, location)

    if not canonical_order:
        raise EvaluationError("정답 데이터가 비어 있습니다.")
    return canonical_order, aliases, records_by_id


def _resolve_prediction_id(
    record: Mapping[str, Any], aliases: Mapping[str, str], location: str
) -> str:
    supplied = [record[field] for field in IDENTIFIER_FIELDS if field in record]
    if not supplied:
        raise PredictionValidationError(
            f"{location}: id, essay_id, document_id 중 하나가 필요합니다."
        )
    if any(not isinstance(value, str) or not value for value in supplied):
        raise PredictionValidationError(f"{location}: 식별자는 비어 있지 않은 문자열이어야 합니다.")

    resolved = {aliases[value] for value in supplied if value in aliases}
    unknown = [value for value in supplied if value not in aliases]
    if unknown:
        raise PredictionValidationError(
            f"{location}: 정답 데이터에 없는 식별자입니다: {unknown}"
        )
    if len(resolved) != 1:
        raise PredictionValidationError(f"{location}: 서로 다른 레코드를 가리키는 식별자입니다.")
    return resolved.pop()


def evaluate_predictions(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    require_rationale: bool = False,
) -> dict[str, Any]:
    """ID로 정렬한 뒤 세 공식 영역만 평가한다. score.average는 읽지 않는다."""
    canonical_order, aliases, truth_by_id = _truth_index(ground_truth)
    rounded_by_id: dict[str, dict[str, int]] = {}

    for index, record in enumerate(predictions):
        location = f"predictions[{index}]"
        if not isinstance(record, Mapping):
            raise PredictionValidationError(f"{location}: JSON 객체여야 합니다.")
        canonical = _resolve_prediction_id(record, aliases, location)
        if canonical in rounded_by_id:
            raise PredictionValidationError(f"{location}: 중복 예측 id '{canonical}'입니다.")
        rounded_by_id[canonical] = validate_prediction(
            _prediction_payload(record, location),
            require_rationale=require_rationale,
            location=location,
        )

    missing = [canonical for canonical in canonical_order if canonical not in rounded_by_id]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise PredictionValidationError(
            f"예측이 없는 정답 레코드가 {len(missing)}개입니다: {preview}{suffix}"
        )

    dimension_metrics: dict[str, dict[str, float]] = {}
    for dimension in DIMENSIONS:
        truth_values = [
            _ground_truth_score(truth_by_id[canonical], dimension, f"ground_truth[{canonical}]")
            for canonical in canonical_order
        ]
        prediction_values = [
            rounded_by_id[canonical][dimension] for canonical in canonical_order
        ]
        dimension_metrics[dimension] = {
            "rmse": rmse(truth_values, prediction_values),
            "spearman": spearman_correlation(truth_values, prediction_values),
        }

    return {
        "n_samples": len(canonical_order),
        "rounding": "ROUND_HALF_UP",
        "score_range": [1, 5],
        "dimensions": dimension_metrics,
        "mean": {
            "rmse": float(
                np.mean([dimension_metrics[name]["rmse"] for name in DIMENSIONS])
            ),
            "spearman": float(
                np.mean([dimension_metrics[name]["spearman"] for name in DIMENSIONS])
            ),
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{line_number} JSON 파싱 실패: {exc}") from exc
            if not isinstance(record, dict):
                raise EvaluationError(f"{path}:{line_number}: JSON 객체여야 합니다.")
            records.append(record)
    return records


def load_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path}: JSON 파싱 실패: {exc}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("predictions"), list):
        payload = payload["predictions"]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise EvaluationError(
            f"{path}: JSON 객체 배열 또는 predictions 배열을 가진 객체여야 합니다."
        )
    return payload


def discover_validation_file(data_dir: Path) -> Path:
    matches = sorted(
        path
        for path in data_dir.rglob("*.jsonl")
        if path.is_file() and "validation" in path.stem.lower()
    )
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "없음"
        raise EvaluationError(
            f"validation JSONL은 정확히 1개여야 합니다. 발견 결과: {found}"
        )
    return matches[0]


def ground_truth_as_predictions(
    ground_truth: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """평가 파이프라인 sanity check용 예측을 메모리에서 만든다."""
    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(ground_truth):
        if not isinstance(record, Mapping):
            raise EvaluationError(f"ground_truth[{index}]: JSON 객체여야 합니다.")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise EvaluationError(f"ground_truth[{index}].id: 문자열이어야 합니다.")
        predictions.append(
            {
                "id": identifier,
                **{
                    dimension: {
                        "score": _ground_truth_score(record, dimension, f"ground_truth[{index}]"),
                        "rationale": "ground-truth sanity check",
                    }
                    for dimension in DIMENSIONS
                },
            }
        )
    return predictions


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="정답 validation JSONL. 생략하면 data/raw에서 자동 탐색합니다.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path, help="예측 JSON 또는 JSONL")
    mode.add_argument(
        "--sanity-check",
        action="store_true",
        help="validation 정답 점수를 예측으로 넣어 평가 파이프라인을 점검합니다.",
    )
    parser.add_argument(
        "--require-rationale",
        action="store_true",
        help="각 영역 rationale의 존재와 문자열 타입을 필수로 검사합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth_path = (
        args.ground_truth.resolve()
        if args.ground_truth
        else discover_validation_file(project_root() / "data" / "raw")
    )
    ground_truth = load_jsonl(truth_path)
    predictions = (
        ground_truth_as_predictions(ground_truth)
        if args.sanity_check
        else load_predictions(args.predictions.resolve())
    )
    metrics = evaluate_predictions(
        ground_truth,
        predictions,
        require_rationale=args.require_rationale,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
