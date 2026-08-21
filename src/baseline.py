"""Train 통계만 사용하는 Global Mean / Prompt Mean baseline."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .evaluate import (
        DIMENSIONS,
        EvaluationError,
        evaluate_predictions,
        load_jsonl,
        round_half_up_score,
    )
except ImportError:  # python src/baseline.py로 직접 실행할 때
    from evaluate import (  # type: ignore[no-redef]
        DIMENSIONS,
        EvaluationError,
        evaluate_predictions,
        load_jsonl,
        round_half_up_score,
    )


@dataclass(frozen=True)
class TrainStatistics:
    global_means: dict[str, float]
    prompt_means: dict[str, dict[str, float]]
    prompt_counts: dict[str, int]
    train_count: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_split_file(data_dir: Path, split: str) -> Path:
    matches = sorted(
        path
        for path in data_dir.rglob("*.jsonl")
        if path.is_file() and split.lower() in path.stem.lower()
    )
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "없음"
        raise EvaluationError(
            f"{split} JSONL은 정확히 1개여야 합니다. 발견 결과: {found}"
        )
    return matches[0]


def _train_score(record: Mapping[str, Any], dimension: str, location: str) -> float:
    score_object = record.get("score")
    if not isinstance(score_object, Mapping) or dimension not in score_object:
        raise EvaluationError(f"{location}.score.{dimension}: train 점수가 없습니다.")
    value = score_object[dimension]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EvaluationError(f"{location}.score.{dimension}: 숫자여야 합니다.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 1 or numeric > 5:
        raise EvaluationError(
            f"{location}.score.{dimension}: 유한한 1~5 점수여야 합니다."
        )
    return numeric


def calculate_train_statistics(
    train_records: Sequence[Mapping[str, Any]],
) -> TrainStatistics:
    """오직 train 레코드의 세 영역 점수로 전체·prompt별 평균을 계산한다."""
    if not train_records:
        raise EvaluationError("train 데이터가 비어 있습니다.")

    global_values = {dimension: [] for dimension in DIMENSIONS}
    prompt_values: dict[str, dict[str, list[float]]] = {}

    for index, record in enumerate(train_records):
        location = f"train[{index}]"
        if not isinstance(record, Mapping):
            raise EvaluationError(f"{location}: JSON 객체여야 합니다.")
        prompt_num = record.get("prompt_num")
        if not isinstance(prompt_num, str) or not prompt_num:
            raise EvaluationError(f"{location}.prompt_num: 문자열이어야 합니다.")
        if prompt_num not in prompt_values:
            prompt_values[prompt_num] = {
                dimension: [] for dimension in DIMENSIONS
            }

        for dimension in DIMENSIONS:
            score = _train_score(record, dimension, location)
            global_values[dimension].append(score)
            prompt_values[prompt_num][dimension].append(score)

    global_means = {
        dimension: math.fsum(values) / len(values)
        for dimension, values in global_values.items()
    }
    prompt_means = {
        prompt_num: {
            dimension: math.fsum(values[dimension]) / len(values[dimension])
            for dimension in DIMENSIONS
        }
        for prompt_num, values in sorted(
            prompt_values.items(), key=lambda item: _prompt_sort_key(item[0])
        )
    }
    prompt_counts = {
        prompt_num: len(values[DIMENSIONS[0]])
        for prompt_num, values in sorted(
            prompt_values.items(), key=lambda item: _prompt_sort_key(item[0])
        )
    }
    return TrainStatistics(
        global_means=global_means,
        prompt_means=prompt_means,
        prompt_counts=prompt_counts,
        train_count=len(train_records),
    )


def _prompt_sort_key(value: str) -> tuple[int, str]:
    digits = "".join(character for character in value if character.isdigit())
    return (int(digits) if digits else 10**9, value)


def _validation_identity(record: Mapping[str, Any], location: str) -> str:
    identifier = record.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise EvaluationError(f"{location}.id: 비어 있지 않은 문자열이어야 합니다.")
    return identifier


def _prediction_record(identifier: str, means: Mapping[str, float]) -> dict[str, Any]:
    return {
        "id": identifier,
        **{
            dimension: {"score": means[dimension]}
            for dimension in DIMENSIONS
        },
    }


def build_global_predictions(
    validation_records: Sequence[Mapping[str, Any]],
    global_means: Mapping[str, float],
) -> list[dict[str, Any]]:
    """validation에서는 id만 읽고 모든 샘플에 train 전체 평균을 부여한다."""
    return [
        _prediction_record(
            _validation_identity(record, f"validation[{index}]"), global_means
        )
        for index, record in enumerate(validation_records)
    ]


def build_prompt_predictions(
    validation_records: Sequence[Mapping[str, Any]],
    statistics: TrainStatistics,
) -> tuple[list[dict[str, Any]], int]:
    """validation의 prompt_num에 맞는 train 평균을 사용하고 미관측 prompt는 fallback한다."""
    predictions: list[dict[str, Any]] = []
    fallback_count = 0
    for index, record in enumerate(validation_records):
        location = f"validation[{index}]"
        identifier = _validation_identity(record, location)
        prompt_num = record.get("prompt_num")
        if not isinstance(prompt_num, str) or not prompt_num:
            raise EvaluationError(f"{location}.prompt_num: 문자열이어야 합니다.")

        means = statistics.prompt_means.get(prompt_num)
        if means is None:
            means = statistics.global_means
            fallback_count += 1
        predictions.append(_prediction_record(identifier, means))
    return predictions, fallback_count


def _rounded_means(means: Mapping[str, float]) -> dict[str, int]:
    return {
        dimension: round_half_up_score(value, f"train_mean.{dimension}")
        for dimension, value in means.items()
    }


def run_baselines(
    train_records: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statistics = calculate_train_statistics(train_records)

    global_predictions = build_global_predictions(
        validation_records, statistics.global_means
    )
    prompt_predictions, fallback_count = build_prompt_predictions(
        validation_records, statistics
    )

    global_metrics = evaluate_predictions(validation_records, global_predictions)
    prompt_metrics = evaluate_predictions(validation_records, prompt_predictions)

    return {
        "date": date.today().isoformat(),
        "train_count": statistics.train_count,
        "validation_count": len(validation_records),
        "train_statistics": {
            "global_means": statistics.global_means,
            "global_means_after_official_rounding": _rounded_means(
                statistics.global_means
            ),
            "prompt_means": {
                prompt_num: {
                    "count": statistics.prompt_counts[prompt_num],
                    "means": means,
                    "means_after_official_rounding": _rounded_means(means),
                }
                for prompt_num, means in statistics.prompt_means.items()
            },
        },
        "baselines": {
            "global_mean": {
                "method": "train 영역별 전체 평균을 모든 validation 샘플에 예측",
                "metrics": global_metrics,
            },
            "prompt_mean": {
                "method": "train prompt_num별 영역 평균; 미관측 prompt는 global mean fallback",
                "fallback_count": fallback_count,
                "metrics": prompt_metrics,
            },
        },
    }


def _format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_markdown_summary(results: Mapping[str, Any]) -> str:
    baseline_rows = []
    detail_sections = []
    for key, display_name in (
        ("global_mean", "Global Mean"),
        ("prompt_mean", "Prompt Mean"),
    ):
        baseline = results["baselines"][key]
        metrics = baseline["metrics"]
        baseline_rows.append(
            [
                display_name,
                _format_metric(metrics["mean"]["rmse"]),
                _format_metric(metrics["mean"]["spearman"]),
            ]
        )
        detail_rows = [
            [
                dimension,
                _format_metric(metrics["dimensions"][dimension]["rmse"]),
                _format_metric(metrics["dimensions"][dimension]["spearman"]),
            ]
            for dimension in DIMENSIONS
        ]
        detail_rows.append(
            [
                "mean",
                _format_metric(metrics["mean"]["rmse"]),
                _format_metric(metrics["mean"]["spearman"]),
            ]
        )
        detail_sections.extend(
            [
                f"## {display_name}",
                "",
                baseline["method"],
                "",
                _markdown_table(["영역", "RMSE", "Spearman"], detail_rows),
                "",
            ]
        )

    statistics = results["train_statistics"]
    global_rows = [
        [
            dimension,
            f"{statistics['global_means'][dimension]:.6f}",
            statistics["global_means_after_official_rounding"][dimension],
        ]
        for dimension in DIMENSIONS
    ]
    prompt_rows = []
    for prompt_num, prompt_stats in statistics["prompt_means"].items():
        prompt_rows.append(
            [
                prompt_num,
                prompt_stats["count"],
                *[f"{prompt_stats['means'][dimension]:.6f}" for dimension in DIMENSIONS],
                *[
                    prompt_stats["means_after_official_rounding"][dimension]
                    for dimension in DIMENSIONS
                ],
            ]
        )

    fallback_count = results["baselines"]["prompt_mean"]["fallback_count"]
    return "\n".join(
        [
            "# Statistical Baseline Results",
            "",
            f"- 날짜: {results['date']}",
            f"- train: {results['train_count']}건",
            f"- validation: {results['validation_count']}건",
            "- 평가는 `src/evaluate.py`의 ROUND_HALF_UP 및 공식 세 영역 지표를 사용",
            "",
            "## 요약",
            "",
            _markdown_table(["Baseline", "평균 RMSE", "평균 Spearman"], baseline_rows),
            "",
            *detail_sections,
            "Spearman이 `undefined`이면 공식 정수화 후 예측이 모두 같아 순위 상관계수가 수학적으로 정의되지 않은 것이다.",
            "",
            "## Train global means",
            "",
            _markdown_table(["영역", "원래 평균", "공식 정수화"], global_rows),
            "",
            "## Train prompt_num means",
            "",
            _markdown_table(
                [
                    "prompt_num",
                    "count",
                    "content",
                    "organization",
                    "expression",
                    "content 정수화",
                    "organization 정수화",
                    "expression 정수화",
                ],
                prompt_rows,
            ),
            "",
            f"Prompt Mean의 미관측 prompt fallback 사용 건수: **{fallback_count}건**",
            "",
        ]
    )


def save_results(results: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_results.json"
    markdown_path = output_dir / "baseline_summary.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown_summary(results), encoding="utf-8")
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "outputs" / "baselines"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_path = discover_split_file(data_dir, "train")
    validation_path = discover_split_file(data_dir, "validation")
    results = run_baselines(load_jsonl(train_path), load_jsonl(validation_path))
    json_path, markdown_path = save_results(results, args.output_dir.resolve())
    print(json.dumps(results["baselines"], ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
