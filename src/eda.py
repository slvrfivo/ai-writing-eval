"""2026 AI말평 글쓰기 채점 데이터의 재현 가능한 탐색적 분석.

원본 JSONL은 읽기 전용으로 열며, 분석 결과만 outputs/ 아래에 기록한다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLITS = ("train", "validation")
TOP_LEVEL_FIELDS = ("id", "document_id", "prompt_num", "prompt", "essay", "score")
SCORE_FIELDS = ("content", "organization", "expression", "average")
COMPONENT_SCORES = SCORE_FIELDS[:3]
FLAT_SCORE_FIELDS = tuple(f"score.{name}" for name in SCORE_FIELDS)
LENGTH_COLUMN = "essay_length"
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class Dataset:
    split: str
    path: Path
    records: list[dict[str, Any]]
    frame: pd.DataFrame


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_jsonl(data_dir: Path) -> dict[str, Path]:
    """data_dir에서 train/validation JSONL을 이름으로 자동 탐색한다."""
    candidates = sorted(path for path in data_dir.rglob("*.jsonl") if path.is_file())
    discovered: dict[str, Path] = {}

    for split in SPLITS:
        matches = [path for path in candidates if split in path.stem.lower()]
        if len(matches) != 1:
            names = ", ".join(str(path) for path in matches) or "없음"
            raise FileNotFoundError(
                f"'{split}' JSONL은 정확히 1개여야 합니다. 발견 결과: {names}"
            )
        discovered[split] = matches[0]

    return discovered


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """JSONL을 UTF-8로 로드하고 중첩 점수 필드를 평탄화한다."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSON 파싱 실패: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}의 레코드가 JSON 객체가 아닙니다.")
            records.append(record)

    frame = pd.json_normalize(records, sep=".")
    for field in (*TOP_LEVEL_FIELDS[:-1], *FLAT_SCORE_FIELDS):
        if field not in frame.columns:
            frame[field] = np.nan

    for field in FLAT_SCORE_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame[LENGTH_COLUMN] = frame["essay"].map(
        lambda value: len(value) if isinstance(value, str) else np.nan
    )
    return records, frame


def load_datasets(data_dir: Path) -> dict[str, Dataset]:
    paths = discover_jsonl(data_dir)
    datasets: dict[str, Dataset] = {}
    for split, path in paths.items():
        records, frame = load_jsonl(path)
        datasets[split] = Dataset(split, path, records, frame)
    return datasets


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def schema_rows(dataset: Dataset) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for field in TOP_LEVEL_FIELDS:
        present = sum(field in record for record in dataset.records)
        observed = sorted(
            {type_name(record[field]) for record in dataset.records if field in record}
        )
        rows.append([field, ", ".join(observed) or "-", present, len(dataset.records)])

    for score_field in SCORE_FIELDS:
        present = 0
        observed: set[str] = set()
        for record in dataset.records:
            score = record.get("score")
            if isinstance(score, dict) and score_field in score:
                present += 1
                observed.add(type_name(score[score_field]))
        rows.append(
            [f"score.{score_field}", ", ".join(sorted(observed)) or "-", present, len(dataset.records)]
        )
    return rows


def missing_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if series.dtype == object:
        mask |= series.map(lambda value: isinstance(value, str) and not value.strip())
    return mask


def missing_counts(dataset: Dataset) -> dict[str, int]:
    fields = (*TOP_LEVEL_FIELDS[:-1], *FLAT_SCORE_FIELDS)
    return {field: int(missing_mask(dataset.frame[field]).sum()) for field in fields}


def duplicate_summary(frame: pd.DataFrame, column: str) -> tuple[int, int]:
    if column not in frame:
        return 0, 0
    values = frame.loc[~missing_mask(frame[column]), column]
    duplicated = values[values.duplicated(keep=False)]
    return int(duplicated.nunique()), int(len(duplicated))


def numeric_stats(series: pd.Series) -> dict[str, float | int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    result = {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=1)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }
    for quantile in QUANTILES:
        result[f"q{int(quantile * 100):02d}"] = float(numeric.quantile(quantile))
    return result


def score_stats(dataset: Dataset) -> pd.DataFrame:
    rows = []
    for score in SCORE_FIELDS:
        stats = numeric_stats(dataset.frame[f"score.{score}"])
        rows.append({"score": score, **stats})
    return pd.DataFrame(rows).set_index("score")


def score_granularity(dataset: Dataset) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for score in SCORE_FIELDS:
        values = np.sort(dataset.frame[f"score.{score}"].dropna().unique())
        differences = np.diff(values)
        positive = differences[differences > 1e-9]
        minimum_step = float(positive.min()) if len(positive) else math.nan
        rows.append([dataset.split, score, len(values), minimum_step])
    return rows


def length_stats(dataset: Dataset) -> dict[str, float | int]:
    stats = numeric_stats(dataset.frame[LENGTH_COLUMN])
    lengths = dataset.frame[LENGTH_COLUMN]
    valid = lengths.notna()
    total = int(valid.sum())
    below = int((lengths[valid] < 800).sum())
    above = int((lengths[valid] > 1200).sum())
    outside = below + above
    stats.update(
        {
            "below_800": below,
            "above_1200": above,
            "outside": outside,
            "outside_ratio": outside / total if total else math.nan,
        }
    )
    return stats


def score_bin_counts(dataset: Dataset) -> pd.DataFrame:
    edges = np.arange(1.0, 5.01, 0.5)
    edges = np.append(edges, 5.000001)
    labels = [
        f"{edges[i]:.1f}–<{edges[i + 1]:.1f}" if i < len(edges) - 2 else "5.0"
        for i in range(len(edges) - 1)
    ]
    rows: dict[str, list[int]] = {}
    for score in SCORE_FIELDS:
        binned = pd.cut(
            dataset.frame[f"score.{score}"],
            bins=edges,
            labels=labels,
            right=False,
            include_lowest=True,
        )
        rows[score] = [int(value) for value in binned.value_counts(sort=False).tolist()]
    return pd.DataFrame(rows, index=labels).T


def score_correlations(dataset: Dataset, method: str) -> pd.DataFrame:
    columns = [f"score.{score}" for score in COMPONENT_SCORES]
    matrix = dataset.frame[columns].corr(method=method)
    matrix.index = COMPONENT_SCORES
    matrix.columns = COMPONENT_SCORES
    return matrix


def ks_statistic(left: pd.Series, right: pd.Series) -> float:
    x = np.sort(pd.to_numeric(left, errors="coerce").dropna().to_numpy(dtype=float))
    y = np.sort(pd.to_numeric(right, errors="coerce").dropna().to_numpy(dtype=float))
    if len(x) == 0 or len(y) == 0:
        return math.nan
    points = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, points, side="right") / len(x)
    cdf_y = np.searchsorted(y, points, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def standardized_mean_difference(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce").dropna()
    y = pd.to_numeric(right, errors="coerce").dropna()
    pooled = math.sqrt((float(x.var(ddof=1)) + float(y.var(ddof=1))) / 2)
    return (float(y.mean()) - float(x.mean())) / pooled if pooled else 0.0


def difference_label(smd: float) -> str:
    magnitude = abs(smd)
    if magnitude < 0.2:
        return "작음"
    if magnitude < 0.5:
        return "중간"
    return "큼"


def split_comparison(datasets: dict[str, Dataset]) -> pd.DataFrame:
    train = datasets["train"].frame
    validation = datasets["validation"].frame
    columns = [*FLAT_SCORE_FIELDS, LENGTH_COLUMN]
    rows = []
    for column in columns:
        smd = standardized_mean_difference(train[column], validation[column])
        rows.append(
            {
                "variable": column,
                "train_mean": float(train[column].mean()),
                "validation_mean": float(validation[column].mean()),
                "difference": float(validation[column].mean() - train[column].mean()),
                "smd": smd,
                "ks": ks_statistic(train[column], validation[column]),
                "magnitude": difference_label(smd),
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def prompt_comparison(datasets: dict[str, Dataset]) -> tuple[pd.DataFrame, float]:
    counts = {
        split: dataset.frame["prompt_num"].value_counts().sort_index()
        for split, dataset in datasets.items()
    }
    table = pd.concat(counts, axis=1).fillna(0).astype(int)
    for split in SPLITS:
        table[f"{split}_ratio"] = table[split] / table[split].sum()
    table["difference_pp"] = (
        table["validation_ratio"] - table["train_ratio"]
    ) * 100
    tvd = 0.5 * float(
        (table["validation_ratio"] - table["train_ratio"]).abs().sum()
    )
    return table, tvd


def average_consistency(dataset: Dataset) -> tuple[int, float]:
    components = dataset.frame[[f"score.{name}" for name in COMPONENT_SCORES]]
    raw_mean = components.mean(axis=1)
    expected = np.floor(raw_mean * 100 + 0.5 + 1e-12) / 100
    observed = dataset.frame["score.average"]
    differences = (expected - observed).abs()
    return int((differences > 1e-9).sum()), float(differences.max())


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "axes.unicode_minus": False,
            "font.size": 9,
        }
    )


def plot_score_distributions(datasets: dict[str, Dataset], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    bins = np.arange(0.95, 5.051, 0.1)
    colors = {"train": "#3568a8", "validation": "#e07a3f"}
    for axis, score in zip(axes.flat, SCORE_FIELDS):
        for split in SPLITS:
            axis.hist(
                datasets[split].frame[f"score.{score}"].dropna(),
                bins=bins,
                density=True,
                alpha=0.52,
                color=colors[split],
                label=split,
            )
        axis.set_title(score)
        axis.set_xlim(1, 5)
        axis.set_ylabel("density")
    axes.flat[0].legend()
    fig.suptitle("Score distributions: train vs validation")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_length_distributions(datasets: dict[str, Dataset], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    colors = {"train": "#3568a8", "validation": "#e07a3f"}
    min_length = min(int(dataset.frame[LENGTH_COLUMN].min()) for dataset in datasets.values())
    max_length = max(int(dataset.frame[LENGTH_COLUMN].max()) for dataset in datasets.values())
    lower_bound = min(750, min_length - 5)
    bins = np.linspace(lower_bound, max_length + 1, 40)
    for split in SPLITS:
        axis.hist(
            datasets[split].frame[LENGTH_COLUMN].dropna(),
            bins=bins,
            density=True,
            alpha=0.52,
            color=colors[split],
            label=split,
        )
    axis.axvline(800, color="#555555", linestyle="--", linewidth=1, label="800 chars")
    axis.axvline(1200, color="#222222", linestyle="--", linewidth=1, label="1200 chars")
    axis.set(title="Essay length distributions", xlabel="characters (including spaces)", ylabel="density")
    axis.set_xlim(lower_bound, max_length + 5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def prompt_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    digits = "".join(character for character in text if character.isdigit())
    return (int(digits) if digits else 10**9, text)


def plot_prompt_composition(table: pd.DataFrame, output: Path) -> None:
    table = table.loc[sorted(table.index, key=prompt_sort_key)]
    positions = np.arange(len(table))
    width = 0.38
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions - width / 2, table["train_ratio"] * 100, width, label="train", color="#3568a8")
    axis.bar(
        positions + width / 2,
        table["validation_ratio"] * 100,
        width,
        label="validation",
        color="#e07a3f",
    )
    axis.set_xticks(positions, [str(value) for value in table.index])
    axis.set(title="Prompt composition", xlabel="prompt_num", ylabel="share (%)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(datasets: dict[str, Dataset], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, split in zip(axes, SPLITS):
        matrix = score_correlations(datasets[split], "pearson")
        image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(range(3), COMPONENT_SCORES, rotation=25, ha="right")
        axis.set_yticks(range(3), COMPONENT_SCORES)
        axis.set_title(split)
        for row in range(3):
            for column in range(3):
                value = float(matrix.iloc[row, column])
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.70 else "#111111",
                )
    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.04)
    fig.suptitle("Pearson correlations among scoring dimensions")
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.18, top=0.84, wspace=0.35)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def format_value(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        return f"{float(value):,.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(format_value(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def dataframe_table(frame: pd.DataFrame, index_name: str) -> str:
    headers = [index_name, *[str(column) for column in frame.columns]]
    rows = [
        [index, *[frame.at[index, column] for column in frame.columns]]
        for index in frame.index
    ]
    return markdown_table(headers, rows)


def build_report(datasets: dict[str, Dataset], figures_dir: Path) -> str:
    sections: list[str] = [
        "# EDA 요약: 2026 AI말평 글쓰기 채점 능력 평가",
        "",
        f"생성 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "원본 JSONL은 읽기 전용으로 로드했으며, 글자 수는 Python `len()`으로 공백과 문장부호를 포함해 계산했다. 표준편차는 표본 표준편차(`ddof=1`)이다.",
        "",
        "## 1. 데이터 파일과 규모",
        "",
        markdown_table(
            ["split", "파일", "레코드 수"],
            [
                [split, dataset.path.name, len(dataset.records)]
                for split, dataset in datasets.items()
            ],
        ),
    ]

    sections.extend(["", "## 2. 필드 구조"])
    for split, dataset in datasets.items():
        sections.extend(
            [
                "",
                f"### {split}",
                "",
                markdown_table(["필드", "관측 타입", "존재 레코드", "전체"], schema_rows(dataset)),
            ]
        )

    missing_rows = []
    duplicate_rows = []
    for split, dataset in datasets.items():
        missing = missing_counts(dataset)
        missing_rows.extend([split, field, count] for field, count in missing.items())
        for field in ("id", "document_id", "essay"):
            values, rows = duplicate_summary(dataset.frame, field)
            duplicate_rows.append([split, field, values, rows])
    sections.extend(
        [
            "",
            "## 3. 결측값과 중복",
            "",
            "빈 문자열도 결측으로 집계했다.",
            "",
            markdown_table(["split", "필드", "결측 수"], missing_rows),
            "",
            markdown_table(["split", "필드", "중복 값 수", "영향 행 수"], duplicate_rows),
        ]
    )

    prompt_table, prompt_tvd = prompt_comparison(datasets)
    prompt_display = prompt_table.copy()
    prompt_display["train_ratio"] *= 100
    prompt_display["validation_ratio"] *= 100
    sections.extend(
        [
            "",
            "## 4. prompt_num 구성",
            "",
            "비율 열의 단위는 %이다.",
            "",
            dataframe_table(prompt_display, "prompt_num"),
            "",
            f"두 split의 prompt 구성 총변동거리(TVD)는 **{prompt_tvd:.3f}**이다(0이면 동일, 1이면 완전 분리).",
            "",
            "![prompt composition](figures/prompt_composition.png)",
        ]
    )

    sections.extend(["", "## 5. 점수 기술통계"])
    for split, dataset in datasets.items():
        stats = score_stats(dataset)[
            ["count", "mean", "median", "std", "min", "q05", "q25", "q50", "q75", "q95", "max"]
        ]
        sections.extend(["", f"### {split}", "", dataframe_table(stats, "score")])

    granularity_rows = []
    for dataset in datasets.values():
        granularity_rows.extend(score_granularity(dataset))
    sections.extend(
        [
            "",
            "### 점수 라벨 세분도",
            "",
            markdown_table(
                ["split", "score", "고유값 수", "관측 최소 간격"],
                granularity_rows,
            ),
        ]
    )

    distribution_rows = []
    for split, dataset in datasets.items():
        distribution = score_bin_counts(dataset)
        for score, row in distribution.iterrows():
            distribution_rows.append([split, score, *row.tolist()])
    bin_headers = ["split", "score", *score_bin_counts(datasets["train"]).columns.tolist()]
    sections.extend(
        [
            "",
            "### 점수 구간별 개수",
            "",
            markdown_table(bin_headers, distribution_rows),
            "",
            "![score distributions](figures/score_distributions.png)",
        ]
    )

    length_rows = []
    length_outlier_rows = []
    for split, dataset in datasets.items():
        stats = length_stats(dataset)
        length_rows.append(
            [
                split,
                stats["count"],
                stats["mean"],
                stats["median"],
                stats["std"],
                stats["min"],
                stats["q05"],
                stats["q25"],
                stats["q75"],
                stats["q95"],
                stats["max"],
                stats["below_800"],
                stats["above_1200"],
                stats["outside_ratio"] * 100,
            ]
        )
        outliers = dataset.frame.loc[
            (dataset.frame[LENGTH_COLUMN] < 800)
            | (dataset.frame[LENGTH_COLUMN] > 1200),
            ["id", "document_id", "prompt_num", LENGTH_COLUMN],
        ]
        for _, row in outliers.iterrows():
            length_outlier_rows.append(
                [
                    split,
                    row["id"],
                    row["document_id"],
                    row["prompt_num"],
                    int(row[LENGTH_COLUMN]),
                    "800 미만" if row[LENGTH_COLUMN] < 800 else "1200 초과",
                ]
            )
    sections.extend(
        [
            "",
            "## 6. essay 글자 수",
            "",
            markdown_table(
                [
                    "split",
                    "count",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "q05",
                    "q25",
                    "q75",
                    "q95",
                    "max",
                    "<800",
                    ">1200",
                    "범위 밖(%)",
                ],
                length_rows,
            ),
            "",
            "![essay length distributions](figures/essay_length_distributions.png)",
            "",
            "### 800–1200자 범위 밖 레코드",
            "",
            markdown_table(
                ["split", "id", "document_id", "prompt_num", "글자 수", "구분"],
                length_outlier_rows,
            ),
        ]
    )

    sections.extend(["", "## 7. content / organization / expression 상관관계"])
    for split, dataset in datasets.items():
        sections.extend(
            [
                "",
                f"### {split}: Pearson",
                "",
                dataframe_table(score_correlations(dataset, "pearson"), "score"),
                "",
                f"### {split}: Spearman",
                "",
                dataframe_table(score_correlations(dataset, "spearman"), "score"),
            ]
        )
    sections.extend(["", "![score correlations](figures/score_correlations.png)"])

    comparison = split_comparison(datasets)
    sections.extend(
        [
            "",
            "## 8. train / validation 차이",
            "",
            "SMD는 `(validation 평균 - train 평균) / pooled 표준편차`이다. 절댓값 0.2 미만은 작음, 0.2–0.5는 중간, 0.5 이상은 큼으로 표시했다. KS는 두 누적분포의 최대 차이이다.",
            "",
            dataframe_table(comparison, "변수"),
        ]
    )

    train_ids = set(datasets["train"].frame["id"].dropna())
    validation_ids = set(datasets["validation"].frame["id"].dropna())
    train_documents = set(datasets["train"].frame["document_id"].dropna())
    validation_documents = set(datasets["validation"].frame["document_id"].dropna())
    train_essays = set(datasets["train"].frame["essay"].dropna())
    validation_essays = set(datasets["validation"].frame["essay"].dropna())

    average_rows = []
    anomaly_lines = []
    for split, dataset in datasets.items():
        inconsistent, max_difference = average_consistency(dataset)
        average_rows.append(
            [split, inconsistent, inconsistent / len(dataset.frame) * 100, max_difference]
        )
        missing_total = sum(missing_counts(dataset).values())
        duplicate_ids, duplicate_id_rows = duplicate_summary(dataset.frame, "id")
        outside_length = int(
            (
                (dataset.frame[LENGTH_COLUMN] < 800)
                | (dataset.frame[LENGTH_COLUMN] > 1200)
            ).sum()
        )
        out_of_range = int(
            (
                (dataset.frame[list(FLAT_SCORE_FIELDS)] < 1)
                | (dataset.frame[list(FLAT_SCORE_FIELDS)] > 5)
            ).any(axis=1).sum()
        )
        anomaly_lines.append(
            f"- **{split}**: 결측 {missing_total}건, 중복 id 값 {duplicate_ids}개(영향 행 {duplicate_id_rows}개), 1–5 범위 밖 점수 행 {out_of_range}개, 800–1200자 범위 밖 essay {outside_length}개."
        )

    cross_overlap = [
        ["id", len(train_ids & validation_ids)],
        ["document_id", len(train_documents & validation_documents)],
        ["essay exact text", len(train_essays & validation_essays)],
    ]
    max_smd_variable = comparison["smd"].abs().idxmax()
    max_smd = float(comparison.loc[max_smd_variable, "smd"])
    max_prompt_pp = float(prompt_table["difference_pp"].abs().max())

    sections.extend(
        [
            "",
            "### split 간 누출 후보",
            "",
            markdown_table(["비교 키", "교집합 수"], cross_overlap),
            "",
            "### average 재계산 검증",
            "",
            "표시된 세 영역 점수의 산술평균을 일반적인 반올림(0.5 올림)으로 소수 둘째 자리까지 계산해 제공된 `average`와 비교했다.",
            "",
            markdown_table(["split", "불일치 행 수", "불일치율(%)", "최대 절대차"], average_rows),
            "",
            "## 9. 핵심 발견과 확인 사항",
            "",
            *anomaly_lines,
            f"- 표시된 세 영역 점수의 단순 평균 반올림과 `average`가 다른 행은 train **{average_rows[0][1]}건 ({average_rows[0][2]:.2f}%)**, validation **{average_rows[1][1]}건 ({average_rows[1][2]:.2f}%)**이며 최대 차이는 0.01이다. 반올림 전 원점수 또는 별도 산식의 존재를 확인해야 한다.",
            "- 학습 라벨은 `content` 0.1, `organization`·`expression` 0.25, `average` 0.01의 최소 간격이 관측된다. 그러나 2026-08-06 최신 공지는 모델이 1–5 정수를 출력해야 하며 실수 출력은 사사오입해 정수로 변환한다고 명시한다.",
            f"- 가장 큰 split 간 표준화 평균차는 `{max_smd_variable}`의 **{max_smd:.3f} ({difference_label(max_smd)})**이다.",
            f"- prompt_num별 구성비의 최대 차이는 **{max_prompt_pp:.2f}%p**, 전체 TVD는 **{prompt_tvd:.3f}**이다.",
            "- 800–1200자 밖의 글은 오류로 단정할 수 없다. 데이터 분포상의 이상 후보이므로 길이 기반 특성을 사용할 때 별도 확인해야 한다.",
            "- 최신 반올림 규칙에 따라 변환된 정수를 기준으로 RMSE와 Spearman을 계산하며 기존 제출 결과도 같은 기준으로 재산출한다. 데이터의 `average` 관측 규칙과 모델 출력 변환 규칙은 구분해야 한다.",
            "",
            "## 10. 모델링 시사점",
            "",
            "- 세 영역 점수는 서로 연관되어 있으므로 공유 인코더와 영역별 출력 헤드를 함께 검토하되, 공식 프롬프트의 ‘영역별 독립 판단’ 원칙을 유지해야 한다.",
            "- 정량 평가는 RMSE와 Spearman의 비중이 같으므로 절대 오차와 순위 보존을 동시에 검증해야 한다.",
            "- 연속 점수 회귀를 실험하더라도 최종 검증에서는 실수 예측을 사사오입한 1–5 정수로 변환해 공식 평가 조건을 재현해야 한다.",
            "- prompt_num별 성능과 길이 구간별 성능을 별도로 추적해 특정 문항·길이에 대한 편향을 점검해야 한다.",
            "- 점수뿐 아니라 실제 essay에 근거한 영역별 한국어 rationale가 LLM Judge 및 사람 평가 대상이므로, 근거 생성 품질을 점수 회귀와 별도 평가해야 한다.",
        ]
    )

    return "\n".join(sections).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(data_dir)
    configure_plot_style()
    plot_score_distributions(datasets, figures_dir / "score_distributions.png")
    plot_length_distributions(datasets, figures_dir / "essay_length_distributions.png")
    prompt_table, _ = prompt_comparison(datasets)
    plot_prompt_composition(prompt_table, figures_dir / "prompt_composition.png")
    plot_correlations(datasets, figures_dir / "score_correlations.png")

    report = build_report(datasets, figures_dir)
    report_path = output_dir / "eda_summary.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"EDA report: {report_path}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
