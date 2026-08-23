"""Short rubric-only rationales for score-focused QLoRA v1 targets."""

from __future__ import annotations

try:
    from ..evaluate import DIMENSIONS
except ImportError:  # python src/train_qlora.py
    from evaluate import DIMENSIONS


TARGET_CONSTRUCTION_VERSION = "rubric_general_v1"

RUBRIC_RATIONALES: dict[str, dict[int, str]] = {
    "content": {
        1: "주장과 근거가 기준을 거의 충족하지 못하고 논리적 연결에 심각한 결함이 있는 수준이다.",
        2: "주장과 근거가 충분하지 않고 논리적 연결의 주요 결함으로 기준 충족이 제한적인 수준이다.",
        3: "주장과 근거 및 논리적 연결에 장점과 약점이 함께 있어 기준을 부분적으로 충족하는 수준이다.",
        4: "주장과 근거가 전반적으로 충실하고 논리적 연결이 타당하여 기준을 잘 충족하는 수준이다.",
        5: "주장과 근거가 매우 충실하고 구체적이며 논리적 연결의 결함이 거의 없는 수준이다.",
    },
    "organization": {
        1: "글의 구조와 문단 연결 및 전개 순서가 기준을 거의 충족하지 못하는 수준이다.",
        2: "글의 구조와 문단 연결 또는 전개 순서에 주요 결함이 있어 기준 충족이 제한적인 수준이다.",
        3: "글의 구조와 문단 연결 및 전개 순서에 장점과 약점이 함께 있어 기준을 부분적으로 충족하는 수준이다.",
        4: "글의 구조가 드러나고 문단 연결과 전개 순서가 전반적으로 자연스러워 기준을 잘 충족하는 수준이다.",
        5: "글의 구조가 뚜렷하고 문단 연결과 전개 순서가 매우 자연스러우며 결함이 거의 없는 수준이다.",
    },
    "expression": {
        1: "문장과 어휘 및 어문 규범이 기준을 거의 충족하지 못하고 심각한 결함이 있는 수준이다.",
        2: "문장이나 어휘 또는 어문 규범에 주요 결함이 있어 이해와 기준 충족이 제한적인 수준이다.",
        3: "문장과 어휘 및 어문 규범에 장점과 약점이 함께 있어 기준을 부분적으로 충족하는 수준이다.",
        4: "문장이 전반적으로 자연스럽고 어휘와 어문 규범이 적절하여 기준을 잘 충족하는 수준이다.",
        5: "문장이 매우 자연스럽고 어휘와 어문 규범의 결함이 거의 없어 이해하기 쉬운 수준이다.",
    },
}


def rationale_for(dimension: str, score: int) -> str:
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown scoring dimension: {dimension}")
    if isinstance(score, bool) or score not in RUBRIC_RATIONALES[dimension]:
        raise ValueError(f"score must be an integer from 1 to 5: {score!r}")
    return RUBRIC_RATIONALES[dimension][score]
