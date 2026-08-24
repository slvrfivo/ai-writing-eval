"""Export the competition-best QLoRA v1 adapter as a standalone HF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from .llm.exporting import (
        DEFAULT_ADAPTER_PATH,
        DEFAULT_MAX_SHARD_SIZE,
        DEFAULT_OUTPUT_PATH,
        DEFAULT_SMOKE_SAMPLES,
        export_merged_model,
    )
    from .llm.pipeline import InferenceConfig
except ImportError:  # python src/export_merged_hf.py
    from llm.exporting import (
        DEFAULT_ADAPTER_PATH,
        DEFAULT_MAX_SHARD_SIZE,
        DEFAULT_OUTPUT_PATH,
        DEFAULT_SMOKE_SAMPLES,
        export_merged_model,
    )
    from llm.pipeline import InferenceConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--validation-input",
        type=Path,
        required=True,
        help="validation JSONL used for post-export smoke inference",
    )
    parser.add_argument(
        "--smoke-samples",
        type=positive_integer,
        default=DEFAULT_SMOKE_SAMPLES,
    )
    parser.add_argument("--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE)
    parser.add_argument(
        "--compare-quantized",
        action="store_true",
        help="also compare scores with pinned base-4bit plus the unmerged adapter",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = project_root()
    config = InferenceConfig.from_json(
        root / "configs" / "qwen3_4b_zero_shot.json"
    )
    metadata = export_merged_model(
        adapter_path=args.adapter,
        output_path=args.output_dir,
        validation_input=args.validation_input,
        inference_config=config,
        project_root=root,
        smoke_sample_count=args.smoke_samples,
        max_shard_size=args.max_shard_size,
        compare_quantized=args.compare_quantized,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
