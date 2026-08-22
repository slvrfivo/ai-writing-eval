"""Run the first Qwen3-4B zero-shot writing-score baseline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

try:
    from .llm.modeling import load_quantized_qwen
    from .llm.pipeline import InferenceConfig, run_inference_pipeline
except ImportError:  # python src/zero_shot.py
    from llm.modeling import load_quantized_qwen
    from llm.pipeline import InferenceConfig, run_inference_pipeline


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input JSONL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for raw generations, predictions, failures, and metadata",
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        help="maximum number of unfinished samples to attempt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {input_path}")
    config = InferenceConfig.from_json(
        project_root() / "configs" / "qwen3_4b_zero_shot.json"
    )

    loaded = load_quantized_qwen(config.model_id)

    import torch

    result = run_inference_pipeline(
        input_path=input_path,
        output_dir=args.output_dir.resolve(),
        tokenizer=loaded.tokenizer,
        model=loaded.model,
        model_revision=loaded.revision,
        runtime_versions=loaded.runtime_versions,
        config=config,
        limit=args.limit,
        torch_module=torch,
        cuda_memory_before_load=loaded.cuda_memory_before_load,
        cuda_memory_after_load=loaded.cuda_memory_after_load,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
