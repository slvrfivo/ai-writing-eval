"""Train the first weighted score-focused QLoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from .qlora.config import QLoRAConfig
    from .qlora.data import reject_validation_path
    from .qlora.modeling import load_qlora_model, load_training_tokenizer
    from .qlora.training import inspect_training_file, run_qlora_training
except ImportError:  # python src/train_qlora.py
    from qlora.config import QLoRAConfig
    from qlora.data import reject_validation_path
    from qlora.modeling import load_qlora_model, load_training_tokenizer
    from qlora.training import inspect_training_file, run_qlora_training


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="train JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "qwen3_4b_qlora_v1.json",
    )
    parser.add_argument(
        "--max-seq-length",
        type=positive_integer,
        help="optional config override; no truncation is applied",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_integer,
        help="optimizer-step cap, e.g. 20 for the A100 MIG smoke benchmark",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train_path = args.input.resolve()
    reject_validation_path(train_path)
    if not train_path.is_file():
        raise FileNotFoundError(f"train JSONL does not exist: {train_path}")
    config = QLoRAConfig.from_json(args.config.resolve()).with_max_seq_length(
        args.max_seq_length
    )

    tokenizer = load_training_tokenizer(config)
    prepared = inspect_training_file(
        train_path, tokenizer=tokenizer, config=config
    )
    loaded = load_qlora_model(config, tokenizer=tokenizer)
    metadata = run_qlora_training(
        train_path=train_path,
        output_dir=args.output_dir.resolve(),
        config=config,
        prepared=prepared,
        loaded=loaded,
        max_steps=args.max_steps,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
