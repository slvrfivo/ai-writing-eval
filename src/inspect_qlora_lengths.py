"""Inspect complete official-chat QLoRA sequence lengths without truncation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from .qlora.config import QLoRAConfig
    from .qlora.data import reject_validation_path
    from .qlora.diagnostics import (
        build_loss_mask_debug,
        calculate_role_token_statistics,
    )
    from .qlora.modeling import load_training_tokenizer
    from .qlora.tokenization import MIXED_BOUNDARY_POLICY
    from .qlora.training import inspect_training_file
except ImportError:  # python src/inspect_qlora_lengths.py
    from qlora.config import QLoRAConfig
    from qlora.data import reject_validation_path
    from qlora.diagnostics import (
        build_loss_mask_debug,
        calculate_role_token_statistics,
    )
    from qlora.modeling import load_training_tokenizer
    from qlora.tokenization import MIXED_BOUNDARY_POLICY
    from qlora.training import inspect_training_file


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
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "qwen3_4b_qlora_v1.json",
    )
    parser.add_argument(
        "--max-seq-length",
        type=positive_integer,
        help="optional candidate limit; any longer sample causes an error",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--debug-first-sample",
        action="store_true",
        help="include an assistant-token loss-mask inspection for the first sample",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input.resolve()
    reject_validation_path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"train JSONL does not exist: {input_path}")
    config = QLoRAConfig.from_json(args.config.resolve()).with_max_seq_length(
        args.max_seq_length
    )
    tokenizer = load_training_tokenizer(config)
    prepared = inspect_training_file(
        input_path, tokenizer=tokenizer, config=config
    )
    report = {
        "model_id": config.model_id,
        "revision": config.revision,
        "dataset_size": len(prepared.examples),
        "target_construction_version": config.target_construction_version,
        "mixed_boundary_token_policy": MIXED_BOUNDARY_POLICY,
        "max_seq_length_checked": config.max_seq_length,
        "token_length_stats": prepared.token_length_stats.as_dict(),
        "role_token_stats": calculate_role_token_statistics(
            prepared.examples, config.loss_weights
        ),
    }
    if args.debug_first_sample:
        report["first_sample_loss_mask_debug"] = build_loss_mask_debug(
            prepared.examples[0], tokenizer=tokenizer
        )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.resolve().write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
