from __future__ import annotations

import unittest
from pathlib import Path

from src.infer import parse_args
from src.zero_shot import parse_args as legacy_parse_args


class ZeroShotCliTests(unittest.TestCase):
    def test_required_paths_and_limit_are_parsed(self) -> None:
        args = parse_args(
            [
                "--input",
                "input.jsonl",
                "--output-dir",
                "outputs/zero_shot/qwen3_4b",
                "--limit",
                "1",
                "--adapter",
                "/mnt/checkpoints/qwen3_4b_qlora_v1/final_adapter",
            ]
        )
        self.assertEqual(args.input, Path("input.jsonl"))
        self.assertEqual(args.output_dir, Path("outputs/zero_shot/qwen3_4b"))
        self.assertEqual(args.limit, 1)
        self.assertEqual(
            args.adapter,
            Path("/mnt/checkpoints/qwen3_4b_qlora_v1/final_adapter"),
        )

    def test_adapter_is_optional(self) -> None:
        args = parse_args(
            [
                "--input",
                "input.jsonl",
                "--output-dir",
                "outputs/zero_shot/qwen3_4b",
            ]
        )
        self.assertIsNone(args.adapter)

    def test_legacy_zero_shot_entry_point_still_works(self) -> None:
        self.assertIs(legacy_parse_args, parse_args)


if __name__ == "__main__":
    unittest.main()
