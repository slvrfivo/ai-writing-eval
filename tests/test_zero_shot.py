from __future__ import annotations

import unittest
from pathlib import Path

from src.zero_shot import parse_args


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
            ]
        )
        self.assertEqual(args.input, Path("input.jsonl"))
        self.assertEqual(args.output_dir, Path("outputs/zero_shot/qwen3_4b"))
        self.assertEqual(args.limit, 1)


if __name__ == "__main__":
    unittest.main()
