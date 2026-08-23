from __future__ import annotations

import unittest
from pathlib import Path

from src.inspect_qlora_lengths import parse_args as parse_inspector_args
from src.train_qlora import parse_args as parse_training_args


class QLoRACliTests(unittest.TestCase):
    def test_inspector_accepts_candidate_length(self) -> None:
        args = parse_inspector_args(
            [
                "--input",
                "train.jsonl",
                "--max-seq-length",
                "3072",
                "--debug-first-sample",
            ]
        )
        self.assertEqual(args.input, Path("train.jsonl"))
        self.assertEqual(args.max_seq_length, 3072)
        self.assertTrue(args.debug_first_sample)

    def test_training_supports_twenty_step_benchmark(self) -> None:
        args = parse_training_args(
            [
                "--input",
                "train.jsonl",
                "--output-dir",
                "checkpoints/qlora-v1",
                "--max-seq-length",
                "3072",
                "--max-steps",
                "20",
            ]
        )
        self.assertEqual(args.output_dir, Path("checkpoints/qlora-v1"))
        self.assertEqual(args.max_seq_length, 3072)
        self.assertEqual(args.max_steps, 20)


if __name__ == "__main__":
    unittest.main()
