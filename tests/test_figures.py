from __future__ import annotations

import unittest

from src import make_figures
from src.qlora.targets import build_assistant_target


class FigureTests(unittest.TestCase):
    def test_committed_figures_match_generator(self) -> None:
        for path, render in make_figures.FIGURES.items():
            with self.subTest(path.name):
                committed = path.read_text(encoding="utf-8").replace("\r\n", "\n")
                self.assertEqual(
                    committed,
                    render(),
                    f"{path.name} is stale; run python src/make_figures.py",
                )

    def test_distributions_are_parsed_from_experiment_log(self) -> None:
        markdown = make_figures.EXPERIMENTS_PATH.read_text(encoding="utf-8")
        series, gt_means = make_figures.parse_distributions(markdown)

        self.assertEqual(list(series), ["Zero-shot", "QLoRA v1 (selected)", "QLoRA v1b"])
        self.assertEqual(series["Zero-shot"]["organization"], {1: 0, 2: 0, 3: 11, 4: 84, 5: 304})
        self.assertEqual(series["QLoRA v1 (selected)"]["organization"][3], 242)
        for label, total in (("Zero-shot", 399), ("QLoRA v1 (selected)", 400), ("QLoRA v1b", 400)):
            for dimension in ("content", "organization", "expression"):
                self.assertEqual(sum(series[label][dimension].values()), total, (label, dimension))
        self.assertAlmostEqual(gt_means["expression"], 3.676875)

    def test_ground_truth_counts_match_validation_size_and_means(self) -> None:
        markdown = make_figures.EXPERIMENTS_PATH.read_text(encoding="utf-8")
        truth = make_figures.parse_ground_truth(markdown)
        _, gt_means = make_figures.parse_distributions(markdown)

        self.assertEqual(truth["organization"], {1: 5, 2: 56, 3: 141, 4: 157, 5: 41})
        for dimension, counts in truth.items():
            self.assertEqual(sum(counts.values()), 400, dimension)
            # the rounded mean must stay within half a point of the unrounded mean
            rounded_mean = sum(score * n for score, n in counts.items()) / 400
            self.assertLess(abs(rounded_mean - gt_means[dimension]), 0.5, dimension)

    def test_model_metrics_are_parsed_from_experiment_log(self) -> None:
        rows = make_figures.parse_model_metrics(
            make_figures.EXPERIMENTS_PATH.read_text(encoding="utf-8")
        )
        by_label = {label: (rmse, spearman) for label, rmse, spearman, _ in rows}

        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(by_label["Mean baseline"][0], 0.787031)
        self.assertIsNone(by_label["Mean baseline"][1])
        self.assertAlmostEqual(by_label["QLoRA v1 (selected)"][0], 0.6384419374257172)
        self.assertAlmostEqual(by_label["QLoRA v1, official test"][1], 0.5697)

    def test_correlations_are_parsed_from_experiment_log(self) -> None:
        data = make_figures.parse_correlations(
            make_figures.EXPERIMENTS_PATH.read_text(encoding="utf-8")
        )
        self.assertEqual(list(data), ["Train (2,000)", "Validation (400)"])
        self.assertAlmostEqual(data["Validation (400)"][("content", "organization")], 0.741)

    def test_missing_section_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "not found"):
            make_figures.parse_distributions("# Experiment Log\n")

    def test_target_lines_keep_every_character_in_order(self) -> None:
        target = build_assistant_target(make_figures.EXAMPLE_SCORES)
        lines = make_figures.target_lines(target)

        self.assertEqual("".join(text for line in lines for text, _ in line), target.text)
        self.assertEqual(len(lines), 6)  # key+score line and rationale line per dimension
        self.assertTrue(all(line[0][1] == "rationale" for line in lines[1::2]))
        score_chunks = [text for line in lines for text, role in line if role == "score"]
        self.assertEqual(score_chunks, ["4", "3", "4"])


if __name__ == "__main__":
    unittest.main()
