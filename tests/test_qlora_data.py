from __future__ import annotations

import json
import unittest

from src.qlora.data import TrainingDataError, build_training_sample, reject_validation_path
from src.qlora.rationale_templates import RUBRIC_RATIONALES, rationale_for
from src.qlora.targets import (
    AssistantTarget,
    CharacterSpan,
    build_assistant_target,
    validate_character_spans,
)


class QLoRADataTests(unittest.TestCase):
    def record(self) -> dict:
        return {
            "id": "sample-1",
            "document_id": "document-1",
            "prompt": "논제",
            "essay": "글 본문",
            "score": {
                "content": 3.5,
                "organization": 4.25,
                "expression": 3.25,
                "average": object(),
            },
        }

    def test_average_is_ignored_and_round_half_up_is_reused(self) -> None:
        sample = build_training_sample(self.record())
        self.assertEqual(
            sample.gold_scores,
            {"content": 4, "organization": 4, "expression": 3},
        )

    def test_gold_scores_and_rubric_rationales_form_complete_json(self) -> None:
        sample = build_training_sample(self.record())
        target = build_assistant_target(sample.gold_scores)
        payload = json.loads(target.text)
        self.assertEqual(set(payload), {"content", "organization", "expression"})
        for dimension, score in sample.gold_scores.items():
            self.assertEqual(payload[dimension]["score"], score)
            self.assertEqual(payload[dimension]["rationale"], rationale_for(dimension, score))
        self.assertNotIn("average", target.text)

    def test_target_has_exactly_three_non_overlapping_score_spans(self) -> None:
        target = build_assistant_target(build_training_sample(self.record()).gold_scores)
        validate_character_spans(target)
        score_spans = [span for span in target.spans if span.role == "score"]

        self.assertEqual(len(score_spans), 3)
        self.assertEqual(
            [span.dimension for span in score_spans],
            ["content", "organization", "expression"],
        )
        self.assertTrue(
            all(
                left.end == right.start
                for left, right in zip(target.spans, target.spans[1:])
            )
        )
        self.assertEqual(target.spans[0].start, 0)
        self.assertEqual(target.spans[-1].end, len(target.text))

    def test_overlapping_role_spans_are_rejected(self) -> None:
        invalid = AssistantTarget(
            text="abc",
            spans=(
                CharacterSpan(0, 2, "structure"),
                CharacterSpan(1, 3, "score", "content"),
            ),
            version="test",
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_character_spans(invalid)

    def test_rationale_templates_cover_only_dimensions_and_scores(self) -> None:
        self.assertEqual(
            set(RUBRIC_RATIONALES), {"content", "organization", "expression"}
        )
        for templates in RUBRIC_RATIONALES.values():
            self.assertEqual(set(templates), {1, 2, 3, 4, 5})
            self.assertTrue(all(value.strip() for value in templates.values()))
        self.assertIn("논리적 연결", rationale_for("content", 4))
        self.assertIn("문단 연결", rationale_for("organization", 4))
        self.assertIn("어문 규범", rationale_for("expression", 4))

    def test_validation_path_is_rejected(self) -> None:
        from pathlib import Path

        with self.assertRaisesRegex(TrainingDataError, "validation JSONL is forbidden"):
            reject_validation_path(Path("competition_validation.jsonl"))


if __name__ == "__main__":
    unittest.main()
