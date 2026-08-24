from __future__ import annotations

import unittest
from collections import Counter

from src.qlora.data import build_training_sample
from src.qlora.diagnostics import (
    build_loss_mask_debug,
    calculate_role_token_statistics,
)
from src.qlora.tokenization import (
    SequenceLengthError,
    TokenizationError,
    calculate_token_length_stats,
    encode_training_example,
    pad_batch_python,
)


LOSS_WEIGHTS = {
    "prompt": 0.0,
    "structure": 0.25,
    "score": 10.0,
    "rationale": 0.05,
}


class OffsetFakeTokenizer:
    """Character tokenizer that deliberately makes every digit two tokens."""

    def __init__(self) -> None:
        self.rendered_calls: list[str] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.assert_chat_kwargs(kwargs)
        prefix = (
            "<SYSTEM>"
            + messages[0]["content"]
            + "<USER>"
            + messages[1]["content"]
            + "<ASSISTANT>"
        )
        rendered = (
            prefix
            if len(messages) == 2
            else prefix + messages[2]["content"] + "<EOS>"
        )
        self.rendered_calls.append(rendered)
        return rendered

    @staticmethod
    def assert_chat_kwargs(kwargs: dict) -> None:
        if kwargs.get("tokenize") is not False:
            raise AssertionError("chat template must render text before offset tokenization")
        expected_generation = kwargs.get("add_generation_prompt")
        if not isinstance(expected_generation, bool):
            raise AssertionError("add_generation_prompt must be explicit")

    def __call__(self, text: str, **kwargs: object) -> dict:
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for index, character in enumerate(text):
            repetitions = 2 if character.isdigit() else 1
            for repetition in range(repetitions):
                input_ids.append(ord(character) * 2 + repetition)
                offsets.append((index, index + 1))
        result = {"input_ids": input_ids}
        if kwargs.get("return_attention_mask", False):
            result["attention_mask"] = [1] * len(input_ids)
        if kwargs.get("return_offsets_mapping", False):
            result["offset_mapping"] = offsets
        return result

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        del kwargs
        return "".join(chr(token_id // 2) for token_id in token_ids)


class BoundaryMergingFakeTokenizer(OffsetFakeTokenizer):
    def __init__(self, boundary: str) -> None:
        super().__init__()
        self.boundary = boundary
        self.merged_tokens: dict[int, str] = {}

    def __call__(self, text: str, **kwargs: object) -> dict:
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            merge = (
                self.boundary == "rationale_end"
                and text[index : index + 2] == '."'
            ) or (
                self.boundary == "score_start"
                and text[index] == ":"
                and index + 1 < len(text)
                and text[index + 1].isdigit()
            )
            if merge:
                token_id = 1_000_000 + index
                input_ids.append(token_id)
                offsets.append((index, index + 2))
                self.merged_tokens[token_id] = text[index : index + 2]
                index += 2
                continue
            repetitions = 2 if text[index].isdigit() else 1
            for repetition in range(repetitions):
                input_ids.append(ord(text[index]) * 2 + repetition)
                offsets.append((index, index + 1))
            index += 1
        result = {"input_ids": input_ids}
        if kwargs.get("return_attention_mask", False):
            result["attention_mask"] = [1] * len(input_ids)
        if kwargs.get("return_offsets_mapping", False):
            result["offset_mapping"] = offsets
        return result

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        del kwargs
        return "".join(
            self.merged_tokens.get(token_id, chr(token_id // 2))
            for token_id in token_ids
        )


def record() -> dict:
    return {
        "id": "sample-1",
        "document_id": "document-1",
        "prompt": "논제",
        "essay": "본문",
        "validation_score": "MUST_NOT_LEAK",
        "score": {
            "content": 3.5,
            "organization": 4.25,
            "expression": 3.25,
            "average": "AVERAGE_MUST_NOT_LEAK",
        },
    }


class QLoRATokenizationTests(unittest.TestCase):
    def test_official_prompt_roles_weights_and_multi_token_scores(self) -> None:
        tokenizer = OffsetFakeTokenizer()
        example = encode_training_example(
            build_training_sample(record()),
            tokenizer=tokenizer,
            prompt_version="writing_scoring_2026-07-20",
            loss_weights=LOSS_WEIGHTS,
        )

        full_rendered = tokenizer.rendered_calls[-1]
        self.assertIn("[역할]", full_rendered)
        self.assertIn("논제", full_rendered)
        self.assertIn("본문", full_rendered)
        self.assertNotIn("MUST_NOT_LEAK", full_rendered)
        self.assertNotIn("AVERAGE_MUST_NOT_LEAK", full_rendered)

        role_to_weight = dict(zip(example.token_roles, example.loss_weights))
        self.assertEqual(role_to_weight["prompt"], 0.0)
        self.assertEqual(role_to_weight["structure"], 0.25)
        self.assertEqual(role_to_weight["score"], 10.0)
        self.assertEqual(role_to_weight["rationale"], 0.05)
        self.assertEqual(example.token_roles.count("score"), 6)
        self.assertTrue(
            all(
                weight == 10.0
                for role, weight in zip(example.token_roles, example.loss_weights)
                if role == "score"
            )
        )
        self.assertEqual(
            [span.dimension for span in example.target.spans if span.role == "score"],
            ["content", "organization", "expression"],
        )
        self.assertTrue(
            all(role == "prompt" for role in example.token_roles[: example.prompt_token_count])
        )
        self.assertTrue(
            all(weight == 0.0 for weight in example.loss_weights[: example.prompt_token_count])
        )
        self.assertTrue(
            all(
                role in {"structure", "score", "rationale"}
                for role in example.token_roles[example.prompt_token_count :]
            )
        )

        debug = build_loss_mask_debug(example, tokenizer=tokenizer)
        self.assertEqual(debug["invariants"]["score_span_count"], 3)
        self.assertTrue(debug["invariants"]["exactly_three_score_spans"])
        self.assertTrue(debug["invariants"]["role_spans_non_overlapping_and_complete"])
        self.assertTrue(debug["invariants"]["all_score_spans_have_tokens"])
        self.assertTrue(debug["invariants"]["all_gold_score_tokens_have_score_role"])
        self.assertTrue(
            debug["invariants"]["score_surrounding_structure_tokens_correct"]
        )
        self.assertTrue(debug["invariants"]["rationale_tokens_have_rationale_role"])
        self.assertTrue(debug["invariants"]["prompt_tokens_all_weight_zero"])
        self.assertTrue(debug["invariants"]["assistant_tokens_exactly_one_role"])
        self.assertEqual(debug["invariants"]["mixed_boundary_token_count"], 0)
        self.assertEqual(debug["mixed_boundary_tokens"], [])
        self.assertTrue(all(span["token_count"] >= 1 for span in debug["score_spans"]))
        self.assertEqual(len(debug["tokens"]), example.token_length)
        self.assertTrue(
            all(
                token["role"] == "prompt" and token["weight"] == 0.0
                for token in debug["tokens"][: example.prompt_token_count]
            )
        )

    def test_rationale_ending_mixed_tokens_are_assigned_deterministically(self) -> None:
        tokenizer = BoundaryMergingFakeTokenizer("rationale_end")
        example = encode_training_example(
            build_training_sample(record()),
            tokenizer=tokenizer,
            prompt_version="writing_scoring_2026-07-20",
            loss_weights=LOSS_WEIGHTS,
        )
        boundaries = example.mixed_boundary_tokens

        self.assertEqual(len(boundaries), 3)
        self.assertEqual(
            [boundary.dimension for boundary in boundaries],
            ["content", "organization", "expression"],
        )
        self.assertTrue(
            all(
                boundary.overlapping_roles == ("rationale", "structure")
                and boundary.assigned_role == "rationale"
                and example.token_roles[boundary.token_index] == "rationale"
                and example.loss_weights[boundary.token_index] == 0.05
                for boundary in boundaries
            )
        )
        debug = build_loss_mask_debug(example, tokenizer=tokenizer)
        self.assertEqual(debug["invariants"]["mixed_boundary_token_count"], 3)
        self.assertTrue(
            debug["invariants"][
                "exactly_one_rationale_ending_mixed_token_per_dimension"
            ]
        )
        self.assertEqual(
            [row["decoded_token"] for row in debug["mixed_boundary_tokens"]],
            ['."', '."', '."'],
        )

    def test_unexpected_score_structure_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            TokenizationError, "unexpected cross-role token overlap"
        ):
            encode_training_example(
                build_training_sample(record()),
                tokenizer=BoundaryMergingFakeTokenizer("score_start"),
                prompt_version="writing_scoring_2026-07-20",
                loss_weights=LOSS_WEIGHTS,
            )

    def test_role_weighted_mass_is_calculated_exactly(self) -> None:
        example = encode_training_example(
            build_training_sample(record()),
            tokenizer=OffsetFakeTokenizer(),
            prompt_version="writing_scoring_2026-07-20",
            loss_weights=LOSS_WEIGHTS,
        )
        report = calculate_role_token_statistics([example, example], LOSS_WEIGHTS)
        counts = Counter(example.token_roles)
        expected_mass = {
            role: 2 * counts[role] * LOSS_WEIGHTS[role] for role in LOSS_WEIGHTS
        }
        total_mass = sum(expected_mass.values())

        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["total_tokens"], 2 * example.token_length)
        self.assertAlmostEqual(report["supervised_weighted_mass"], total_mass)
        for role in LOSS_WEIGHTS:
            role_report = report["roles"][role]
            self.assertEqual(role_report["tokens"], 2 * counts[role])
            self.assertEqual(role_report["mean_tokens_per_sample"], counts[role])
            self.assertAlmostEqual(
                role_report["weighted_mass"], expected_mass[role]
            )
            self.assertAlmostEqual(
                role_report["weighted_share"], expected_mass[role] / total_mass
            )

    def test_class_multiplier_applies_only_to_score_tokens_by_dimension(self) -> None:
        class_weights = {
            dimension: {label: 1.0 for label in range(1, 6)}
            for dimension in ("content", "organization", "expression")
        }
        class_weights["content"][4] = 1.5
        class_weights["organization"][4] = 0.8
        class_weights["expression"][3] = 2.0
        example = encode_training_example(
            build_training_sample(record()),
            tokenizer=OffsetFakeTokenizer(),
            prompt_version="writing_scoring_2026-07-20",
            loss_weights=LOSS_WEIGHTS,
            score_class_weights=class_weights,
        )

        expected_score_weights = {
            "content": 15.0,
            "organization": 8.0,
            "expression": 20.0,
        }
        for role, dimension, weight in zip(
            example.token_roles, example.token_dimensions, example.loss_weights
        ):
            if role == "score":
                self.assertEqual(weight, expected_score_weights[dimension])
            elif role == "structure":
                self.assertEqual(weight, 0.25)
            elif role == "rationale":
                self.assertEqual(weight, 0.05)
            else:
                self.assertEqual(weight, 0.0)

    def test_max_length_fails_without_truncating(self) -> None:
        with self.assertRaises(SequenceLengthError) as context:
            encode_training_example(
                build_training_sample(record()),
                tokenizer=OffsetFakeTokenizer(),
                prompt_version="writing_scoring_2026-07-20",
                loss_weights=LOSS_WEIGHTS,
                max_seq_length=10,
            )
        self.assertGreater(context.exception.actual, context.exception.maximum)

    def test_token_length_statistics(self) -> None:
        stats = calculate_token_length_stats([100, 200, 300, 2100, 3100, 4100])
        self.assertEqual(stats.minimum, 100)
        self.assertEqual(stats.p50, 300)
        self.assertEqual(stats.p90, 4100)
        self.assertEqual(stats.maximum, 4100)
        self.assertEqual(stats.over_2048, 3)
        self.assertEqual(stats.over_3072, 2)
        self.assertEqual(stats.over_4096, 1)

    def test_collator_padding_has_zero_weight_and_ignored_label(self) -> None:
        batch = pad_batch_python(
            [
                {
                    "input_ids": [1, 2],
                    "attention_mask": [1, 1],
                    "labels": [1, 2],
                    "loss_weights": [0.0, 10.0],
                },
                {
                    "input_ids": [3],
                    "attention_mask": [1],
                    "labels": [3],
                    "loss_weights": [1.0],
                },
            ],
            pad_token_id=99,
        )
        self.assertEqual(batch["input_ids"][1], [3, 99])
        self.assertEqual(batch["labels"][1], [3, -100])
        self.assertEqual(batch["loss_weights"][1], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
