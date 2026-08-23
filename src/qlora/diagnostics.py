"""Aggregate role mass and inspect one sample's weighted loss mask."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .targets import CharacterSpan, validate_character_spans
from .tokenization import TOKEN_ROLES, TokenizedExample


ASSISTANT_ROLES = {"structure", "score", "rationale"}


def calculate_role_token_statistics(
    examples: Sequence[TokenizedExample],
    loss_weights: Mapping[str, float],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("role statistics require at least one example")
    if set(loss_weights) != set(TOKEN_ROLES):
        raise ValueError(f"loss_weights must contain exactly {TOKEN_ROLES}")

    counts = {role: 0 for role in TOKEN_ROLES}
    for example in examples:
        if len(example.token_roles) != example.token_length:
            raise ValueError("token role count does not match token length")
        for role in example.token_roles:
            if role not in counts:
                raise ValueError(f"unknown token role: {role}")
            counts[role] += 1

    total_tokens = sum(counts.values())
    weighted_masses = {
        role: counts[role] * float(loss_weights[role]) for role in TOKEN_ROLES
    }
    supervised_weighted_mass = sum(weighted_masses.values())
    if supervised_weighted_mass <= 0:
        raise ValueError("supervised weighted mass must be positive")

    roles = {}
    for role in TOKEN_ROLES:
        roles[role] = {
            "tokens": counts[role],
            "mean_tokens_per_sample": counts[role] / len(examples),
            "token_share": counts[role] / total_tokens,
            "weight": float(loss_weights[role]),
            "weighted_mass": weighted_masses[role],
            "weighted_share": weighted_masses[role] / supervised_weighted_mass,
        }
    return {
        "sample_count": len(examples),
        "total_tokens": total_tokens,
        "supervised_weighted_mass": supervised_weighted_mass,
        "roles": roles,
    }


def _overlapping_spans(
    example: TokenizedExample, offset: tuple[int, int]
) -> list[CharacterSpan]:
    start, end = offset
    if start == end:
        return []
    return [
        span
        for span in example.target.spans
        if start < example.target_start + span.end
        and end > example.target_start + span.start
    ]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_loss_mask_debug(
    example: TokenizedExample, *, tokenizer: Any
) -> dict[str, Any]:
    """Return token rows and explicit invariants for the first training sample."""
    validate_character_spans(example.target)
    score_spans = [span for span in example.target.spans if span.role == "score"]
    score_span_rows = []
    all_score_tokens_correct = True
    all_score_spans_have_tokens = True
    for span in score_spans:
        token_indices = [
            index
            for index, offset in enumerate(example.token_offsets)
            if index >= example.prompt_token_count
            and offset[0] < example.target_start + span.end
            and offset[1] > example.target_start + span.start
        ]
        all_score_spans_have_tokens &= bool(token_indices)
        all_score_tokens_correct &= all(
            example.token_roles[index] == "score" for index in token_indices
        )
        score_span_rows.append(
            {
                "dimension": span.dimension,
                "gold_text": example.target.text[span.start : span.end],
                "token_indices": token_indices,
                "token_count": len(token_indices),
            }
        )

    unambiguous_structure_correct = True
    rationale_tokens_correct = True
    for index in range(example.prompt_token_count, example.token_length):
        spans = _overlapping_spans(example, example.token_offsets[index])
        roles = {span.role for span in spans}
        if roles == {"structure"}:
            unambiguous_structure_correct &= example.token_roles[index] == "structure"
        elif roles == {"rationale"}:
            rationale_tokens_correct &= example.token_roles[index] == "rationale"

    mixed_boundary_tokens = [
        {
            "token_index": boundary.token_index,
            "token_id": example.input_ids[boundary.token_index],
            "decoded_token": _decode_token(
                tokenizer, example.input_ids[boundary.token_index]
            ),
            "overlapping_roles": list(boundary.overlapping_roles),
            "assigned_role": boundary.assigned_role,
            "dimension": boundary.dimension,
        }
        for boundary in example.mixed_boundary_tokens
    ]
    expected_dimensions = [span.dimension for span in score_spans]
    mixed_dimensions = [row["dimension"] for row in mixed_boundary_tokens]

    token_rows = [
        {
            "index": index,
            "token_id": example.input_ids[index],
            "decoded_token": _decode_token(tokenizer, example.input_ids[index]),
            "role": example.token_roles[index],
            "weight": example.loss_weights[index],
            "dimension": example.token_dimensions[index],
        }
        for index in range(example.token_length)
    ]
    assistant_tokens = token_rows[example.prompt_token_count :]
    prompt_all_zero = all(
        example.loss_weights[index] == 0.0
        for index in range(example.prompt_token_count)
    )
    assistant_exactly_one_role = all(
        token["role"] in ASSISTANT_ROLES for token in assistant_tokens
    )
    return {
        "sample_id": example.sample_id,
        "prompt_token_count": example.prompt_token_count,
        "assistant_token_count": len(assistant_tokens),
        "invariants": {
            "score_span_count": len(score_spans),
            "exactly_three_score_spans": len(score_spans) == 3,
            "role_spans_non_overlapping_and_complete": True,
            "all_score_spans_have_tokens": all_score_spans_have_tokens,
            "all_gold_score_tokens_have_score_role": all_score_tokens_correct,
            "score_surrounding_structure_tokens_correct": unambiguous_structure_correct,
            "rationale_tokens_have_rationale_role": rationale_tokens_correct,
            "prompt_tokens_all_weight_zero": prompt_all_zero,
            "assistant_tokens_exactly_one_role": assistant_exactly_one_role,
            "mixed_boundary_token_count": len(mixed_boundary_tokens),
            "mixed_boundary_dimensions": mixed_dimensions,
            "exactly_one_rationale_ending_mixed_token_per_dimension": (
                mixed_dimensions == expected_dimensions
            ),
        },
        "score_spans": score_span_rows,
        "mixed_boundary_tokens": mixed_boundary_tokens,
        "tokens": token_rows,
        "assistant_tokens": assistant_tokens,
    }
