"""Configuration validation for the first weighted QLoRA experiment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ROLE_NAMES = ("prompt", "structure", "score", "rationale")
SUPPORTED_TARGET_CONSTRUCTION_VERSION = "rubric_general_v1"


class QLoRAConfigError(ValueError):
    """Raised when the training configuration is unsafe or incomplete."""


def _object(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise QLoRAConfigError(f"{key} must be a JSON object")
    return value


@dataclass(frozen=True)
class QLoRAConfig:
    model_id: str
    revision: str
    prompt_version: str
    target_construction_version: str
    max_seq_length: int | None
    quantization: dict[str, Any]
    score_loss_weight: float
    structure_loss_weight: float
    rationale_loss_weight: float
    class_balancing: dict[str, Any]
    lora: dict[str, Any]
    training: dict[str, Any]

    @classmethod
    def from_json(cls, path: Path) -> "QLoRAConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise QLoRAConfigError("training config must be a JSON object")
        config = cls(
            model_id=payload.get("model_id"),
            revision=payload.get("revision"),
            prompt_version=payload.get("prompt_version"),
            target_construction_version=payload.get("target_construction_version"),
            max_seq_length=payload.get("max_seq_length"),
            quantization=_object(payload, "quantization"),
            score_loss_weight=payload.get("score_loss_weight"),
            structure_loss_weight=payload.get("structure_loss_weight"),
            rationale_loss_weight=payload.get("rationale_loss_weight"),
            class_balancing=_object(payload, "class_balancing"),
            lora=_object(payload, "lora"),
            training=_object(payload, "training"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise QLoRAConfigError("model_id must be a non-empty string")
        if not isinstance(self.revision, str) or not COMMIT_SHA_PATTERN.fullmatch(
            self.revision
        ):
            raise QLoRAConfigError("revision must be an immutable 40-character SHA")
        for name, value in (
            ("prompt_version", self.prompt_version),
            ("target_construction_version", self.target_construction_version),
        ):
            if not isinstance(value, str) or not value:
                raise QLoRAConfigError(f"{name} must be a non-empty string")
        if self.target_construction_version != SUPPORTED_TARGET_CONSTRUCTION_VERSION:
            raise QLoRAConfigError(
                "unsupported target_construction_version: "
                f"{self.target_construction_version}"
            )
        if self.max_seq_length is not None and (
            isinstance(self.max_seq_length, bool)
            or not isinstance(self.max_seq_length, int)
            or self.max_seq_length <= 0
        ):
            raise QLoRAConfigError("max_seq_length must be null or a positive integer")

        expected_quantization = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        }
        if self.quantization != expected_quantization:
            raise QLoRAConfigError("quantization must match the NF4/BF16 QLoRA setup")

        for name, value in (
            ("score_loss_weight", self.score_loss_weight),
            ("structure_loss_weight", self.structure_loss_weight),
            ("rationale_loss_weight", self.rationale_loss_weight),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise QLoRAConfigError(f"{name} must be non-negative")

        expected_class_balancing_keys = {
            "enabled",
            "method",
            "clip_min",
            "clip_max",
        }
        if set(self.class_balancing) != expected_class_balancing_keys:
            raise QLoRAConfigError(
                "class_balancing must contain enabled, method, clip_min, clip_max"
            )
        if not isinstance(self.class_balancing["enabled"], bool):
            raise QLoRAConfigError("class_balancing.enabled must be boolean")
        if self.class_balancing["method"] != "bounded_inverse_sqrt_v1":
            raise QLoRAConfigError(
                "class_balancing.method must be bounded_inverse_sqrt_v1"
            )
        if (
            self.class_balancing["clip_min"] != 0.75
            or self.class_balancing["clip_max"] != 2.0
        ):
            raise QLoRAConfigError("class_balancing clip bounds must be [0.75, 2.0]")

        required_lora = {
            "target_modules": "all-linear",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if self.lora != required_lora:
            raise QLoRAConfigError("LoRA config does not match weighted QLoRA v1")

        required_training = {
            "num_train_epochs",
            "learning_rate",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "bf16",
            "gradient_checkpointing",
            "optimizer",
            "warmup_ratio",
            "lr_scheduler_type",
            "weight_decay",
            "max_grad_norm",
            "seed",
            "logging_steps",
            "save_steps",
            "save_total_limit",
        }
        missing = required_training - set(self.training)
        if missing:
            raise QLoRAConfigError(f"training config is missing fields: {sorted(missing)}")

    def with_max_seq_length(self, value: int | None) -> "QLoRAConfig":
        selected = self.max_seq_length if value is None else value
        updated = QLoRAConfig(
            model_id=self.model_id,
            revision=self.revision,
            prompt_version=self.prompt_version,
            target_construction_version=self.target_construction_version,
            max_seq_length=selected,
            quantization=dict(self.quantization),
            score_loss_weight=self.score_loss_weight,
            structure_loss_weight=self.structure_loss_weight,
            rationale_loss_weight=self.rationale_loss_weight,
            class_balancing=dict(self.class_balancing),
            lora=dict(self.lora),
            training=dict(self.training),
        )
        updated.validate()
        return updated

    def with_class_balancing(self, enabled: bool | None) -> "QLoRAConfig":
        if enabled is None:
            return self
        if not isinstance(enabled, bool):
            raise QLoRAConfigError("class balancing override must be boolean")
        updated = QLoRAConfig(
            model_id=self.model_id,
            revision=self.revision,
            prompt_version=self.prompt_version,
            target_construction_version=self.target_construction_version,
            max_seq_length=self.max_seq_length,
            quantization=dict(self.quantization),
            score_loss_weight=self.score_loss_weight,
            structure_loss_weight=self.structure_loss_weight,
            rationale_loss_weight=self.rationale_loss_weight,
            class_balancing={**self.class_balancing, "enabled": enabled},
            lora=dict(self.lora),
            training=dict(self.training),
        )
        updated.validate()
        return updated

    @property
    def loss_weights(self) -> dict[str, float]:
        return {
            "prompt": 0.0,
            "structure": float(self.structure_loss_weight),
            "score": float(self.score_loss_weight),
            "rationale": float(self.rationale_loss_weight),
        }
