"""End-to-end weighted QLoRA preparation, training, and metadata recording."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from ..llm.modeling import cuda_memory_report
    from ..llm.prompting import load_prompt_snapshot
except ImportError:  # python src/train_qlora.py
    from llm.modeling import cuda_memory_report
    from llm.prompting import load_prompt_snapshot

from .config import QLoRAConfig
from .data import TrainingSample, iter_training_samples
from .loss import weighted_trainer_class
from .modeling import LoadedQLoRA, trainable_parameter_stats
from .tokenization import (
    TokenLengthStats,
    TokenizedExample,
    TokenizedTrainingDataset,
    WeightedDataCollator,
    calculate_token_length_stats,
    encode_training_example,
)


RUN_METADATA_FILENAME = "run_metadata.json"


@dataclass(frozen=True)
class PreparedTrainingData:
    examples: list[TokenizedExample]
    token_length_stats: TokenLengthStats


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def prepare_training_data(
    samples: Iterable[TrainingSample],
    *,
    tokenizer: Any,
    config: QLoRAConfig,
) -> PreparedTrainingData:
    examples = [
        encode_training_example(
            sample,
            tokenizer=tokenizer,
            prompt_version=config.prompt_version,
            loss_weights=config.loss_weights,
            max_seq_length=config.max_seq_length,
        )
        for sample in samples
    ]
    stats = calculate_token_length_stats(
        [example.token_length for example in examples]
    )
    return PreparedTrainingData(examples=examples, token_length_stats=stats)


def inspect_training_file(
    path: Path, *, tokenizer: Any, config: QLoRAConfig
) -> PreparedTrainingData:
    return prepare_training_data(
        iter_training_samples(path), tokenizer=tokenizer, config=config
    )


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reset_peak_memory(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()


def _training_arguments(
    config: QLoRAConfig, *, output_dir: Path, max_steps: int | None
) -> Any:
    from transformers import TrainingArguments

    values = config.training
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(values["num_train_epochs"]),
        max_steps=-1 if max_steps is None else max_steps,
        learning_rate=float(values["learning_rate"]),
        per_device_train_batch_size=int(values["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(values["gradient_accumulation_steps"]),
        bf16=bool(values["bf16"]),
        gradient_checkpointing=bool(values["gradient_checkpointing"]),
        optim=str(values["optimizer"]),
        warmup_ratio=float(values["warmup_ratio"]),
        lr_scheduler_type=str(values["lr_scheduler_type"]),
        weight_decay=float(values["weight_decay"]),
        max_grad_norm=float(values["max_grad_norm"]),
        seed=int(values["seed"]),
        data_seed=int(values["seed"]),
        logging_steps=int(values["logging_steps"]),
        save_steps=int(values["save_steps"]),
        save_total_limit=int(values["save_total_limit"]),
        save_strategy="steps",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )


def run_qlora_training(
    *,
    train_path: Path,
    output_dir: Path,
    config: QLoRAConfig,
    prepared: PreparedTrainingData,
    loaded: LoadedQLoRA,
    max_steps: int | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Train and save only PEFT adapter checkpoints plus reproducibility metadata."""
    if config.max_seq_length is None:
        raise ValueError("max_seq_length must be selected from inspector statistics")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer")
    if torch_module is None:
        import torch as torch_module

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / RUN_METADATA_FILENAME
    snapshot = load_prompt_snapshot(config.prompt_version)
    parameter_stats = trainable_parameter_stats(loaded.model)
    training_args = _training_arguments(
        config, output_dir=output_dir, max_steps=max_steps
    )
    dataset = TokenizedTrainingDataset(prepared.examples)
    collator = WeightedDataCollator(loaded.tokenizer.pad_token_id)
    trainer_type = weighted_trainer_class()
    trainer = trainer_type(
        model=loaded.model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    expected_steps = math.ceil(
        len(dataset)
        / (
            int(config.training["per_device_train_batch_size"])
            * int(config.training["gradient_accumulation_steps"])
        )
    )
    metadata: dict[str, Any] = {
        "model_id": config.model_id,
        "revision": config.revision,
        "package_versions": dict(loaded.runtime_versions),
        "dataset_path": str(train_path.resolve()),
        "dataset_size": len(dataset),
        "token_length_stats": prepared.token_length_stats.as_dict(),
        "max_seq_length": config.max_seq_length,
        "prompt_version": snapshot.version,
        "prompt_snapshot_sha256": snapshot.sha256,
        "target_construction_version": config.target_construction_version,
        "loss_weights": dict(config.loss_weights),
        "quantization_config": dict(config.quantization),
        "lora_config": dict(config.lora),
        "training_config": dict(config.training),
        **parameter_stats,
        "requested_max_steps": max_steps,
        "expected_optimizer_steps_for_one_epoch": expected_steps,
        "started_at": utc_now(),
        "finished_at": None,
        "status": "running",
        "elapsed_seconds": None,
        "optimizer_steps": 0,
        "seconds_per_optimizer_step": None,
        "estimated_one_epoch_seconds": None,
        "train_loss": None,
        "cuda_memory_before_model_load": loaded.cuda_memory_before_load,
        "cuda_memory_after_model_load": loaded.cuda_memory_after_load,
    }
    _write_metadata(metadata_path, metadata)
    _reset_peak_memory(torch_module)
    started = time.perf_counter()
    failure: BaseException | None = None
    train_result: Any | None = None
    try:
        train_result = trainer.train()
        final_adapter_dir = output_dir / "final_adapter"
        loaded.model.save_pretrained(final_adapter_dir, safe_serialization=True)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        elapsed = time.perf_counter() - started
        steps = int(getattr(trainer.state, "global_step", 0))
        seconds_per_step = None if steps == 0 else elapsed / steps
        try:
            memory = cuda_memory_report(torch_module)
        except Exception as exc:
            memory = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        metrics = getattr(train_result, "metrics", {}) if train_result is not None else {}
        metadata.update(
            {
                "finished_at": utc_now(),
                "status": "failed" if failure is not None else "completed",
                "elapsed_seconds": elapsed,
                "optimizer_steps": steps,
                "seconds_per_optimizer_step": seconds_per_step,
                "estimated_one_epoch_seconds": (
                    None
                    if seconds_per_step is None
                    else seconds_per_step * expected_steps
                ),
                "train_loss": metrics.get("train_loss"),
                "cuda_memory_after_training": memory,
                "peak_allocated_vram_bytes": memory.get(
                    "peak_allocated_bytes", 0
                ),
                "peak_reserved_vram_bytes": memory.get("peak_reserved_bytes", 0),
            }
        )
        if failure is not None:
            metadata["error"] = {
                "type": type(failure).__name__,
                "message": str(failure),
            }
        _write_metadata(metadata_path, metadata)
    return metadata
