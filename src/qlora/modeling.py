"""Load the immutable Qwen revision and attach an unmerged QLoRA adapter."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

try:
    from ..llm.modeling import (
        ModelLoadError,
        ModelOffloadError,
        ModelPlacement,
        cuda_memory_report,
        inspect_model_placement,
        require_hf_home,
    )
except ImportError:  # python src/train_qlora.py
    from llm.modeling import (
        ModelLoadError,
        ModelOffloadError,
        ModelPlacement,
        cuda_memory_report,
        inspect_model_placement,
        require_hf_home,
    )

from .config import QLoRAConfig


@dataclass(frozen=True)
class LoadedQLoRA:
    tokenizer: Any
    model: Any
    placement: ModelPlacement
    runtime_versions: dict[str, str]
    cuda_memory_before_load: dict[str, Any]
    cuda_memory_after_load: dict[str, Any]


def load_training_tokenizer(config: QLoRAConfig) -> Any:
    """Load only the tokenizer at the exact configured base-model revision."""
    from transformers import AutoTokenizer

    cache_dir = require_hf_home() / "hub"
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=False,
        cache_dir=cache_dir,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ModelLoadError("QLoRA token role mapping requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ModelLoadError("tokenizer needs an EOS token for padding")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def training_runtime_versions(torch_module: Any) -> dict[str, str]:
    return {
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "peft": importlib.metadata.version("peft"),
        "torch": str(torch_module.__version__),
    }


def trainable_parameter_stats(model: Any) -> dict[str, int | float]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = 0.0 if total == 0 else trainable / total
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": ratio,
    }


def load_qlora_model(config: QLoRAConfig, *, tokenizer: Any) -> LoadedQLoRA:
    """Load NF4 base weights and attach LoRA without merging the base model."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cache_dir = require_hf_home() / "hub"
    before_load = cuda_memory_report(torch)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        trust_remote_code=False,
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    placement = inspect_model_placement(model)
    if placement.has_cpu_or_disk_offload:
        raise ModelOffloadError(placement)

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(config.training["gradient_checkpointing"]),
    )
    lora_config = LoraConfig(
        target_modules="all-linear",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, revision=config.revision)
    model.config.use_cache = False
    model.train()
    after_load = cuda_memory_report(torch)
    return LoadedQLoRA(
        tokenizer=tokenizer,
        model=model,
        placement=placement,
        runtime_versions=training_runtime_versions(torch),
        cuda_memory_before_load=before_load,
        cuda_memory_after_load=after_load,
    )
