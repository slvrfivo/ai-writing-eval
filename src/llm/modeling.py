"""Load the pinned Qwen model revision with bitsandbytes 4-bit quantization."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LOGGER = logging.getLogger(__name__)
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class ModelLoadError(RuntimeError):
    """Raised when the requested model cannot be loaded safely."""


class ModelOffloadError(ModelLoadError):
    """Raised when automatic placement offloads modules to CPU or disk."""

    def __init__(self, placement: "ModelPlacement") -> None:
        super().__init__(
            "device_map='auto' offloaded model modules to CPU or disk: "
            f"{placement.offloaded_modules}"
        )
        self.placement = placement


@dataclass(frozen=True)
class ModelPlacement:
    device_map: dict[str, str]
    offloaded_modules: dict[str, str]

    @property
    def has_cpu_or_disk_offload(self) -> bool:
        return bool(self.offloaded_modules)


@dataclass(frozen=True)
class LoadedModel:
    model_id: str
    revision: str
    tokenizer: Any
    model: Any
    placement: ModelPlacement
    hf_home: Path
    runtime_versions: dict[str, str]
    cuda_memory_before_load: dict[str, Any]
    cuda_memory_after_load: dict[str, Any]


def require_hf_home() -> Path:
    """Return the configured Hugging Face home without silently choosing a cache."""
    value = os.environ.get("HF_HOME")
    if not value:
        raise ModelLoadError("HF_HOME must be set before resolving or loading a model")
    return Path(value).expanduser().resolve()


def resolve_model_revision(model_id: str, *, api: Any | None = None) -> str:
    """Resolve the current repository HEAD to an immutable Hugging Face commit SHA."""
    require_hf_home()
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    model_info = api.model_info(repo_id=model_id)
    revision = getattr(model_info, "sha", None)
    if not isinstance(revision, str) or not COMMIT_SHA_PATTERN.fullmatch(revision):
        raise ModelLoadError(
            f"Hugging Face did not return a valid 40-character commit SHA for {model_id}"
        )
    return revision


def inspect_model_placement(model: Any) -> ModelPlacement:
    """Inspect the Accelerate device map and identify CPU or disk offload."""
    raw_device_map = getattr(model, "hf_device_map", None)
    if not isinstance(raw_device_map, Mapping):
        return ModelPlacement(device_map={}, offloaded_modules={})

    device_map = {str(name): str(device) for name, device in raw_device_map.items()}
    offloaded = {
        name: device
        for name, device in device_map.items()
        if device.lower() in {"cpu", "disk"}
    }
    return ModelPlacement(device_map=device_map, offloaded_modules=offloaded)


def cuda_memory_report(torch_module: Any) -> dict[str, Any]:
    """Collect CUDA allocation and device capacity without mutating CUDA state."""
    cuda = torch_module.cuda
    if not cuda.is_available():
        return {
            "available": False,
            "device": None,
            "total_vram_bytes": 0,
            "total_vram_gib": 0.0,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }

    device_index = cuda.current_device()
    properties = cuda.get_device_properties(device_index)
    total_bytes = int(properties.total_memory)
    return {
        "available": True,
        "device": cuda.get_device_name(device_index),
        "device_index": int(device_index),
        "total_vram_bytes": total_bytes,
        "total_vram_gib": round(total_bytes / 1024**3, 3),
        "allocated_bytes": int(cuda.memory_allocated(device_index)),
        "reserved_bytes": int(cuda.memory_reserved(device_index)),
        "peak_allocated_bytes": int(cuda.max_memory_allocated(device_index)),
        "peak_reserved_bytes": int(cuda.max_memory_reserved(device_index)),
    }


def runtime_versions(torch_module: Any) -> dict[str, str]:
    return {
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "torch": str(torch_module.__version__),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
    }


def load_quantized_qwen(
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    *,
    allow_cpu_disk_offload: bool = False,
    api: Any | None = None,
) -> LoadedModel:
    """Resolve one immutable revision and load tokenizer and model from it.

    Resolving the repository metadata happens before either ``from_pretrained``
    call. Both tokenizer and model receive the same exact commit SHA.
    """
    hf_home = require_hf_home()
    hub_cache = hf_home / "hub"
    revision = resolve_model_revision(model_id, api=api)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    before_load = cuda_memory_report(torch)
    LOGGER.info("CUDA memory before model load: %s", json.dumps(before_load))

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        cache_dir=hub_cache,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        cache_dir=hub_cache,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    placement = inspect_model_placement(model)
    after_load = cuda_memory_report(torch)
    LOGGER.info("Model device map: %s", json.dumps(placement.device_map))
    LOGGER.info("CUDA memory after model load: %s", json.dumps(after_load))
    if placement.has_cpu_or_disk_offload and not allow_cpu_disk_offload:
        raise ModelOffloadError(placement)

    return LoadedModel(
        model_id=model_id,
        revision=revision,
        tokenizer=tokenizer,
        model=model,
        placement=placement,
        hf_home=hf_home,
        runtime_versions=runtime_versions(torch),
        cuda_memory_before_load=before_load,
        cuda_memory_after_load=after_load,
    )
