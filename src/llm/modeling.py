"""Load the pinned Qwen model revision with bitsandbytes 4-bit quantization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
from dataclasses import dataclass, field
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
    adapter_metadata: dict[str, Any] = field(
        default_factory=lambda: {"enabled": False}
    )


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


def runtime_versions(
    torch_module: Any, *, include_peft: bool = False
) -> dict[str, str]:
    versions = {
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "torch": str(torch_module.__version__),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
    }
    if include_peft:
        versions["peft"] = importlib.metadata.version("peft")
    return versions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ModelLoadError(f"{description} must be a JSON object: {path}")
    return payload


def _adapter_training_metadata(adapter_path: Path) -> tuple[Path | None, dict[str, Any]]:
    for candidate in (
        adapter_path / "run_metadata.json",
        adapter_path.parent / "run_metadata.json",
    ):
        if candidate.is_file():
            return candidate, _load_json_object(candidate, "adapter training metadata")
    return None, {}


def inspect_adapter(
    adapter_path: Path,
    *,
    expected_model_id: str,
    expected_revision: str | None,
) -> dict[str, Any]:
    """Validate one local PEFT adapter and return reproducibility metadata."""
    resolved_path = adapter_path.expanduser().resolve()
    if not resolved_path.is_dir():
        raise ModelLoadError(f"adapter path is not a directory: {resolved_path}")

    config_path = resolved_path / "adapter_config.json"
    if not config_path.is_file():
        raise ModelLoadError(f"adapter_config.json does not exist: {config_path}")
    adapter_config = _load_json_object(config_path, "adapter config")

    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model:
        raise ModelLoadError("adapter config needs base_model_name_or_path")
    if base_model != expected_model_id:
        raise ModelLoadError(
            "adapter base model is incompatible: "
            f"adapter={base_model!r}, requested={expected_model_id!r}"
        )

    peft_type = adapter_config.get("peft_type")
    task_type = adapter_config.get("task_type")
    if not isinstance(peft_type, str) or peft_type.upper() != "LORA":
        raise ModelLoadError(f"adapter peft_type must be LORA, got {peft_type!r}")
    if not isinstance(task_type, str) or task_type.upper() != "CAUSAL_LM":
        raise ModelLoadError(
            f"adapter task_type must be CAUSAL_LM, got {task_type!r}"
        )

    adapter_revision = adapter_config.get("revision")
    if adapter_revision is not None and (
        not isinstance(adapter_revision, str)
        or not COMMIT_SHA_PATTERN.fullmatch(adapter_revision)
    ):
        raise ModelLoadError(
            "adapter revision must be an immutable 40-character SHA when present"
        )
    if (
        expected_revision is not None
        and adapter_revision is not None
        and adapter_revision != expected_revision
    ):
        raise ModelLoadError(
            "adapter revision is incompatible: "
            f"adapter={adapter_revision!r}, requested={expected_revision!r}"
        )

    training_metadata_path, training_metadata = _adapter_training_metadata(
        resolved_path
    )
    metadata_model_id = training_metadata.get("model_id")
    if metadata_model_id is not None and metadata_model_id != expected_model_id:
        raise ModelLoadError(
            "adapter training metadata model_id is incompatible: "
            f"adapter={metadata_model_id!r}, requested={expected_model_id!r}"
        )
    metadata_revision = training_metadata.get("revision")
    if metadata_revision is not None and (
        not isinstance(metadata_revision, str)
        or not COMMIT_SHA_PATTERN.fullmatch(metadata_revision)
    ):
        raise ModelLoadError(
            "adapter training metadata revision must be an immutable "
            "40-character SHA"
        )
    if (
        adapter_revision is not None
        and metadata_revision is not None
        and adapter_revision != metadata_revision
    ):
        raise ModelLoadError(
            "adapter config and training metadata revisions disagree: "
            f"config={adapter_revision!r}, metadata={metadata_revision!r}"
        )
    if (
        expected_revision is not None
        and metadata_revision is not None
        and metadata_revision != expected_revision
    ):
        raise ModelLoadError(
            "adapter training metadata revision is incompatible: "
            f"adapter={metadata_revision!r}, requested={expected_revision!r}"
        )

    weight_candidates = (
        resolved_path / "adapter_model.safetensors",
        resolved_path / "adapter_model.bin",
    )
    weight_path = next((path for path in weight_candidates if path.is_file()), None)
    if weight_path is None:
        raise ModelLoadError(
            "adapter weights do not exist (expected adapter_model.safetensors or "
            "adapter_model.bin)"
        )

    fingerprint_files = (config_path, weight_path)
    file_manifest = {
        path.name: {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in fingerprint_files
    }
    fingerprint_payload = json.dumps(
        file_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()

    verified_revision = adapter_revision or metadata_revision
    revision_source = (
        "adapter_config"
        if adapter_revision is not None
        else "training_metadata"
        if metadata_revision is not None
        else None
    )
    return {
        "enabled": True,
        "path": str(resolved_path),
        "peft_type": peft_type,
        "task_type": task_type,
        "base_model_name_or_path": base_model,
        "revision": verified_revision,
        "revision_source": revision_source,
        "adapter_name": "default",
        "is_trainable": False,
        "loaded_for_inference": True,
        "inference_mode": adapter_config.get("inference_mode"),
        "merged": False,
        "config": adapter_config,
        "training_metadata_path": (
            str(training_metadata_path.resolve())
            if training_metadata_path is not None
            else None
        ),
        "fingerprint_algorithm": "sha256(canonical JSON file manifest)",
        "fingerprint_sha256": fingerprint,
        "files": file_manifest,
    }


def load_quantized_qwen(
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    *,
    allow_cpu_disk_offload: bool = False,
    api: Any | None = None,
    adapter_path: Path | None = None,
) -> LoadedModel:
    """Resolve one immutable revision and load tokenizer and model from it.

    Resolving the repository metadata happens before either ``from_pretrained``
    call. Both tokenizer and model receive the same exact commit SHA.
    """
    hf_home = require_hf_home()
    hub_cache = hf_home / "hub"
    revision = resolve_model_revision(model_id, api=api)
    adapter_metadata = (
        inspect_adapter(
            adapter_path,
            expected_model_id=model_id,
            expected_revision=revision,
        )
        if adapter_path is not None
        else {"enabled": False}
    )

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
    base_placement = inspect_model_placement(model)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(adapter_path.expanduser().resolve())
        )
    model.eval()

    placement = inspect_model_placement(model)
    if not placement.device_map:
        placement = base_placement
    after_load = cuda_memory_report(torch)
    LOGGER.info("Model device map: %s", json.dumps(placement.device_map))
    LOGGER.info("CUDA memory after model load: %s", json.dumps(after_load))
    if not allow_cpu_disk_offload:
        if base_placement.has_cpu_or_disk_offload:
            raise ModelOffloadError(base_placement)
        if placement.has_cpu_or_disk_offload:
            raise ModelOffloadError(placement)

    return LoadedModel(
        model_id=model_id,
        revision=revision,
        tokenizer=tokenizer,
        model=model,
        placement=placement,
        hf_home=hf_home,
        runtime_versions=runtime_versions(
            torch, include_peft=adapter_metadata["enabled"]
        ),
        cuda_memory_before_load=before_load,
        cuda_memory_after_load=after_load,
        adapter_metadata=adapter_metadata,
    )
