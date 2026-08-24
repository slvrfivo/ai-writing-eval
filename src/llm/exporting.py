"""Export a pinned QLoRA adapter as a standalone BF16 Hugging Face model."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .modeling import ModelLoadError, inspect_adapter, inspect_model_placement
from .parsing import DIMENSIONS, parse_model_output
from .pipeline import InferenceConfig, InferenceSample, generate_one, iter_input_samples
from .prompting import load_prompt_snapshot


BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DEFAULT_ADAPTER_PATH = Path("/mnt/checkpoints/qwen3_4b_qlora_v1/final_adapter")
DEFAULT_OUTPUT_PATH = Path("/mnt/submissions/qwen3_4b_qlora_v1_merged")
DEFAULT_SUBMISSION_ROOT = Path("/mnt/submissions")
DEFAULT_MAX_SHARD_SIZE = "4GB"
DEFAULT_SMOKE_SAMPLES = 3
EXPORT_METADATA_FILENAME = "export_metadata.json"
GIB = 1024**3
MIN_AVAILABLE_RAM_BYTES = 12 * GIB
MIN_FREE_DISK_BYTES = 20 * GIB
FORBIDDEN_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


class ExportError(RuntimeError):
    """Raised when a standalone model cannot be exported safely."""


class ResourceSafetyError(ExportError):
    """Raised before model loading when RAM or disk headroom is insufficient."""


class ExportValidationError(ExportError):
    """Raised when the saved standalone model fails local validation."""


@dataclass(frozen=True)
class ResourceReport:
    total_ram_bytes: int
    available_ram_bytes: int
    output_disk_free_bytes: int
    output_disk_total_bytes: int
    minimum_available_ram_bytes: int = MIN_AVAILABLE_RAM_BYTES
    minimum_free_disk_bytes: int = MIN_FREE_DISK_BYTES

    @property
    def safe(self) -> bool:
        return (
            self.available_ram_bytes >= self.minimum_available_ram_bytes
            and self.output_disk_free_bytes >= self.minimum_free_disk_bytes
        )

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "total_ram_bytes": self.total_ram_bytes,
            "total_ram_gib": round(self.total_ram_bytes / GIB, 3),
            "available_ram_bytes": self.available_ram_bytes,
            "available_ram_gib": round(self.available_ram_bytes / GIB, 3),
            "output_disk_free_bytes": self.output_disk_free_bytes,
            "output_disk_free_gib": round(self.output_disk_free_bytes / GIB, 3),
            "output_disk_total_bytes": self.output_disk_total_bytes,
            "output_disk_total_gib": round(self.output_disk_total_bytes / GIB, 3),
            "minimum_available_ram_bytes": self.minimum_available_ram_bytes,
            "minimum_available_ram_gib": round(
                self.minimum_available_ram_bytes / GIB, 3
            ),
            "minimum_free_disk_bytes": self.minimum_free_disk_bytes,
            "minimum_free_disk_gib": round(
                self.minimum_free_disk_bytes / GIB, 3
            ),
            "safe": self.safe,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _linux_memory_bytes() -> tuple[int, int]:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        for line in meminfo.read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
        if "MemTotal" in values and "MemAvailable" in values:
            return values["MemTotal"], values["MemAvailable"]

    required = ("SC_PHYS_PAGES", "SC_AVPHYS_PAGES", "SC_PAGE_SIZE")
    if not all(name in os.sysconf_names for name in required):
        raise ResourceSafetyError("cannot determine total and available system RAM")
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    return (
        int(os.sysconf("SC_PHYS_PAGES")) * page_size,
        int(os.sysconf("SC_AVPHYS_PAGES")) * page_size,
    )


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ResourceSafetyError(
                f"cannot find an existing parent for output path: {path}"
            )
        candidate = parent
    return candidate


def inspect_system_resources(output_path: Path) -> ResourceReport:
    """Inspect host RAM and the filesystem that will hold the merged model."""
    total_ram, available_ram = _linux_memory_bytes()
    disk = shutil.disk_usage(_nearest_existing_path(output_path))
    return ResourceReport(
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        output_disk_free_bytes=int(disk.free),
        output_disk_total_bytes=int(disk.total),
    )


def require_safe_resources(report: ResourceReport) -> None:
    if report.safe:
        return
    details = report.as_dict()
    failures = []
    if report.available_ram_bytes < report.minimum_available_ram_bytes:
        failures.append(
            "available RAM "
            f"{details['available_ram_gib']} GiB < "
            f"{details['minimum_available_ram_gib']} GiB"
        )
    if report.output_disk_free_bytes < report.minimum_free_disk_bytes:
        failures.append(
            "free output disk "
            f"{details['output_disk_free_gib']} GiB < "
            f"{details['minimum_free_disk_gib']} GiB"
        )
    raise ResourceSafetyError(
        "BF16 CPU merge preflight failed: " + "; ".join(failures)
    )


def validate_output_path(
    output_path: Path, *, allowed_root: Path = DEFAULT_SUBMISSION_ROOT
) -> Path:
    resolved = output_path.expanduser().resolve()
    root = allowed_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ExportError(
            f"merged model output must stay under {root}: {resolved}"
        ) from exc
    if relative == Path("."):
        raise ExportError("merged model output must be a subdirectory of the root")
    if resolved.exists() and not resolved.is_dir():
        raise ExportError(f"output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise ExportError(f"output directory must be empty: {resolved}")
    return resolved


def source_git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExportError("cannot resolve the source Git commit") from exc
    return {"commit": commit, "dirty": dirty}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _config_invariants(config: Any) -> dict[str, Any]:
    keys = (
        "architectures",
        "model_type",
        "max_position_embeddings",
        "rope_scaling",
        "rope_theta",
        "sliding_window",
        "use_sliding_window",
        "vocab_size",
    )
    return {key: _json_safe(getattr(config, key, None)) for key in keys}


def _tokenizer_invariants(tokenizer: Any) -> dict[str, Any]:
    return {
        "chat_template": getattr(tokenizer, "chat_template", None),
        "special_tokens_map": _json_safe(
            getattr(tokenizer, "special_tokens_map", {})
        ),
        "all_special_ids": list(getattr(tokenizer, "all_special_ids", [])),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _runtime_versions(torch_module: Any) -> dict[str, str]:
    return {
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
        "peft": importlib.metadata.version("peft"),
        "torch": str(torch_module.__version__),
    }


def load_smoke_samples(path: Path, count: int) -> list[InferenceSample]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ExportError("smoke sample count must be a positive integer")
    if not path.is_file():
        raise ExportError(f"validation JSONL does not exist: {path}")
    samples = list(islice(iter_input_samples(path), count))
    if len(samples) != count:
        raise ExportError(
            f"validation JSONL contains only {len(samples)} samples; {count} required"
        )
    return samples


def run_strict_smoke_validation(
    samples: Sequence[InferenceSample],
    *,
    tokenizer: Any,
    model: Any,
    config: InferenceConfig,
    torch_module: Any,
) -> dict[str, Any]:
    """Run official-prompt inference and strictly parse each generated JSON object."""
    config.validate()
    if hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(config.seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available() and hasattr(cuda, "manual_seed_all"):
        cuda.manual_seed_all(config.seed)

    results = []
    for sample in samples:
        generated = generate_one(
            sample,
            tokenizer=tokenizer,
            model=model,
            config=config,
            torch_module=torch_module,
        )
        parsed = parse_model_output(generated.raw_output)
        record: dict[str, Any] = {
            **sample.identity_fields(),
            "raw_output": generated.raw_output,
            "generated_tokens": generated.generated_tokens,
            "generation_truncated": generated.generation_truncated,
            "strict_parse_ok": parsed.ok,
        }
        if parsed.ok:
            assert parsed.value is not None
            record["scores"] = {
                dimension: parsed.value[dimension]["score"]
                for dimension in DIMENSIONS
            }
        else:
            assert parsed.failure is not None
            record["error"] = asdict(parsed.failure)
        results.append(record)

    success_count = sum(bool(item["strict_parse_ok"]) for item in results)
    return {
        "sample_count": len(samples),
        "success_count": success_count,
        "failure_count": len(samples) - success_count,
        "passed": success_count == len(samples),
        "samples": results,
    }


def compare_smoke_scores(
    merged_report: Mapping[str, Any], quantized_report: Mapping[str, Any]
) -> dict[str, Any]:
    merged_samples = list(merged_report.get("samples", []))
    quantized_samples = list(quantized_report.get("samples", []))
    if len(merged_samples) != len(quantized_samples):
        raise ExportValidationError("comparison smoke reports have different sizes")

    comparisons = []
    for merged, quantized in zip(merged_samples, quantized_samples):
        if merged.get("id") != quantized.get("id"):
            raise ExportValidationError("comparison smoke sample IDs do not match")
        merged_scores = merged.get("scores")
        quantized_scores = quantized.get("scores")
        both_valid = isinstance(merged_scores, dict) and isinstance(
            quantized_scores, dict
        )
        item: dict[str, Any] = {
            "id": merged.get("id"),
            "both_strict_parse_ok": both_valid,
            "merged_scores": merged_scores,
            "base_4bit_lora_scores": quantized_scores,
        }
        if both_valid:
            item["score_delta_merged_minus_4bit_lora"] = {
                dimension: merged_scores[dimension] - quantized_scores[dimension]
                for dimension in DIMENSIONS
            }
            item["exact_score_match"] = merged_scores == quantized_scores
        comparisons.append(item)
    return {"enabled": True, "samples": comparisons}


def _validate_saved_files(output_path: Path) -> dict[str, Any]:
    forbidden = [name for name in FORBIDDEN_ADAPTER_FILES if (output_path / name).exists()]
    if forbidden:
        raise ExportValidationError(
            f"merged repository still contains adapter artifacts: {forbidden}"
        )
    required = ("config.json", "generation_config.json", "tokenizer_config.json")
    missing = [name for name in required if not (output_path / name).is_file()]
    if missing:
        raise ExportValidationError(f"merged repository is missing files: {missing}")
    weights = sorted(path.name for path in output_path.glob("*.safetensors"))
    if not weights:
        raise ExportValidationError("merged repository has no safetensors weights")
    files = {
        path.name: path.stat().st_size
        for path in sorted(output_path.iterdir())
        if path.is_file()
    }
    return {
        "required_files_present": True,
        "adapter_artifacts_absent": True,
        "weight_files": weights,
        "files_size_bytes": files,
    }


def _release_cuda_cache(torch_module: Any) -> None:
    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available() and hasattr(cuda, "empty_cache"):
        cuda.empty_cache()


def _load_quantized_adapter_for_comparison(
    *,
    adapter_path: Path,
    cache_dir: Path,
    torch_module: Any,
    auto_model_cls: Any,
    peft_model_cls: Any,
    bitsandbytes_config_cls: Any,
) -> Any:
    quantization = bitsandbytes_config_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_module.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = auto_model_cls.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_REVISION,
        trust_remote_code=False,
        cache_dir=cache_dir,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    placement = inspect_model_placement(base)
    if placement.has_cpu_or_disk_offload:
        raise ModelLoadError(
            "4-bit comparison model unexpectedly used CPU/disk offload: "
            f"{placement.offloaded_modules}"
        )
    model = peft_model_cls.from_pretrained(
        base, str(adapter_path), is_trainable=False
    )
    model.eval()
    return model


def export_merged_model(
    *,
    adapter_path: Path,
    output_path: Path,
    validation_input: Path,
    inference_config: InferenceConfig,
    project_root: Path,
    smoke_sample_count: int = DEFAULT_SMOKE_SAMPLES,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    compare_quantized: bool = False,
    allowed_output_root: Path = DEFAULT_SUBMISSION_ROOT,
    resource_report: ResourceReport | None = None,
    git_state: Mapping[str, Any] | None = None,
    runtime_versions: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
    auto_tokenizer_cls: Any | None = None,
    auto_model_cls: Any | None = None,
    peft_model_cls: Any | None = None,
    bitsandbytes_config_cls: Any | None = None,
    adapter_inspector: Callable[..., dict[str, Any]] = inspect_adapter,
) -> dict[str, Any]:
    """Merge one compatible LoRA adapter into the pinned BF16 base model."""
    if inference_config.model_id != BASE_MODEL_ID:
        raise ExportError("inference config model_id does not match the pinned base")
    inference_config.validate()
    resolved_output = validate_output_path(output_path, allowed_root=allowed_output_root)
    resolved_adapter = adapter_path.expanduser().resolve()
    resolved_validation = validation_input.expanduser().resolve()
    adapter_metadata = adapter_inspector(
        resolved_adapter,
        expected_model_id=BASE_MODEL_ID,
        expected_revision=BASE_REVISION,
    )
    resources = resource_report or inspect_system_resources(resolved_output)
    require_safe_resources(resources)
    samples = load_smoke_samples(resolved_validation, smoke_sample_count)

    if torch_module is None:
        import torch as torch_module
    if auto_tokenizer_cls is None or auto_model_cls is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        auto_tokenizer_cls = auto_tokenizer_cls or AutoTokenizer
        auto_model_cls = auto_model_cls or AutoModelForCausalLM
    if peft_model_cls is None:
        from peft import PeftModel

        peft_model_cls = PeftModel

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise ExportError("HF_HOME must be set before exporting the model")
    cache_dir = Path(hf_home).expanduser().resolve() / "hub"
    source = dict(git_state or source_git_state(project_root))
    versions = dict(runtime_versions or _runtime_versions(torch_module))
    snapshot = load_prompt_snapshot(inference_config.prompt_version)

    resolved_output.mkdir(parents=True, exist_ok=True)
    metadata_path = resolved_output / EXPORT_METADATA_FILENAME
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "base_model": BASE_MODEL_ID,
        "pinned_revision": BASE_REVISION,
        "adapter_path": str(resolved_adapter),
        "adapter_fingerprint_sha256": adapter_metadata["fingerprint_sha256"],
        "adapter": adapter_metadata,
        "source_git_commit": source.get("commit"),
        "source_git_dirty": source.get("dirty"),
        "dtype": "bfloat16",
        "merge_device": "cpu",
        "merge_method": "PeftModel.merge_and_unload(safe_merge=True)",
        "safe_serialization": True,
        "max_shard_size": max_shard_size,
        "output_path": str(resolved_output),
        "resource_preflight": resources.as_dict(),
        "package_versions": versions,
        "prompt_version": snapshot.version,
        "prompt_snapshot_sha256": snapshot.sha256,
        "generation_config": inference_config.generation_metadata(),
        "validation": None,
        "comparison": {"enabled": False},
    }
    _write_metadata(metadata_path, metadata)

    try:
        tokenizer = auto_tokenizer_cls.from_pretrained(
            BASE_MODEL_ID,
            revision=BASE_REVISION,
            trust_remote_code=False,
            cache_dir=cache_dir,
        )
        tokenizer_before = _tokenizer_invariants(tokenizer)
        if not isinstance(tokenizer_before["chat_template"], str) or not (
            tokenizer_before["chat_template"].strip()
        ):
            raise ExportValidationError("base tokenizer has no usable chat template")
        base_model = auto_model_cls.from_pretrained(
            BASE_MODEL_ID,
            revision=BASE_REVISION,
            torch_dtype=torch_module.bfloat16,
            trust_remote_code=False,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
        config_before = _config_invariants(base_model.config)
        peft_model = peft_model_cls.from_pretrained(
            base_model, str(resolved_adapter), is_trainable=False
        )
        peft_model.eval()
        merged_model = peft_model.merge_and_unload(safe_merge=True)
        merged_model.eval()
        merged_model.save_pretrained(
            resolved_output,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        generation_config = getattr(merged_model, "generation_config", None)
        if generation_config is None:
            raise ExportValidationError("merged model has no generation_config")
        generation_config.save_pretrained(resolved_output)
        tokenizer.save_pretrained(resolved_output)
        saved_files = _validate_saved_files(resolved_output)

        del peft_model, base_model, merged_model
        _release_cuda_cache(torch_module)

        reloaded_tokenizer = auto_tokenizer_cls.from_pretrained(
            resolved_output,
            trust_remote_code=False,
            local_files_only=True,
        )
        reloaded_model = auto_model_cls.from_pretrained(
            resolved_output,
            torch_dtype=torch_module.bfloat16,
            trust_remote_code=False,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        reloaded_model.eval()
        config_after = _config_invariants(reloaded_model.config)
        tokenizer_after = _tokenizer_invariants(reloaded_tokenizer)
        invariants = {
            "config_preserved": config_before == config_after,
            "tokenizer_preserved": tokenizer_before == tokenizer_after,
            "base_config": config_before,
            "reloaded_config": config_after,
            "base_tokenizer": tokenizer_before,
            "reloaded_tokenizer": tokenizer_after,
        }
        if not invariants["config_preserved"]:
            raise ExportValidationError("base architecture/rope/context config changed")
        if not invariants["tokenizer_preserved"]:
            raise ExportValidationError("tokenizer special tokens/chat template changed")

        smoke = run_strict_smoke_validation(
            samples,
            tokenizer=reloaded_tokenizer,
            model=reloaded_model,
            config=inference_config,
            torch_module=torch_module,
        )
        metadata["validation"] = {
            "local_tokenizer_reload": True,
            "local_model_reload": True,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "saved_files": saved_files,
            "invariants": invariants,
            "smoke": smoke,
        }

        del reloaded_model
        _release_cuda_cache(torch_module)

        if compare_quantized:
            if bitsandbytes_config_cls is None:
                from transformers import BitsAndBytesConfig

                bitsandbytes_config_cls = BitsAndBytesConfig
            quantized_model = _load_quantized_adapter_for_comparison(
                adapter_path=resolved_adapter,
                cache_dir=cache_dir,
                torch_module=torch_module,
                auto_model_cls=auto_model_cls,
                peft_model_cls=peft_model_cls,
                bitsandbytes_config_cls=bitsandbytes_config_cls,
            )
            quantized_smoke = run_strict_smoke_validation(
                samples,
                tokenizer=reloaded_tokenizer,
                model=quantized_model,
                config=inference_config,
                torch_module=torch_module,
            )
            metadata["comparison"] = compare_smoke_scores(smoke, quantized_smoke)
            metadata["comparison"]["base_4bit_lora_smoke"] = quantized_smoke
            del quantized_model
            _release_cuda_cache(torch_module)

        if not smoke["passed"]:
            raise ExportValidationError(
                "merged BF16 model failed strict JSON smoke validation"
            )
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        metadata["finished_at"] = utc_now()
        _write_metadata(metadata_path, metadata)

    return metadata
