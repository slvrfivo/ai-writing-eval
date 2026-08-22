"""Single-sample zero-shot inference with durable JSONL outputs."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO

from .modeling import cuda_memory_report
from .parsing import parse_model_output
from .prompting import build_messages, load_prompt_snapshot


RAW_GENERATIONS_FILENAME = "raw_generations.jsonl"
PREDICTIONS_FILENAME = "predictions.jsonl"
FAILURES_FILENAME = "failures.jsonl"
RUN_METADATA_FILENAME = "run_metadata.json"


class PipelineInputError(ValueError):
    """Raised when input or configuration cannot satisfy the pipeline contract."""


@dataclass(frozen=True)
class InferenceConfig:
    model_id: str
    quantization: str
    compute_dtype: str
    double_quant: bool
    batch_size: int
    max_new_tokens: int
    do_sample: bool
    seed: int
    prompt_version: str

    @classmethod
    def from_json(cls, path: Path) -> "InferenceConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PipelineInputError("inference config must be a JSON object")
        try:
            config = cls(**payload)
        except TypeError as exc:
            raise PipelineInputError(f"invalid inference config fields: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        expected = {
            "quantization": (self.quantization, "nf4"),
            "compute_dtype": (self.compute_dtype, "bfloat16"),
            "double_quant": (self.double_quant, True),
            "batch_size": (self.batch_size, 1),
            "max_new_tokens": (self.max_new_tokens, 512),
            "do_sample": (self.do_sample, False),
        }
        mismatches = {
            name: actual
            for name, (actual, required) in expected.items()
            if actual != required
        }
        if mismatches:
            raise PipelineInputError(
                f"config does not match the first zero-shot baseline: {mismatches}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PipelineInputError("seed must be an integer")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise PipelineInputError("model_id must be a non-empty string")
        if not isinstance(self.prompt_version, str) or not self.prompt_version:
            raise PipelineInputError("prompt_version must be a non-empty string")

    def quantization_metadata(self) -> dict[str, Any]:
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": self.quantization,
            "bnb_4bit_compute_dtype": self.compute_dtype,
            "bnb_4bit_use_double_quant": self.double_quant,
        }

    def generation_metadata(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "termination": "model/tokenizer EOS only",
        }


@dataclass(frozen=True)
class InferenceSample:
    sample_id: str
    prediction_id: str
    document_id: str | None
    prompt: str
    essay: str

    def identity_fields(self) -> dict[str, str]:
        fields = {"id": self.sample_id}
        if self.document_id is not None:
            fields["document_id"] = self.document_id
        return fields


@dataclass(frozen=True)
class PipelineResult:
    attempted_count: int
    success_count: int
    failure_count: int
    skipped_count: int
    metadata_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sample_from_record(record: Mapping[str, Any], location: str) -> InferenceSample:
    """Project an input record onto identifiers, prompt, and essay only."""
    if not isinstance(record, Mapping):
        raise PipelineInputError(f"{location}: record must be a JSON object")

    identifier = record.get("id")
    document_id = record.get("document_id")
    essay_id = record.get("essay_id")
    if identifier is not None and (not isinstance(identifier, str) or not identifier):
        raise PipelineInputError(f"{location}.id: must be a non-empty string")
    if document_id is not None and (
        not isinstance(document_id, str) or not document_id
    ):
        raise PipelineInputError(
            f"{location}.document_id: must be a non-empty string"
        )
    if essay_id is not None and (not isinstance(essay_id, str) or not essay_id):
        raise PipelineInputError(f"{location}.essay_id: must be a non-empty string")
    if identifier is None and document_id is None and essay_id is None:
        raise PipelineInputError(f"{location}: id, document_id, or essay_id is required")

    prompt = record.get("prompt")
    essay = record.get("essay")
    if not isinstance(prompt, str) or not prompt:
        raise PipelineInputError(f"{location}.prompt: must be a non-empty string")
    if not isinstance(essay, str) or not essay:
        raise PipelineInputError(f"{location}.essay: must be a non-empty string")

    sample_id = identifier or document_id or essay_id
    assert isinstance(sample_id, str)
    prediction_id = document_id or essay_id or sample_id
    return InferenceSample(
        sample_id=sample_id,
        prediction_id=prediction_id,
        document_id=document_id,
        prompt=prompt,
        essay=essay,
    )


def iter_input_samples(path: Path) -> Iterator[InferenceSample]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineInputError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            yield sample_from_record(payload, f"{path}:{line_number}")


def completed_prediction_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineInputError(
                    f"{path}:{line_number}: invalid existing prediction JSON"
                ) from exc
            identifier = record.get("essay_id") if isinstance(record, dict) else None
            if not isinstance(identifier, str) or not identifier:
                raise PipelineInputError(
                    f"{path}:{line_number}: existing prediction needs essay_id"
                )
            judge = record.get("judge")
            try:
                serialized_judge = json.dumps(
                    judge, ensure_ascii=False, allow_nan=False
                )
            except (TypeError, ValueError) as exc:
                raise PipelineInputError(
                    f"{path}:{line_number}: existing judge is not valid JSON"
                ) from exc
            parsed = parse_model_output(serialized_judge)
            if not parsed.ok:
                assert parsed.failure is not None
                raise PipelineInputError(
                    f"{path}:{line_number}: existing judge is not a completed "
                    f"prediction ({parsed.failure.code})"
                )
            if identifier in completed:
                raise PipelineInputError(
                    f"{path}:{line_number}: duplicate essay_id '{identifier}'"
                )
            completed.add(identifier)
    return completed


def _model_input_device(model: Any) -> Any | None:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, TypeError):
        return None


def _move_model_inputs(model_inputs: Any, device: Any | None) -> Any:
    if device is None:
        return model_inputs
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    if isinstance(model_inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
    raise PipelineInputError("tokenizer output cannot be moved to the model device")


def _input_token_count(input_ids: Any) -> int:
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(input_ids[0])


def generate_one(
    sample: InferenceSample,
    *,
    tokenizer: Any,
    model: Any,
    config: InferenceConfig,
    torch_module: Any,
) -> str:
    """Generate one response using only the official prompt and normal EOS."""
    messages = build_messages(
        sample.prompt,
        sample.essay,
        version=config.prompt_version,
    )
    rendered_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(rendered_chat, return_tensors="pt")
    model_inputs = _move_model_inputs(model_inputs, _model_input_device(model))
    input_token_count = _input_token_count(model_inputs["input_ids"])

    inference_mode = getattr(torch_module, "inference_mode", None)
    context = inference_mode() if inference_mode is not None else nullcontext()
    with context:
        generated = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=config.max_new_tokens,
        )

    new_token_ids = generated[0][input_token_count:]
    if hasattr(new_token_ids, "tolist"):
        new_token_ids = new_token_ids.tolist()
    return tokenizer.decode(new_token_ids, skip_special_tokens=True)


def _write_jsonl(stream: TextIO, record: Mapping[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_resume_metadata(
    *,
    metadata_path: Path,
    artifact_paths: tuple[Path, ...],
    expected: Mapping[str, Any],
) -> None:
    has_artifacts = any(path.exists() and path.stat().st_size > 0 for path in artifact_paths)
    if not metadata_path.exists():
        if has_artifacts:
            raise PipelineInputError(
                "cannot resume existing outputs without run_metadata.json"
            )
        return

    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineInputError("existing run_metadata.json is invalid") from exc
    if not isinstance(previous, dict):
        raise PipelineInputError("existing run_metadata.json must be a JSON object")

    mismatches = {
        key: {"previous": previous.get(key), "current": current}
        for key, current in expected.items()
        if previous.get(key) != current
    }
    if mismatches:
        raise PipelineInputError(
            "resume metadata does not match the current inference run: "
            + json.dumps(mismatches, ensure_ascii=False, allow_nan=False)
        )


def _seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    if hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available() and hasattr(cuda, "manual_seed_all"):
        cuda.manual_seed_all(seed)


def _reset_cuda_peak_memory(torch_module: Any) -> None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.reset_peak_memory_stats()


def _metadata_with_memory(
    metadata: dict[str, Any], memory: Mapping[str, Any]
) -> None:
    metadata.update(
        {
            "cuda_device": memory.get("device"),
            "total_vram_bytes": memory.get("total_vram_bytes", 0),
            "total_vram_gib": memory.get("total_vram_gib", 0.0),
            "peak_allocated_vram_bytes": memory.get("peak_allocated_bytes", 0),
            "peak_allocated_vram_gib": round(
                int(memory.get("peak_allocated_bytes", 0)) / 1024**3, 3
            ),
            "peak_reserved_vram_bytes": memory.get("peak_reserved_bytes", 0),
            "peak_reserved_vram_gib": round(
                int(memory.get("peak_reserved_bytes", 0)) / 1024**3, 3
            ),
        }
    )


def run_inference_pipeline(
    *,
    input_path: Path,
    output_dir: Path,
    tokenizer: Any,
    model: Any,
    model_revision: str,
    runtime_versions: Mapping[str, str],
    config: InferenceConfig,
    limit: int | None = None,
    torch_module: Any | None = None,
    cuda_memory_before_load: Mapping[str, Any] | None = None,
    cuda_memory_after_load: Mapping[str, Any] | None = None,
) -> PipelineResult:
    """Run durable batch-size-one inference, resuming successful predictions."""
    if limit is not None and (isinstance(limit, bool) or limit <= 0):
        raise PipelineInputError("limit must be a positive integer")
    config.validate()
    snapshot = load_prompt_snapshot(config.prompt_version)

    if torch_module is None:
        import torch as torch_module

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / RAW_GENERATIONS_FILENAME
    predictions_path = output_dir / PREDICTIONS_FILENAME
    failures_path = output_dir / FAILURES_FILENAME
    metadata_path = output_dir / RUN_METADATA_FILENAME

    resume_contract = {
        "model_id": config.model_id,
        "model_revision": model_revision,
        "versions": dict(runtime_versions),
        "prompt_version": snapshot.version,
        "prompt_snapshot_sha256": snapshot.sha256,
        "quantization_config": config.quantization_metadata(),
        "generation_config": config.generation_metadata(),
        "input_path": str(input_path.resolve()),
    }
    _validate_resume_metadata(
        metadata_path=metadata_path,
        artifact_paths=(raw_path, predictions_path, failures_path),
        expected=resume_contract,
    )

    completed_ids = completed_prediction_ids(predictions_path)
    existing_success_count = len(completed_ids)
    _seed_everything(config.seed, torch_module)
    _reset_cuda_peak_memory(torch_module)
    initial_memory = cuda_memory_report(torch_module)

    metadata: dict[str, Any] = {
        **resume_contract,
        "transformers_version": runtime_versions.get("transformers"),
        "accelerate_version": runtime_versions.get("accelerate"),
        "torch_version": runtime_versions.get("torch"),
        "bitsandbytes_version": runtime_versions.get("bitsandbytes"),
        "output_dir": str(output_dir.resolve()),
        "limit": limit,
        "started_at": utc_now(),
        "finished_at": None,
        "status": "running",
        "existing_success_count": existing_success_count,
        "attempted_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "skipped_count": 0,
        "cuda_memory_at_inference_start": initial_memory,
    }
    if cuda_memory_before_load is not None:
        metadata["cuda_memory_before_model_load"] = dict(cuda_memory_before_load)
    if cuda_memory_after_load is not None:
        metadata["cuda_memory_after_model_load"] = dict(cuda_memory_after_load)
    _metadata_with_memory(metadata, initial_memory)
    _write_metadata(metadata_path, metadata)

    attempted_count = 0
    success_count = 0
    failure_count = 0
    skipped_count = 0
    pipeline_error: BaseException | None = None

    try:
        with (
            raw_path.open("a", encoding="utf-8") as raw_stream,
            predictions_path.open("a", encoding="utf-8") as prediction_stream,
            failures_path.open("a", encoding="utf-8") as failure_stream,
        ):
            for sample in iter_input_samples(input_path):
                if sample.prediction_id in completed_ids:
                    skipped_count += 1
                    continue
                if limit is not None and attempted_count >= limit:
                    break

                attempted_count += 1
                raw_output = generate_one(
                    sample,
                    tokenizer=tokenizer,
                    model=model,
                    config=config,
                    torch_module=torch_module,
                )
                raw_record = {**sample.identity_fields(), "raw_output": raw_output}
                _write_jsonl(raw_stream, raw_record)

                parsed = parse_model_output(raw_output)
                if parsed.ok:
                    assert parsed.value is not None
                    prediction = {
                        "essay_id": sample.prediction_id,
                        "judge": parsed.value,
                    }
                    _write_jsonl(prediction_stream, prediction)
                    completed_ids.add(sample.prediction_id)
                    success_count += 1
                else:
                    assert parsed.failure is not None
                    failure = {
                        **sample.identity_fields(),
                        "raw_output": raw_output,
                        "error": asdict(parsed.failure),
                    }
                    _write_jsonl(failure_stream, failure)
                    failure_count += 1
    except BaseException as exc:
        pipeline_error = exc
        raise
    finally:
        try:
            final_memory = cuda_memory_report(torch_module)
        except Exception as memory_error:
            final_memory = initial_memory
            metadata["cuda_memory_report_error"] = {
                "type": type(memory_error).__name__,
                "message": str(memory_error),
            }
        metadata.update(
            {
                "finished_at": utc_now(),
                "status": "failed" if pipeline_error is not None else "completed",
                "attempted_count": attempted_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "skipped_count": skipped_count,
                "completed_success_count": len(completed_ids),
                "cuda_memory_at_inference_end": final_memory,
            }
        )
        if pipeline_error is not None:
            metadata["error"] = {
                "type": type(pipeline_error).__name__,
                "message": str(pipeline_error),
            }
        _metadata_with_memory(metadata, final_memory)
        _write_metadata(metadata_path, metadata)

    return PipelineResult(
        attempted_count=attempted_count,
        success_count=success_count,
        failure_count=failure_count,
        skipped_count=skipped_count,
        metadata_path=metadata_path,
    )
