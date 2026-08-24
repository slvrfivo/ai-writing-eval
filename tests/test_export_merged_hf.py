from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from src.export_merged_hf import parse_args
from src.llm.exporting import (
    BASE_MODEL_ID,
    BASE_REVISION,
    DEFAULT_ADAPTER_PATH,
    DEFAULT_OUTPUT_PATH,
    ExportError,
    ResourceReport,
    ResourceSafetyError,
    compare_smoke_scores,
    export_merged_model,
    validate_output_path,
)
from src.llm.pipeline import InferenceConfig


VALID_OUTPUT = json.dumps(
    {
        "content": {"score": 3, "rationale": "content evidence"},
        "organization": {"score": 4, "rationale": "organization evidence"},
        "expression": {"score": 2, "rationale": "expression evidence"},
    }
)


def inference_config() -> InferenceConfig:
    return InferenceConfig(
        model_id=BASE_MODEL_ID,
        quantization="nf4",
        compute_dtype="bfloat16",
        double_quant=True,
        batch_size=1,
        max_new_tokens=1024,
        do_sample=False,
        seed=42,
        prompt_version="writing_scoring_2026-07-20",
    )


def write_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": BASE_MODEL_ID,
                "revision": BASE_REVISION,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "inference_mode": True,
            }
        ),
        encoding="utf-8",
    )
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


def write_validation(path: Path) -> None:
    records = [
        {
            "id": f"sample-{index}",
            "document_id": f"doc-{index}",
            "prompt": f"prompt {index}",
            "essay": f"essay {index}",
            "score": {"content": 1, "organization": 5, "expression": 1},
        }
        for index in range(3)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def safe_resources() -> ResourceReport:
    return ResourceReport(
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=24 * 1024**3,
        output_disk_free_bytes=100 * 1024**3,
        output_disk_total_bytes=200 * 1024**3,
    )


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    __version__ = "2.5.1+cu124"
    bfloat16 = object()
    cuda = FakeCuda()

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    @staticmethod
    def inference_mode() -> nullcontext:
        return nullcontext()


class FakeConfig:
    architectures = ["Qwen3ForCausalLM"]
    model_type = "qwen3"
    max_position_embeddings = 32768
    rope_scaling = None
    rope_theta = 1000000.0
    sliding_window = None
    use_sliding_window = False
    vocab_size = 151936


class FakeGenerationConfig:
    def __init__(self, calls: dict[str, object]) -> None:
        self.calls = calls

    def save_pretrained(self, output: Path) -> None:
        self.calls["generation_config_saved"] = str(output)
        (output / "generation_config.json").write_text("{}", encoding="utf-8")


class FakeTokenizer:
    chat_template = "{{ messages }}"
    special_tokens_map = {"eos_token": "<|im_end|>"}
    all_special_ids = [151643, 151645]
    bos_token_id = None
    eos_token_id = 151645
    pad_token_id = 151643

    def __init__(self, calls: dict[str, object]) -> None:
        self.calls = calls

    def save_pretrained(self, output: Path) -> None:
        self.calls["tokenizer_saved"] = str(output)
        (output / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (output / "tokenizer.json").write_text("{}", encoding="utf-8")

    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
        self.calls.setdefault("chat_calls", []).append((messages, kwargs))
        return "rendered official chat"

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[list[int]]]:
        self.calls.setdefault("tokenizer_calls", []).append((text, kwargs))
        return {"input_ids": [[11, 12]], "attention_mask": [[1, 1]]}

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        self.calls.setdefault("decode_calls", []).append((token_ids, kwargs))
        return VALID_OUTPUT


class FakeTokenizerLoader:
    calls: dict[str, object]

    @classmethod
    def from_pretrained(cls, source: object, **kwargs: object) -> FakeTokenizer:
        cls.calls.setdefault("tokenizer_loads", []).append((source, kwargs))
        return FakeTokenizer(cls.calls)


class FakeModel:
    device = "cpu"

    def __init__(self, calls: dict[str, object], kind: str) -> None:
        self.calls = calls
        self.kind = kind
        self.config = FakeConfig()
        self.generation_config = FakeGenerationConfig(calls)

    def eval(self) -> None:
        self.calls.setdefault("eval", []).append(self.kind)

    def save_pretrained(self, output: Path, **kwargs: object) -> None:
        self.calls["model_save"] = {"output": str(output), **kwargs}
        (output / "config.json").write_text("{}", encoding="utf-8")
        (output / "model-00001-of-00002.safetensors").write_bytes(b"weights")

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.calls.setdefault("generate", []).append(kwargs)
        return [[11, 12, 101, 102]]


class FakeModelLoader:
    calls: dict[str, object]

    @classmethod
    def from_pretrained(cls, source: object, **kwargs: object) -> FakeModel:
        cls.calls.setdefault("model_loads", []).append((source, kwargs))
        kind = "base" if source == BASE_MODEL_ID else "reloaded"
        return FakeModel(cls.calls, kind)


class FakePeftWrapper:
    def __init__(self, base: FakeModel, calls: dict[str, object]) -> None:
        self.base = base
        self.calls = calls

    def eval(self) -> None:
        self.calls["peft_eval"] = True

    def merge_and_unload(self, **kwargs: object) -> FakeModel:
        self.calls["merge"] = kwargs
        return FakeModel(self.calls, "merged")


class FakePeftModel:
    calls: dict[str, object]

    @classmethod
    def from_pretrained(
        cls, base: FakeModel, adapter_path: str, **kwargs: object
    ) -> FakePeftWrapper:
        cls.calls["peft_load"] = {
            "base": base,
            "adapter_path": adapter_path,
            **kwargs,
        }
        return FakePeftWrapper(base, cls.calls)


class ExportMergedTests(unittest.TestCase):
    def test_cli_uses_submission_defaults_and_three_smoke_samples(self) -> None:
        args = parse_args(["--validation-input", "validation.jsonl"])
        self.assertEqual(args.adapter, DEFAULT_ADAPTER_PATH)
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_PATH)
        self.assertEqual(args.smoke_samples, 3)
        self.assertFalse(args.compare_quantized)

    def test_output_must_stay_under_submission_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ExportError, "must stay under"):
                validate_output_path(root / "outside", allowed_root=root / "allowed")

    def test_low_resources_fail_before_any_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation.jsonl"
            write_validation(validation)
            unsafe = ResourceReport(
                total_ram_bytes=9 * 1024**3,
                available_ram_bytes=8 * 1024**3,
                output_disk_free_bytes=10 * 1024**3,
                output_disk_total_bytes=20 * 1024**3,
            )
            with self.assertRaisesRegex(ResourceSafetyError, "preflight failed"):
                export_merged_model(
                    adapter_path=root / "adapter",
                    output_path=root / "submissions" / "model",
                    validation_input=validation,
                    inference_config=inference_config(),
                    project_root=root,
                    allowed_output_root=root / "submissions",
                    resource_report=unsafe,
                    adapter_inspector=lambda *args, **kwargs: {
                        "fingerprint_sha256": "f" * 64
                    },
                )

    def test_load_merge_save_reload_and_strict_smoke_flow(self) -> None:
        calls: dict[str, object] = {}
        FakeTokenizerLoader.calls = calls
        FakeModelLoader.calls = calls
        FakePeftModel.calls = calls
        fake_torch = FakeTorch()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            output = root / "submissions" / "merged"
            validation = root / "validation.jsonl"
            write_adapter(adapter)
            write_validation(validation)
            with patch.dict(os.environ, {"HF_HOME": str(root / "hf-cache")}):
                metadata = export_merged_model(
                    adapter_path=adapter,
                    output_path=output,
                    validation_input=validation,
                    inference_config=inference_config(),
                    project_root=root,
                    allowed_output_root=root / "submissions",
                    resource_report=safe_resources(),
                    git_state={"commit": "a" * 40, "dirty": False},
                    runtime_versions={
                        "transformers": "4.55.4",
                        "accelerate": "1.10.1",
                        "peft": "0.16.0",
                        "torch": "2.5.1+cu124",
                    },
                    torch_module=fake_torch,
                    auto_tokenizer_cls=FakeTokenizerLoader,
                    auto_model_cls=FakeModelLoader,
                    peft_model_cls=FakePeftModel,
                )

            tokenizer_loads = calls["tokenizer_loads"]
            model_loads = calls["model_loads"]
            self.assertEqual(tokenizer_loads[0][0], BASE_MODEL_ID)
            self.assertEqual(tokenizer_loads[0][1]["revision"], BASE_REVISION)
            self.assertFalse(tokenizer_loads[0][1]["trust_remote_code"])
            self.assertEqual(model_loads[0][0], BASE_MODEL_ID)
            self.assertEqual(model_loads[0][1]["revision"], BASE_REVISION)
            self.assertIs(model_loads[0][1]["torch_dtype"], fake_torch.bfloat16)
            self.assertEqual(model_loads[0][1]["device_map"], {"": "cpu"})
            self.assertFalse(model_loads[0][1]["trust_remote_code"])
            self.assertEqual(calls["peft_load"]["adapter_path"], str(adapter.resolve()))
            self.assertFalse(calls["peft_load"]["is_trainable"])
            self.assertEqual(calls["merge"], {"safe_merge": True})
            self.assertEqual(
                calls["model_save"],
                {
                    "output": str(output.resolve()),
                    "safe_serialization": True,
                    "max_shard_size": "4GB",
                },
            )
            self.assertEqual(model_loads[1][0], output.resolve())
            self.assertIs(model_loads[1][1]["torch_dtype"], fake_torch.bfloat16)
            self.assertFalse(model_loads[1][1]["trust_remote_code"])
            self.assertTrue(model_loads[1][1]["local_files_only"])
            self.assertEqual(len(calls["generate"]), 3)
            self.assertEqual(len(calls["chat_calls"]), 3)
            self.assertTrue(metadata["validation"]["smoke"]["passed"])
            self.assertTrue(metadata["validation"]["invariants"]["config_preserved"])
            self.assertTrue(
                metadata["validation"]["invariants"]["tokenizer_preserved"]
            )
            self.assertEqual(metadata["adapter_fingerprint_sha256"], metadata["adapter"]["fingerprint_sha256"])
            self.assertEqual(metadata["source_git_commit"], "a" * 40)
            self.assertEqual(metadata["merge_device"], "cpu")
            self.assertFalse((output / "adapter_config.json").exists())
            self.assertFalse((output / "adapter_model.safetensors").exists())
            stored = json.loads(
                (output / "export_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored["status"], "completed")

    def test_score_comparison_reports_dimension_deltas(self) -> None:
        merged = {
            "samples": [
                {"id": "a", "scores": {"content": 4, "organization": 3, "expression": 2}}
            ]
        }
        quantized = {
            "samples": [
                {"id": "a", "scores": {"content": 3, "organization": 3, "expression": 4}}
            ]
        }
        report = compare_smoke_scores(merged, quantized)
        self.assertEqual(
            report["samples"][0]["score_delta_merged_minus_4bit_lora"],
            {"content": 1, "organization": 0, "expression": -2},
        )
        self.assertFalse(report["samples"][0]["exact_score_match"])


if __name__ == "__main__":
    unittest.main()
