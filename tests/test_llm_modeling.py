from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm.modeling import (
    ModelLoadError,
    inspect_adapter,
    inspect_model_placement,
    load_quantized_qwen,
    resolve_model_revision,
)


REVISION = "a" * 40


def write_adapter(
    root: Path,
    *,
    base_model: str = "owner/model",
    revision: str | None = REVISION,
    weights: bytes = b"adapter weights",
) -> Path:
    adapter_path = root / "adapter"
    adapter_path.mkdir()
    config = {
        "base_model_name_or_path": base_model,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": 16,
    }
    if revision is not None:
        config["revision"] = revision
    (adapter_path / "adapter_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (adapter_path / "adapter_model.safetensors").write_bytes(weights)
    return adapter_path


class FakeApi:
    def __init__(self, revision: str = REVISION) -> None:
        self.revision = revision
        self.repo_ids: list[str] = []

    def model_info(self, *, repo_id: str) -> object:
        self.repo_ids.append(repo_id)
        return types.SimpleNamespace(sha=self.revision)


class ModelingTests(unittest.TestCase):
    def test_revision_is_resolved_from_hugging_face_metadata(self) -> None:
        api = FakeApi()
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"HF_HOME": temporary}
        ):
            revision = resolve_model_revision("owner/model", api=api)
        self.assertEqual(revision, REVISION)
        self.assertEqual(api.repo_ids, ["owner/model"])

    def test_invalid_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"HF_HOME": temporary}
        ):
            with self.assertRaises(ModelLoadError):
                resolve_model_revision("owner/model", api=FakeApi("main"))

    def test_cpu_and_disk_offload_are_reported(self) -> None:
        model = types.SimpleNamespace(
            hf_device_map={"embed": 0, "layer": "cpu", "head": "disk"}
        )
        placement = inspect_model_placement(model)
        self.assertTrue(placement.has_cpu_or_disk_offload)
        self.assertEqual(placement.offloaded_modules, {"layer": "cpu", "head": "disk"})

    def test_loader_uses_same_revision_and_exact_4bit_options(self) -> None:
        calls: dict[str, dict] = {}

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.5.1+cu124"
        fake_torch.bfloat16 = object()
        fake_torch.cuda = FakeCuda()

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs: object) -> None:
                calls["quantization"] = kwargs

        class FakeTokenizerLoader:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
                calls["tokenizer"] = {"model_id": model_id, **kwargs}
                return object()

        class FakeModel:
            hf_device_map = {"": 0}

            def eval(self) -> None:
                calls["eval"] = {"called": True}

        class FakeModelLoader:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
                calls["model"] = {"model_id": model_id, **kwargs}
                return FakeModel()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizerLoader
        fake_transformers.AutoModelForCausalLM = FakeModelLoader
        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig

        versions = {
            "transformers": "4.55.4",
            "accelerate": "1.10.1",
            "bitsandbytes": "0.47.0",
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"HF_HOME": temporary}
        ), patch.dict(
            sys.modules, {"torch": fake_torch, "transformers": fake_transformers}
        ), patch(
            "src.llm.modeling.importlib.metadata.version",
            side_effect=lambda name: versions[name],
        ):
            loaded = load_quantized_qwen("owner/model", api=FakeApi())

        expected_cache = Path(temporary).resolve() / "hub"
        self.assertEqual(loaded.revision, REVISION)
        self.assertEqual(calls["tokenizer"]["revision"], REVISION)
        self.assertEqual(calls["model"]["revision"], REVISION)
        self.assertEqual(calls["tokenizer"]["cache_dir"], expected_cache)
        self.assertEqual(calls["model"]["cache_dir"], expected_cache)
        self.assertFalse(calls["tokenizer"]["trust_remote_code"])
        self.assertFalse(calls["model"]["trust_remote_code"])
        self.assertTrue(calls["model"]["low_cpu_mem_usage"])
        self.assertEqual(calls["model"]["device_map"], "auto")
        self.assertEqual(
            calls["quantization"],
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": fake_torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            },
        )
        self.assertTrue(calls["eval"]["called"])
        self.assertEqual(loaded.adapter_metadata, {"enabled": False})

    def test_adapter_metadata_and_fingerprint_are_stable_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter_path = write_adapter(Path(temporary))
            first = inspect_adapter(
                adapter_path,
                expected_model_id="owner/model",
                expected_revision=REVISION,
            )
            second = inspect_adapter(
                adapter_path,
                expected_model_id="owner/model",
                expected_revision=REVISION,
            )
            expected_weight_hash = hashlib.sha256(b"adapter weights").hexdigest()

            self.assertTrue(first["enabled"])
            self.assertEqual(first["peft_type"], "LORA")
            self.assertEqual(first["task_type"], "CAUSAL_LM")
            self.assertEqual(first["base_model_name_or_path"], "owner/model")
            self.assertFalse(first["is_trainable"])
            self.assertTrue(first["loaded_for_inference"])
            self.assertTrue(first["inference_mode"])
            self.assertFalse(first["merged"])
            self.assertEqual(first["revision_source"], "adapter_config")
            self.assertEqual(
                first["files"]["adapter_model.safetensors"]["sha256"],
                expected_weight_hash,
            )
            self.assertEqual(first["fingerprint_sha256"], second["fingerprint_sha256"])

            (adapter_path / "adapter_model.safetensors").write_bytes(b"changed")
            changed = inspect_adapter(
                adapter_path,
                expected_model_id="owner/model",
                expected_revision=REVISION,
            )
            self.assertNotEqual(
                first["fingerprint_sha256"], changed["fingerprint_sha256"]
            )

    def test_incompatible_adapter_base_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter_path = write_adapter(
                Path(temporary), base_model="different/model"
            )
            with self.assertRaisesRegex(ModelLoadError, "base model is incompatible"):
                inspect_adapter(
                    adapter_path,
                    expected_model_id="owner/model",
                    expected_revision=REVISION,
                )

    def test_missing_adapter_revision_is_allowed_and_metadata_can_supply_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter_path = write_adapter(root, revision=None)
            without_metadata = inspect_adapter(
                adapter_path,
                expected_model_id="owner/model",
                expected_revision=REVISION,
            )
            self.assertIsNone(without_metadata["revision"])
            self.assertIsNone(without_metadata["revision_source"])

            (root / "run_metadata.json").write_text(
                json.dumps({"model_id": "owner/model", "revision": REVISION}),
                encoding="utf-8",
            )
            with_metadata = inspect_adapter(
                adapter_path,
                expected_model_id="owner/model",
                expected_revision=REVISION,
            )
            self.assertEqual(with_metadata["revision"], REVISION)
            self.assertEqual(with_metadata["revision_source"], "training_metadata")
            self.assertEqual(
                with_metadata["training_metadata_path"],
                str((root / "run_metadata.json").resolve()),
            )

    def test_incompatible_adapter_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter_path = write_adapter(Path(temporary), revision="b" * 40)
            with self.assertRaisesRegex(ModelLoadError, "revision is incompatible"):
                inspect_adapter(
                    adapter_path,
                    expected_model_id="owner/model",
                    expected_revision=REVISION,
                )

    def test_adapter_loader_wraps_without_merge_and_sets_eval_mode(self) -> None:
        calls: dict[str, object] = {"merge": False}

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.5.1+cu124"
        fake_torch.bfloat16 = object()
        fake_torch.cuda = FakeCuda()

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs: object) -> None:
                calls["quantization"] = kwargs

        class FakeTokenizerLoader:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
                calls["tokenizer"] = {"model_id": model_id, **kwargs}
                return object()

        class FakeBaseModel:
            hf_device_map = {"": 0}

            def merge_and_unload(self) -> None:
                calls["merge"] = True

        base_model = FakeBaseModel()

        class FakeModelLoader:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeBaseModel:
                calls["model"] = {"model_id": model_id, **kwargs}
                return base_model

        class FakePeftWrapper:
            hf_device_map = {"": 0}

            def eval(self) -> None:
                calls["adapter_eval"] = True

        wrapper = FakePeftWrapper()

        class FakePeftModel:
            @classmethod
            def from_pretrained(
                cls, model: object, adapter_path: str
            ) -> FakePeftWrapper:
                calls["peft"] = {"model": model, "adapter_path": adapter_path}
                return wrapper

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizerLoader
        fake_transformers.AutoModelForCausalLM = FakeModelLoader
        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = FakePeftModel
        versions = {
            "transformers": "4.55.4",
            "accelerate": "1.10.1",
            "bitsandbytes": "0.47.0",
            "peft": "0.16.0",
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter_path = write_adapter(root)
            with patch.dict(os.environ, {"HF_HOME": temporary}), patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "transformers": fake_transformers,
                    "peft": fake_peft,
                },
            ), patch(
                "src.llm.modeling.importlib.metadata.version",
                side_effect=lambda name: versions[name],
            ):
                loaded = load_quantized_qwen(
                    "owner/model", adapter_path=adapter_path, api=FakeApi()
                )

        self.assertIs(calls["peft"]["model"], base_model)
        self.assertEqual(calls["peft"]["adapter_path"], str(adapter_path.resolve()))
        self.assertEqual(calls["tokenizer"]["revision"], REVISION)
        self.assertEqual(calls["model"]["revision"], REVISION)
        self.assertFalse(calls["merge"])
        self.assertTrue(calls["adapter_eval"])
        self.assertIs(loaded.model, wrapper)
        self.assertEqual(loaded.revision, REVISION)
        self.assertTrue(loaded.adapter_metadata["enabled"])
        self.assertEqual(loaded.runtime_versions["peft"], "0.16.0")


if __name__ == "__main__":
    unittest.main()
