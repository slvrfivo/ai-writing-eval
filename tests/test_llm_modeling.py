from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm.modeling import (
    ModelLoadError,
    inspect_model_placement,
    load_quantized_qwen,
    resolve_model_revision,
)


REVISION = "a" * 40


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


if __name__ == "__main__":
    unittest.main()
