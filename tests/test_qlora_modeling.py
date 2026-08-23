from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.qlora.config import QLoRAConfig
from src.qlora.modeling import load_qlora_model, trainable_parameter_stats


def project_config() -> QLoRAConfig:
    return QLoRAConfig.from_json(
        Path(__file__).resolve().parents[1] / "configs" / "qwen3_4b_qlora_v1.json"
    )


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeParameter:
    def __init__(self, count: int, requires_grad: bool) -> None:
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.count


class QLoRAModelingTests(unittest.TestCase):
    def test_config_preserves_unselected_max_length_and_v1_weights(self) -> None:
        config = project_config()
        self.assertIsNone(config.max_seq_length)
        self.assertEqual(
            config.loss_weights,
            {"prompt": 0.0, "structure": 1.0, "score": 10.0, "rationale": 0.1},
        )
        self.assertEqual(config.training["learning_rate"], 5e-5)

    def test_loader_uses_pinned_revision_exact_qlora_options_and_no_merge(self) -> None:
        calls: dict[str, object] = {}
        config = project_config()

        fake_torch = types.ModuleType("torch")
        fake_torch.__version__ = "2.5.1+cu124"
        fake_torch.bfloat16 = object()
        fake_torch.cuda = FakeCuda()

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs: object) -> None:
                calls["quantization"] = kwargs

        class FakeModel:
            hf_device_map = {"": 0}

            def __init__(self) -> None:
                self.config = types.SimpleNamespace(use_cache=True)

            def train(self) -> None:
                calls["train"] = True

        model = FakeModel()

        class FakeModelLoader:
            @classmethod
            def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
                calls["model"] = {"model_id": model_id, **kwargs}
                return model

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = FakeModelLoader
        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig

        def fake_prepare(value: object, **kwargs: object) -> object:
            calls["prepare"] = kwargs
            return value

        class FakeLoraConfig:
            def __init__(self, **kwargs: object) -> None:
                calls["lora"] = kwargs

        def fake_get_peft_model(
            value: object, lora_config: object, **kwargs: object
        ) -> object:
            calls["get_peft_model"] = lora_config
            calls["get_peft_model_kwargs"] = kwargs
            return value

        fake_peft = types.ModuleType("peft")
        fake_peft.LoraConfig = FakeLoraConfig
        fake_peft.prepare_model_for_kbit_training = fake_prepare
        fake_peft.get_peft_model = fake_get_peft_model

        versions = {
            "transformers": "4.55.4",
            "accelerate": "1.10.1",
            "bitsandbytes": "0.47.0",
            "peft": "0.16.0",
        }
        tokenizer = object()
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"HF_HOME": temporary}
        ), patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers, "peft": fake_peft},
        ), patch(
            "src.qlora.modeling.importlib.metadata.version",
            side_effect=lambda name: versions[name],
        ):
            loaded = load_qlora_model(config, tokenizer=tokenizer)

        model_call = calls["model"]
        self.assertEqual(model_call["revision"], config.revision)
        self.assertFalse(model_call["trust_remote_code"])
        self.assertTrue(model_call["low_cpu_mem_usage"])
        self.assertEqual(model_call["device_map"], "auto")
        self.assertEqual(
            calls["quantization"],
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": fake_torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            },
        )
        self.assertEqual(calls["prepare"], {"use_gradient_checkpointing": True})
        self.assertEqual(
            calls["lora"],
            {
                "target_modules": "all-linear",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        )
        self.assertEqual(
            calls["get_peft_model_kwargs"], {"revision": config.revision}
        )
        self.assertIs(loaded.tokenizer, tokenizer)
        self.assertFalse(model.config.use_cache)
        self.assertTrue(calls["train"])

    def test_trainable_parameter_metadata(self) -> None:
        model = types.SimpleNamespace(
            parameters=lambda: iter(
                [FakeParameter(10, False), FakeParameter(2, True)]
            )
        )
        stats = trainable_parameter_stats(model)
        self.assertEqual(stats["trainable_params"], 2)
        self.assertEqual(stats["total_params"], 12)
        self.assertAlmostEqual(stats["trainable_ratio"], 1 / 6)


if __name__ == "__main__":
    unittest.main()
