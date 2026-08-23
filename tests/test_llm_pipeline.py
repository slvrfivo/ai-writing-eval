from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

from src.llm.pipeline import InferenceConfig, run_inference_pipeline


VALID_OUTPUT = json.dumps(
    {
        "content": {"score": 3, "rationale": "내용 근거"},
        "organization": {"score": 4, "rationale": "구성 근거"},
        "expression": {"score": 2, "rationale": "표현 근거"},
    },
    ensure_ascii=False,
)


class FakeCuda:
    def __init__(self) -> None:
        self.reset_count = 0

    @staticmethod
    def is_available() -> bool:
        return False

    def reset_peak_memory_stats(self) -> None:
        self.reset_count += 1


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    @staticmethod
    def inference_mode() -> nullcontext:
        return nullcontext()


class FakeTokenizer:
    def __init__(self, decoded_outputs: list[str]) -> None:
        self.decoded_outputs = list(decoded_outputs)
        self.chat_calls: list[tuple[list[dict[str, str]], dict]] = []
        self.tokenize_calls: list[tuple[str, dict]] = []
        self.decode_calls: list[tuple[list[int], dict]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> str:
        self.chat_calls.append((messages, kwargs))
        return "rendered chat"

    def __call__(self, text: str, **kwargs: object) -> dict:
        self.tokenize_calls.append((text, kwargs))
        return {"input_ids": [[11, 12]], "attention_mask": [[1, 1]]}

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        self.decode_calls.append((list(token_ids), kwargs))
        return self.decoded_outputs.pop(0)


class FakeModel:
    device = "cuda:0"

    def __init__(self, generation_count: int, fail_at: int | None = None) -> None:
        self.outputs = [[11, 12, 101, 102] for _ in range(generation_count)]
        self.generate_calls: list[dict] = []
        self.fail_at = fail_at

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_calls.append(kwargs)
        if self.fail_at is not None and len(self.generate_calls) == self.fail_at:
            raise RuntimeError("simulated CUDA OOM")
        return [self.outputs.pop(0)]


def config() -> InferenceConfig:
    return InferenceConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        quantization="nf4",
        compute_dtype="bfloat16",
        double_quant=True,
        batch_size=1,
        max_new_tokens=1024,
        do_sample=False,
        seed=42,
        prompt_version="writing_scoring_2026-07-20",
    )


def record(index: int, *, score_marker: str | None = None) -> dict:
    payload = {
        "id": f"id-{index}",
        "document_id": f"doc-{index}",
        "prompt": f"주제 {index}",
        "essay": f"본문 {index}",
    }
    if score_marker is not None:
        payload["score"] = {
            "content": score_marker,
            "organization": score_marker,
            "expression": score_marker,
        }
    return payload


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PipelineTests(unittest.TestCase):
    def run_fake(
        self,
        root: Path,
        records: list[dict],
        outputs: list[str],
        *,
        limit: int | None = None,
        model: FakeModel | None = None,
        adapter_metadata: dict | None = None,
    ) -> tuple[object, FakeTokenizer, FakeModel]:
        input_path = root / "input.jsonl"
        output_dir = root / "outputs"
        write_jsonl(input_path, records)
        tokenizer = FakeTokenizer(outputs)
        fake_model = model or FakeModel(len(outputs))
        result = run_inference_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=fake_model,
            model_revision="a" * 40,
            runtime_versions={
                "transformers": "4.55.4",
                "torch": "2.5.1+cu124",
                "bitsandbytes": "0.47.0",
            },
            config=config(),
            limit=limit,
            torch_module=FakeTorch(),
            adapter_metadata=adapter_metadata,
        )
        return result, tokenizer, fake_model

    def test_prompt_connection_new_token_decode_and_success_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = "VALIDATION_SCORE_MUST_NOT_BE_USED_987654"
            result, tokenizer, model = self.run_fake(
                root, [record(1, score_marker=marker)], [VALID_OUTPUT]
            )

            messages, chat_kwargs = tokenizer.chat_calls[0]
            all_message_text = "\n".join(message["content"] for message in messages)
            self.assertNotIn(marker, all_message_text)
            self.assertIn("주제 1", messages[1]["content"])
            self.assertIn("본문 1", messages[1]["content"])
            self.assertEqual(
                chat_kwargs,
                {"tokenize": False, "add_generation_prompt": True},
            )
            self.assertEqual(
                tokenizer.tokenize_calls,
                [("rendered chat", {"return_tensors": "pt"})],
            )
            self.assertEqual(
                tokenizer.decode_calls,
                [([101, 102], {"skip_special_tokens": True})],
            )
            generate_call = model.generate_calls[0]
            self.assertFalse(generate_call["do_sample"])
            self.assertEqual(generate_call["max_new_tokens"], 1024)
            for forbidden in ("temperature", "top_p", "top_k", "stop_strings"):
                self.assertNotIn(forbidden, generate_call)

            raw = read_jsonl(root / "outputs" / "raw_generations.jsonl")
            predictions = read_jsonl(root / "outputs" / "predictions.jsonl")
            metadata = json.loads(
                (root / "outputs" / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(raw[0]["raw_output"], VALID_OUTPUT)
            self.assertEqual(raw[0]["generated_tokens"], 2)
            self.assertFalse(raw[0]["generation_truncated"])
            self.assertEqual(predictions[0]["essay_id"], "doc-1")
            self.assertEqual(predictions[0]["judge"]["content"]["score"], 3)
            self.assertEqual(result.success_count, 1)
            self.assertEqual(metadata["model_revision"], "a" * 40)
            self.assertEqual(metadata["transformers_version"], "4.55.4")
            self.assertEqual(metadata["torch_version"], "2.5.1+cu124")
            self.assertEqual(metadata["bitsandbytes_version"], "0.47.0")
            self.assertEqual(metadata["prompt_snapshot_sha256"], "63684fd1d584f73d93eb91e2c00e8b0a027322112fd69a65c985771aa91403b0")
            self.assertNotIn("stop_strings", metadata["generation_config"])
            self.assertEqual(metadata["max_new_tokens"], 1024)
            self.assertEqual(metadata["truncation_count"], 0)
            self.assertIn("peak_allocated_vram_bytes", metadata)
            self.assertIn("peak_reserved_vram_bytes", metadata)
            self.assertEqual(metadata["adapter"], {"enabled": False})

    def test_parse_failure_preserves_raw_output(self) -> None:
        invalid = "```json\n{}\n```"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _, _ = self.run_fake(root, [record(1)], [invalid])
            failures = read_jsonl(root / "outputs" / "failures.jsonl")
            predictions = read_jsonl(root / "outputs" / "predictions.jsonl")
            self.assertEqual(result.failure_count, 1)
            self.assertEqual(predictions, [])
            self.assertEqual(failures[0]["raw_output"], invalid)
            self.assertEqual(failures[0]["error"]["code"], "markdown_code_fence")

    def test_max_new_tokens_exhaustion_is_recorded_as_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = FakeModel(1)
            model.outputs = [[11, 12, *range(1024)]]
            result, _, _ = self.run_fake(
                root,
                [record(1)],
                ['{"content":'],
                model=model,
            )

            raw = read_jsonl(root / "outputs" / "raw_generations.jsonl")
            failures = read_jsonl(root / "outputs" / "failures.jsonl")
            metadata = json.loads(
                (root / "outputs" / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result.failure_count, 1)
            self.assertEqual(raw[0]["generated_tokens"], 1024)
            self.assertTrue(raw[0]["generation_truncated"])
            self.assertEqual(failures[0]["generated_tokens"], 1024)
            self.assertTrue(failures[0]["generation_truncated"])
            self.assertEqual(failures[0]["error"]["code"], "invalid_json")
            self.assertEqual(metadata["max_new_tokens"], 1024)
            self.assertEqual(metadata["truncation_count"], 1)

    def test_limit_counts_only_attempted_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _, model = self.run_fake(
                root,
                [record(1), record(2), record(3)],
                [VALID_OUTPUT],
                limit=1,
            )
            self.assertEqual(result.attempted_count, 1)
            self.assertEqual(len(model.generate_calls), 1)
            self.assertEqual(
                len(read_jsonl(root / "outputs" / "predictions.jsonl")), 1
            )

    def test_resume_skips_existing_successful_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "outputs"
            self.run_fake(root, [record(1)], [VALID_OUTPUT])

            result, _, model = self.run_fake(
                root,
                [record(1), record(2)],
                [VALID_OUTPUT],
            )
            predictions = read_jsonl(output_dir / "predictions.jsonl")
            metadata = json.loads(
                (output_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result.skipped_count, 1)
            self.assertEqual(result.attempted_count, 1)
            self.assertEqual(len(model.generate_calls), 1)
            self.assertEqual([item["essay_id"] for item in predictions], ["doc-1", "doc-2"])
            self.assertEqual(metadata["existing_success_count"], 1)
            self.assertEqual(metadata["completed_success_count"], 2)

    def test_resume_rejects_invalid_existing_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "outputs"
            self.run_fake(root, [record(1)], [VALID_OUTPUT])
            write_jsonl(
                output_dir / "predictions.jsonl",
                [{"essay_id": "doc-1", "judge": {}}],
            )
            input_path = root / "input.jsonl"
            write_jsonl(input_path, [record(1)])

            with self.assertRaisesRegex(ValueError, "not a completed prediction"):
                run_inference_pipeline(
                    input_path=input_path,
                    output_dir=output_dir,
                    tokenizer=FakeTokenizer([VALID_OUTPUT]),
                    model=FakeModel(1),
                    model_revision="a" * 40,
                    runtime_versions={
                        "transformers": "4.55.4",
                        "torch": "2.5.1+cu124",
                        "bitsandbytes": "0.47.0",
                    },
                    config=config(),
                    torch_module=FakeTorch(),
                )

    def test_resume_rejects_a_different_model_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_fake(root, [record(1)], [VALID_OUTPUT])
            input_path = root / "input.jsonl"

            with self.assertRaisesRegex(ValueError, "resume metadata does not match"):
                run_inference_pipeline(
                    input_path=input_path,
                    output_dir=root / "outputs",
                    tokenizer=FakeTokenizer([VALID_OUTPUT]),
                    model=FakeModel(1),
                    model_revision="b" * 40,
                    runtime_versions={
                        "transformers": "4.55.4",
                        "torch": "2.5.1+cu124",
                        "bitsandbytes": "0.47.0",
                    },
                    config=config(),
                    torch_module=FakeTorch(),
                )

    def test_adapter_metadata_is_recorded(self) -> None:
        adapter = {
            "enabled": True,
            "path": "/mnt/checkpoints/adapter",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
            "fingerprint_sha256": "f" * 64,
            "config": {"peft_type": "LORA"},
            "is_trainable": False,
            "merged": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_fake(
                root,
                [record(1)],
                [VALID_OUTPUT],
                adapter_metadata=adapter,
            )
            metadata = json.loads(
                (root / "outputs" / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["adapter"], adapter)

    def test_resume_rejects_a_different_adapter_fingerprint(self) -> None:
        first_adapter = {
            "enabled": True,
            "path": "/mnt/checkpoints/adapter",
            "fingerprint_sha256": "a" * 64,
            "config": {"peft_type": "LORA"},
        }
        second_adapter = {
            **first_adapter,
            "fingerprint_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_fake(
                root,
                [record(1)],
                [VALID_OUTPUT],
                adapter_metadata=first_adapter,
            )
            with self.assertRaisesRegex(
                ValueError, "resume metadata does not match"
            ):
                self.run_fake(
                    root,
                    [record(1)],
                    [VALID_OUTPUT],
                    adapter_metadata=second_adapter,
                )

    def test_legacy_zero_shot_metadata_without_adapter_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_fake(root, [record(1)], [VALID_OUTPUT])
            metadata_path = root / "outputs" / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["adapter"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            result, _, _ = self.run_fake(
                root,
                [record(1), record(2)],
                [VALID_OUTPUT],
            )
            self.assertEqual(result.skipped_count, 1)
            self.assertEqual(result.success_count, 1)

    def test_prior_outputs_survive_generation_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            output_dir = root / "outputs"
            write_jsonl(input_path, [record(1), record(2)])
            tokenizer = FakeTokenizer([VALID_OUTPUT])
            model = FakeModel(2, fail_at=2)

            with self.assertRaisesRegex(RuntimeError, "CUDA OOM"):
                run_inference_pipeline(
                    input_path=input_path,
                    output_dir=output_dir,
                    tokenizer=tokenizer,
                    model=model,
                    model_revision="a" * 40,
                    runtime_versions={
                        "transformers": "4.55.4",
                        "torch": "2.5.1+cu124",
                        "bitsandbytes": "0.47.0",
                    },
                    config=config(),
                    torch_module=FakeTorch(),
                )

            self.assertEqual(len(read_jsonl(output_dir / "raw_generations.jsonl")), 1)
            self.assertEqual(len(read_jsonl(output_dir / "predictions.jsonl")), 1)
            metadata = json.loads(
                (output_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["error"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
