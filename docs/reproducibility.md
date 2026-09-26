# Reproducibility

This document records the environment and workflow used for the original experiments without depending on machine-specific project paths.

## Original environment

- OS: Ubuntu 18.04
- Python: 3.10.13
- PyTorch: 2.5.1+cu124
- CUDA runtime used by PyTorch: 12.4
- GPU: NVIDIA A100 MIG
- Usable VRAM: approximately 9.5 GiB
- Transformers: 4.55.4
- Accelerate: 1.10.1
- bitsandbytes: 0.47.0
- PEFT: 0.16.0

PyTorch is intentionally not pinned in requirements-gpu.txt because the original GPU environment already supplied the CUDA-enabled PyTorch build.

## Suggested paths

Use any local paths appropriate for your machine. For example:

~~~bash
export PROJECT_ROOT=/path/to/ai-writing-eval
export HF_HOME=/path/to/hf-cache
export CHECKPOINT_DIR=/path/to/checkpoints
~~~

No passwords, SSH keys, API tokens, or other secrets should be stored in the repository.

## Install

~~~bash
cd "$PROJECT_ROOT"

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements-train-gpu.txt
python -m pip check
~~~

On the original GPU machine, PyTorch 2.5.1+cu124 was already installed before the requirements files were installed.

## Training

~~~bash
python src/train_qlora.py \
  --input data/raw/train.jsonl \
  --output-dir "$CHECKPOINT_DIR/qwen3_4b_qlora_v1" \
  --config configs/qwen3_4b_qlora_v1.json
~~~

The training pipeline records run metadata including the model revision, package versions, prompt snapshot hash, sequence-length statistics, loss weights, QLoRA configuration, optimizer-step count, and CUDA memory measurements.

## Local evaluation

Run adapter inference:

~~~bash
python src/zero_shot.py \
  --input data/raw/validation.jsonl \
  --output-dir outputs/qlora_v1 \
  --adapter "$CHECKPOINT_DIR/qwen3_4b_qlora_v1/final_adapter"
~~~

Then evaluate:

~~~bash
python src/evaluate.py \
  --predictions outputs/qlora_v1/predictions.jsonl \
  --require-rationale
~~~

The evaluator applies the official ROUND_HALF_UP score conversion before computing RMSE and Spearman.

## BF16 export

~~~bash
python src/export_merged_hf.py \
  --adapter "$CHECKPOINT_DIR/qwen3_4b_qlora_v1/final_adapter" \
  --output-dir /path/to/submission/qwen3-4b-writing-eval-v1-merged \
  --validation-input data/raw/validation.jsonl
~~~

The exporter safely merges the PEFT adapter, reloads the saved BF16 model locally, verifies configuration/tokenizer invariants, and runs strict-JSON smoke inference.

## Final competition generation override

The final competition artifact used repetition_penalty=1.05 after repetitive degeneration was observed on a small subset of merged-model generations.

The override is recorded in:

~~~text
configs/submission_generation.json
~~~

For exact final-artifact reproduction, set the same value in the exported Hugging Face generation_config.json before serving the model.

The final artifact was validated with a vLLM OpenAI-compatible server using the competition-required health, model-listing, and chat-completions endpoints.
