# Korean Argumentative Essay Scoring with QLoRA

An end-to-end Korean argumentative essay scoring project developed for the 2026 National Institute of Korean Language (NIKL) AI Malpyeong Writing Scoring Evaluation.

Given a writing prompt and an argumentative essay, the model predicts 1–5 scores for Content, Organization, and Expression and generates a Korean rationale for each dimension in a strict JSON format.

The project starts from a Qwen3-4B zero-shot baseline, applies 4-bit QLoRA with a score-focused weighted token loss, evaluates a class-balanced ablation, merges the selected LoRA adapter into standalone BF16 weights, and validates the final Hugging Face artifact for competition serving.

- **Base model:** Qwen/Qwen3-4B-Instruct-2507
- **Fine-tuning:** 4-bit NF4 QLoRA
- **Best local validation:** RMSE **0.6384** / Spearman **0.5237**
- **Official preliminary evaluation:** RMSE **0.6213** / Spearman **0.5697** / LLM Judge **3.0337**
- **Preliminary leaderboard:** **49th of 53 teams**
- **AI Malpyeong Arena:** **842 points / 44th of 50 Arena models**
- **Deployment:** merged BF16 Hugging Face model
- **Model:** https://huggingface.co/slvrfivo/qwen3-4b-writing-eval-v1-merged

## Results

### Local validation

| Model | RMSE ↓ | Spearman ↑ | Notes |
|---|---:|---:|---|
| Statistical baseline | 0.7870 | undefined* | Rounded prediction collapses to 3 / 3 / 4 |
| Qwen3-4B zero-shot | 1.1612 | 0.2753 | 399 / 400 valid predictions |
| **Weighted QLoRA v1** | **0.6384** | **0.5237** | Selected local checkpoint |
| Class-balanced QLoRA v1b | 0.6452 | 0.5189 | Negative ablation |

*The statistical baseline produces a constant prediction after official rounding, so the prediction ranks have zero variance and Spearman correlation is mathematically undefined.

Compared with the zero-shot baseline, weighted QLoRA reduced mean validation RMSE from **1.1612 to 0.6384** and increased mean Spearman correlation from **0.2753 to 0.5237**.

The zero-shot model strongly over-scored essays, especially for Organization. QLoRA substantially improved calibration, although the selected model still concentrated predictions around scores 3 and 4.

The class-balanced ablation generated more rare extreme-score predictions but slightly degraded overall RMSE and Spearman, so the original weighted QLoRA v1 checkpoint was retained.

Detailed experiment logs are available in [reports/experiments.md](reports/experiments.md).

### Official competition evaluation

| Evaluation | RMSE ↓ | Spearman ↑ | LLM Judge | Result |
|---|---:|---:|---:|---|
| Preliminary leaderboard | **0.6213** | **0.5697** | **3.0337** | **49 / 53 teams** |
| AI Malpyeong Arena | — | — | — | **842 points / 44 / 50 models** |

The official preliminary stage used RMSE, Spearman correlation, and an LLM Judge. The top 50 models advanced to the Arena stage, where Korean-language experts compared model-generated scores and rationales.

Official task page: https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=205

Arena overview: https://kli.korean.go.kr/taskOrdtm/taskList.do?clCd=END_TASK&subMenuId=sub01&taskOrdtmId=215

## Task

The input consists of:

- a writing prompt
- a completed Korean argumentative essay

The model returns one JSON object with independent scores and rationales for Content, Organization, and Expression.

~~~json
{
  "content": {
    "score": 4,
    "rationale": "주장과 근거가 주제에 맞게 제시되어 있으나 일부 근거의 구체성이 부족하다."
  },
  "organization": {
    "score": 3,
    "rationale": "전체적인 구조는 갖추고 있지만 문단 간 연결이 다소 자연스럽지 않다."
  },
  "expression": {
    "score": 4,
    "rationale": "대체로 적절한 어휘와 문장을 사용했으나 일부 표현이 반복된다."
  }
}
~~~

Each score must be within the 1–5 range.

## Approach

### 1. Reproduce the competition evaluator

Before fine-tuning, the competition scoring behavior was reproduced locally in src/evaluate.py.

The evaluator:

- validates the prediction JSON structure,
- evaluates Content, Organization, and Expression independently,
- rejects predictions outside the 1–5 range,
- rounds real-valued predictions with ROUND_HALF_UP,
- computes RMSE for absolute scoring error,
- computes Spearman correlation for relative ranking,
- handles tied ranks using average ranks.

This matters because the training labels contain fractional scores, while the official evaluation rounds model predictions before computing the quantitative metrics.

### 2. Establish a Qwen3-4B zero-shot baseline

The initial LLM baseline used Qwen/Qwen3-4B-Instruct-2507 with 4-bit NF4 quantization.

| Setting | Value |
|---|---|
| Quantization | NF4 4-bit |
| Compute dtype | BF16 |
| Double quantization | enabled |
| Batch size | 1 |
| Max new tokens | 1024 |
| Sampling | disabled |
| Seed | 42 |

The zero-shot run produced valid strict-JSON predictions for 399 of 400 validation samples.

| Dimension | RMSE ↓ | Spearman ↑ |
|---|---:|---:|
| Content | 1.0620 | 0.2878 |
| Organization | 1.6904 | 0.2587 |
| Expression | 0.7312 | 0.2794 |
| **Mean** | **1.1612** | **0.2753** |

The largest issue was score calibration. For example, the zero-shot Organization prediction mean was approximately 4.73, compared with a ground-truth mean of approximately 3.29. This motivated supervised adaptation instead of relying only on prompt engineering.

## QLoRA Fine-tuning

### Why QLoRA

The original training environment provided an NVIDIA A100 MIG instance with approximately 9.5 GiB of usable VRAM.

Instead of full-parameter fine-tuning, the project used QLoRA:

1. Load the frozen base model with NF4 4-bit quantization.
2. Attach trainable low-rank LoRA adapters to linear layers.
3. Train only the adapter parameters.
4. Merge the selected adapter back into the original BF16 model for deployment.

This allowed a 4B-parameter model to be adapted within the available GPU memory while still producing a standard standalone Hugging Face model for submission.

### Training configuration

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen3-4B-Instruct-2507 |
| Quantization | NF4 4-bit |
| Compute dtype | BF16 |
| LoRA target modules | all-linear |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Epochs | 1 |
| Learning rate | 5e-5 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Optimizer | paged_adamw_8bit |
| Scheduler | cosine |
| Warmup ratio | 0.05 |
| Gradient checkpointing | enabled |
| Max sequence length | 2048 |
| Seed | 42 |

The complete configuration is stored in configs/qwen3_4b_qlora_v1.json.

Training examples longer than the configured maximum sequence length are rejected instead of being silently truncated.

## Score-focused Weighted Token Loss

A standard causal language-model objective gives supervised output tokens equal importance. That is not ideal for this task: a wrong score token directly affects RMSE and Spearman, while a punctuation error or a rationale token does not have the same impact on the quantitative metrics.

The training pipeline therefore maps each supervised token to a semantic role and assigns role-specific weights:

| Token role | Loss weight |
|---|---:|
| Prompt | 0.00 |
| JSON structure | 0.25 |
| **Score** | **10.00** |
| Rationale | 0.05 |

The objective is a weighted causal cross entropy:

~~~text
weighted loss = Σ(token loss × token weight) / Σ(token weight)
~~~

This puts most of the training pressure on the score predictions while retaining enough supervision to learn the required JSON structure and rationale format.

Implementation:

- src/qlora/tokenization.py — maps target tokens to semantic roles
- src/qlora/loss.py — computes token-weighted causal cross entropy
- src/qlora/training.py — integrates the custom loss with Transformers Trainer

## Ablation: Class-balanced Score Weighting

The selected QLoRA v1 model still showed regression toward middle scores, with predictions concentrated around 3 and 4.

A second experiment therefore multiplied score-token weights by class-dependent weights derived from the training-label distribution.

| Model | RMSE ↓ | Spearman ↑ |
|---|---:|---:|
| **Weighted QLoRA v1** | **0.6384** | **0.5237** |
| Class-balanced QLoRA v1b | 0.6452 | 0.5189 |

The class-balanced model produced more 2-point and 5-point predictions, but overall performance became slightly worse. Expression improved, while Content and Organization degraded.

This experiment was kept as a negative ablation rather than replacing the selected checkpoint.

## Model Export

Training produces a PEFT LoRA adapter rather than a standalone model. For submission, the selected adapter was merged into the pinned BF16 base model with PEFT merge_and_unload using safe merge.

The export pipeline:

1. loads the pinned BF16 base model,
2. loads the trained LoRA adapter,
3. safely merges the adapter,
4. writes safetensors weights,
5. saves the tokenizer and generation configuration,
6. reloads the exported model locally,
7. checks model and tokenizer invariants,
8. runs strict-JSON smoke inference,
9. optionally compares merged-model scores with the original 4-bit base + LoRA path.

The resulting model is available at:

https://huggingface.co/slvrfivo/qwen3-4b-writing-eval-v1-merged

## Final Submission Serving Configuration

During final submission testing, repetitive generation was observed on a small subset of inputs after merging the adapter into BF16 weights.

The final Hugging Face artifact therefore set:

~~~json
{
  "repetition_penalty": 1.05
}
~~~

in its generation_config.json before submission. The repository records this final override in configs/submission_generation.json.

The final artifact was then validated through a vLLM OpenAI-compatible serving path, including the competition-required endpoints:

~~~text
GET  /health
GET  /v1/models
POST /v1/chat/completions
~~~

The local zero-shot configuration predates this submission-time fix and should not be interpreted as the complete final serving configuration.

## Reproduction

### Install dependencies

CPU-side utilities:

~~~bash
pip install -r requirements.txt
~~~

GPU inference:

~~~bash
pip install -r requirements-gpu.txt
~~~

QLoRA training:

~~~bash
pip install -r requirements-train-gpu.txt
~~~

The original experiment environment is documented in docs/reproducibility.md.

### Zero-shot inference

~~~bash
python src/zero_shot.py \
  --input data/raw/validation.jsonl \
  --output-dir outputs/zero_shot
~~~

### QLoRA training

~~~bash
python src/train_qlora.py \
  --input data/raw/train.jsonl \
  --output-dir checkpoints/qwen3_4b_qlora_v1 \
  --config configs/qwen3_4b_qlora_v1.json
~~~

### Adapter inference

~~~bash
python src/zero_shot.py \
  --input data/raw/validation.jsonl \
  --output-dir outputs/qlora_v1 \
  --adapter checkpoints/qwen3_4b_qlora_v1/final_adapter
~~~

### Evaluation

~~~bash
python src/evaluate.py \
  --predictions outputs/qlora_v1/predictions.jsonl \
  --require-rationale
~~~

### Merge into standalone BF16 weights

~~~bash
python src/export_merged_hf.py \
  --adapter checkpoints/qwen3_4b_qlora_v1/final_adapter \
  --output-dir /path/to/submission/qwen3-4b-writing-eval-v1-merged \
  --validation-input data/raw/validation.jsonl
~~~

After export, apply the final generation override recorded in configs/submission_generation.json to the exported generation_config.json before competition serving.

## Project Structure

~~~text
ai-writing-eval/
├── configs/
│   ├── qwen3_4b_zero_shot.json
│   ├── qwen3_4b_qlora_v1.json
│   ├── qwen3_4b_qlora_v1b.json
│   └── submission_generation.json
│
├── docs/
│   ├── competition.md
│   ├── notice_2026-08-06_score_rounding.md
│   └── reproducibility.md
│
├── prompts/
│   └── official/
│
├── reports/
│   └── experiments.md
│
├── src/
│   ├── baseline.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── zero_shot.py
│   ├── train_qlora.py
│   ├── export_merged_hf.py
│   ├── llm/
│   └── qlora/
│
├── tests/
├── requirements.txt
├── requirements-gpu.txt
├── requirements-train-gpu.txt
├── LICENSE
└── README.md
~~~

Training data, generated outputs, checkpoints, and model weights are intentionally excluded from Git.

## Testing

The repository includes tests covering:

- official score rounding and evaluation,
- baseline calculations,
- strict model-output parsing,
- prompt construction,
- inference pipeline behavior,
- QLoRA configuration and CLI,
- training-data preparation,
- token-role assignment,
- weighted token loss,
- class balancing,
- model loading,
- merged-model export validation.

Run all tests with:

~~~bash
python -m unittest discover -s tests -v
~~~

## Limitations

This model was developed specifically for the 2026 AI Malpyeong Korean argumentative writing evaluation and should not be interpreted as a general-purpose or production-grade writing assessment system.

- The training set contains 2,000 essays.
- QLoRA v1 still shows regression toward middle scores.
- Scores 1 and 5 are harder to predict than scores 3 and 4.
- Global class balancing increased extreme predictions but reduced overall validation performance.
- Rationale targets are rubric-oriented and should not be treated as detailed human feedback.
- Local validation reproduces RMSE and Spearman but not the official LLM Judge.
- Performance outside the competition prompts, domains, and rubric has not been established.

## Data

Competition data is not redistributed in this repository.

To reproduce the experiments, obtain the dataset through the official AI Malpyeong distribution process and place the train and validation files under a local data directory.

Data, checkpoints, generated outputs, and model weights are excluded through .gitignore.

## License

The project code is released under the Apache License 2.0. The base model, Qwen/Qwen3-4B-Instruct-2507, is also distributed under Apache-2.0.

Competition data and other third-party materials remain subject to their respective terms and are not covered by this repository license.
