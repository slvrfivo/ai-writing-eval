# Qwen3-4B + QLoRA for Korean Argumentative Essay Scoring

Built for the 2026 **National Institute of Korean Language (NIKL) AI말평 writing-scoring evaluation**.

QLoRA fine-tuning with a score-focused weighted token objective reduced validation RMSE to **0.638** (mean baseline: **0.787**, zero-shot: **1.161**) and raised Spearman from **0.275** to **0.524**. On the hidden official test set, the final submission scored **RMSE 0.621 / Spearman 0.570**.

The selected model was merged into standalone BF16 weights, published on Hugging Face, and validated with vLLM for competition serving.

- **Base model:** `Qwen/Qwen3-4B-Instruct-2507`
- **Fine-tuning:** 4-bit NF4 QLoRA
- **Official test:** RMSE **0.6213** / Spearman **0.5697** / LLM Judge **3.0337**
- **Hugging Face:** https://huggingface.co/slvrfivo/qwen3-4b-writing-eval-v1-merged

## Results

| Model / Evaluation | RMSE ↓ | Spearman ↑ | Notes |
|---|---:|---:|---|
| Statistical mean baseline | 0.7870 | undefined* | Rounded constant prediction |
| Qwen3-4B zero-shot | 1.1612 | 0.2753 | 399 / 400 valid predictions |
| **Weighted QLoRA v1** | **0.6384** | **0.5237** | Selected local checkpoint |
| Class-balanced QLoRA v1b | 0.6452 | 0.5189 | Negative ablation |
| **Official hidden test** | **0.6213** | **0.5697** | LLM Judge 3.0337 |

*The statistical baseline is constant after official rounding, so its prediction ranks have zero variance and Spearman correlation is mathematically undefined.

The preliminary leaderboard placed the submission **49th of 53 teams**. The top 50 models then entered the **AI말평 Arena**, an expert human-evaluation stage; this model recorded **842 points and 44th of 50 Arena models**.

Detailed experiment logs: [reports/experiments.md](reports/experiments.md)

## Task

Given a writing prompt and a Korean argumentative essay, the model predicts 1–5 scores for:

- **Content**
- **Organization**
- **Expression**

It also generates a Korean rationale for each dimension in a strict JSON format.

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

## Approach

### 1. Reproduce the official evaluator

The local evaluator in `src/evaluate.py` reproduces the competition's quantitative scoring behavior:

- validates the required JSON structure,
- applies `ROUND_HALF_UP` before scoring,
- computes RMSE and Spearman for Content, Organization, and Expression,
- handles tied ranks with average ranks.

This was implemented before model tuning so that local experiments used the same score conversion rule as the competition.

### 2. Establish a zero-shot baseline

The first LLM baseline used `Qwen/Qwen3-4B-Instruct-2507` with NF4 4-bit quantization and greedy decoding.

It achieved:

- mean RMSE: **1.1612**
- mean Spearman: **0.2753**
- strict JSON success: **399 / 400**

The model strongly over-scored essays, especially on Organization, motivating supervised adaptation.

### 3. QLoRA fine-tuning

The training environment provided an NVIDIA A100 MIG instance with about **9.5 GiB VRAM**, so the project used QLoRA rather than full-parameter fine-tuning.

The selected run used:

- NF4 4-bit base model
- BF16 compute
- LoRA over all linear layers
- `r=16`, `alpha=32`, `dropout=0.05`
- 1 epoch
- learning rate `5e-5`
- effective batch size 8
- max sequence length 2048

Full configuration: [configs/qwen3_4b_qlora_v1.json](configs/qwen3_4b_qlora_v1.json)

## Score-focused Weighted Token Objective

The task evaluates the predicted scores directly, but a standard causal language-model loss treats all supervised output tokens similarly.

To reflect the task structure, each supervised token was mapped to a semantic role and assigned a different loss weight:

| Token role | Loss weight |
|---|---:|
| Prompt | 0.00 |
| JSON structure | 0.25 |
| **Score** | **10.00** |
| Rationale | 0.05 |

The training objective is:

~~~text
weighted loss = Σ(token loss × token weight) / Σ(token weight)
~~~

This design emphasizes the score tokens while still supervising valid JSON structure and rationale generation.

The weights were chosen **heuristically for this competition setup**, not from a systematic hyperparameter search.

Implementation:

- `src/qlora/tokenization.py` — maps tokens to semantic roles
- `src/qlora/loss.py` — computes weighted causal cross entropy
- `src/qlora/training.py` — integrates the objective with Transformers Trainer

> A uniform-loss QLoRA ablation was not run, so the gain over zero-shot or the statistical baseline should be attributed to the **overall QLoRA fine-tuning setup**, not specifically to the weighted objective.

## Ablation: Class-balanced Score Weighting

The selected model still concentrated predictions around scores 3 and 4, so a second run added class-dependent weights to score tokens to emphasize rare labels.

| Model | RMSE ↓ | Spearman ↑ |
|---|---:|---:|
| **Weighted QLoRA v1** | **0.6384** | **0.5237** |
| Class-balanced QLoRA v1b | 0.6452 | 0.5189 |

The class-balanced version produced more extreme predictions, but overall RMSE and Spearman became slightly worse. It was therefore retained as a negative ablation rather than replacing v1.

## Export and Serving

The selected PEFT adapter was merged into the pinned BF16 base model and exported as a standalone Hugging Face model.

The export pipeline performs:

**merge → local reload → model/tokenizer invariant check → strict JSON smoke test**

During final submission testing, repetitive degeneration appeared on a small subset of merged-model generations. The final Hugging Face artifact therefore used:

~~~json
{
  "repetition_penalty": 1.05
}
~~~

in `generation_config.json`.

The same override is recorded in [configs/submission_generation.json](configs/submission_generation.json).

The final artifact was then validated through a vLLM OpenAI-compatible serving path using the competition-required health, model-listing, and chat-completions endpoints.

## Reproduction

Install dependencies:

~~~bash
pip install -r requirements-train-gpu.txt
~~~

Train the selected QLoRA configuration:

~~~bash
python src/train_qlora.py \
  --input data/raw/train.jsonl \
  --output-dir checkpoints/qwen3_4b_qlora_v1 \
  --config configs/qwen3_4b_qlora_v1.json
~~~

Run validation inference with the adapter:

~~~bash
python src/zero_shot.py \
  --input data/raw/validation.jsonl \
  --output-dir outputs/qlora_v1 \
  --adapter checkpoints/qwen3_4b_qlora_v1/final_adapter
~~~

Evaluate predictions:

~~~bash
python src/evaluate.py \
  --predictions outputs/qlora_v1/predictions.jsonl \
  --require-rationale
~~~

Merge to standalone BF16 weights:

~~~bash
python src/export_merged_hf.py \
  --adapter checkpoints/qwen3_4b_qlora_v1/final_adapter \
  --output-dir /path/to/submission/qwen3-4b-writing-eval-v1-merged \
  --validation-input data/raw/validation.jsonl
~~~

Environment details and reproducibility notes: [docs/reproducibility.md](docs/reproducibility.md)

## Repository Layout

~~~text
ai-writing-eval/
├── configs/        # zero-shot, QLoRA, and final serving configs
├── docs/           # competition and reproducibility notes
├── prompts/        # versioned official prompt
├── reports/        # experiment log and official results
├── src/
│   ├── llm/        # prompting, parsing, inference, export
│   └── qlora/      # QLoRA data, tokenization, loss, training
├── tests/
├── requirements*.txt
├── LICENSE
└── README.md
~~~

## Limitations

- The training set contains 2,000 essays.
- QLoRA v1 still shows regression toward middle scores.
- Scores 1 and 5 are harder to predict than scores 3 and 4.
- A uniform-loss QLoRA ablation was **not** performed, so the specific contribution of weighted token loss is not isolated.
- Class balancing increased extreme predictions but reduced overall validation performance.
- Local validation reproduces RMSE and Spearman, but not the official LLM Judge.
- Performance outside the competition prompts, domain, and rubric has not been established.

## Data and License

Competition data is not redistributed in this repository. Data, checkpoints, generated outputs, and model weights are excluded through `.gitignore`.

Project code is released under the Apache License 2.0. Third-party models, competition data, and official materials remain subject to their own terms.

## Official References

- Task overview: https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=205
- Preliminary leaderboard: https://kli.korean.go.kr/benchmark/taskOrdtm/taskLeaderBoard.do?clCd=ING_TASK&subMenuId=sub04&taskOrdtmId=205
- Arena overview: https://kli.korean.go.kr/taskOrdtm/taskList.do?clCd=END_TASK&subMenuId=sub01&taskOrdtmId=215
