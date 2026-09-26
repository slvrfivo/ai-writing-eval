---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: transformers
pipeline_tag: text-generation
language:
- ko
tags:
- qwen3
- qlora
- korean
- essay-scoring
- writing-assessment
---

# Qwen3-4B Korean Writing Scoring QLoRA v1

A Qwen3-4B model adapted for the 2026 National Institute of Korean Language (NIKL) AI말평 Korean writing-scoring evaluation.

Given a writing prompt and a completed Korean argumentative essay, the model generates a strict JSON object containing 1–5 scores and Korean rationales for three dimensions:

- Content
- Organization
- Expression

## Model details

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Adaptation: 4-bit NF4 QLoRA
- Selected training run: 2,000 samples, 1 epoch
- Export: LoRA adapter merged into standalone BF16 weights
- Primary language: Korean
- Intended task: competition-specific argumentative-essay scoring

The source repository, training configuration, evaluator, experiment log, and export pipeline are available at:

https://github.com/slvrfivo/ai-writing-eval

## Evaluation

### Local validation

| Model | Mean RMSE ↓ | Mean Spearman ↑ |
|---|---:|---:|
| Statistical mean baseline | 0.7870 | undefined* |
| Qwen3-4B zero-shot | 1.1612 | 0.2753 |
| **Selected weighted QLoRA v1** | **0.6384** | **0.5237** |
| Class-balanced QLoRA v1b | 0.6452 | 0.5189 |

*The rounded statistical baseline is constant, so Spearman correlation is undefined.

### Official hidden test

- RMSE: **0.6213**
- Spearman: **0.5697**
- LLM Judge: **3.0337**
- Preliminary leaderboard: **49th of 53 teams**
- AI말평 Arena: **842 points, 44th of 50 models**

The top 50 preliminary models entered the Arena expert human-evaluation stage.

## Training objective

The project used a score-focused weighted causal language-model objective. Supervised target tokens were assigned role-specific weights:

| Token role | Weight |
|---|---:|
| Prompt | 0.00 |
| JSON structure | 0.25 |
| Score | 10.00 |
| Rationale | 0.05 |

These weights were chosen heuristically for the competition setup. A uniform-loss QLoRA ablation was not run, so the improvement over zero-shot and statistical baselines should be attributed to the overall QLoRA fine-tuning setup rather than specifically to the weighted objective.

## Generation

The final competition artifact used:

~~~json
{
  "repetition_penalty": 1.05
}
~~~

This setting was added after repetitive degeneration was observed on a small subset of merged-model generations. It is recorded in the source repository at `configs/submission_generation.json` and is applied by the BF16 export pipeline.

If a serving system overrides the model's generation configuration, preserve `repetition_penalty=1.05` to reproduce the final competition serving setup.

## vLLM serving example

A minimal serving command is:

~~~bash
vllm serve slvrfivo/qwen3-4b-writing-eval-v1-merged \
  --dtype bfloat16
~~~

For competition-equivalent scoring, use the versioned prompt in the source repository:

~~~text
prompts/official/writing_scoring_2026-07-20/
~~~

The repository's prompt builder and inference pipeline show the exact message construction. If the serving layer does not honor the model's saved `generation_config.json`, pass `repetition_penalty=1.05` explicitly in the request/server generation settings.

## Output format

The expected output is one JSON object:

~~~json
{
  "content": {
    "score": 4,
    "rationale": "..."
  },
  "organization": {
    "score": 3,
    "rationale": "..."
  },
  "expression": {
    "score": 4,
    "rationale": "..."
  }
}
~~~

Each score must be an integer from 1 to 5.

## Limitations

- This model was tuned for a specific competition rubric and prompt distribution.
- The training set contained 2,000 essays.
- Predictions still show regression toward middle scores.
- Scores 1 and 5 are harder to predict than scores 3 and 4.
- The model has not been validated as a general-purpose educational assessment system.
- The rationale output should not be treated as equivalent to detailed human feedback.
- The specific contribution of weighted token loss is not isolated because a uniform-loss QLoRA ablation was not performed.
- Performance outside the competition domain and rubric is unknown.

## Data, license, and terms

- Base model: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0)
- Source code in the GitHub repository: Apache License 2.0
- Competition data: not redistributed; subject to the competition's own terms
- Official competition materials: subject to their respective terms

This model artifact is provided as a competition/research artifact, not as a validated high-stakes scoring system. The model card does not grant additional rights over competition data or third-party materials.
