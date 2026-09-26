# Final Results

This file summarizes the final model selection and competition results. Full chronological experiment details are in [experiments.md](experiments.md).

## Selected model

- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Adaptation: 4-bit NF4 QLoRA
- Training data: 2,000 essays
- Epochs: 1
- Selected checkpoint: weighted QLoRA v1
- Final artifact: merged standalone BF16 Hugging Face model
- Model: https://huggingface.co/slvrfivo/qwen3-4b-writing-eval-v1-merged

## Local validation

| Model | Mean RMSE ↓ | Mean Spearman ↑ | Interpretation |
|---|---:|---:|---|
| Statistical mean baseline | 0.7870 | undefined* | Constant rounded predictions |
| Qwen3-4B zero-shot | 1.1612 | 0.2753 | Poor score calibration |
| **Weighted QLoRA v1** | **0.6384** | **0.5237** | Selected checkpoint |
| Class-balanced QLoRA v1b | 0.6452 | 0.5189 | Negative ablation |

*Spearman is undefined for the statistical baseline because all rounded predictions are constant.

The selected QLoRA setup improved over both the statistical mean baseline and the zero-shot LLM baseline on local validation. A uniform-loss QLoRA run was not performed, so this improvement should not be attributed specifically to the weighted token objective.

## Official hidden test

| Metric | Score |
|---|---:|
| RMSE ↓ | **0.6213** |
| Spearman ↑ | **0.5697** |
| LLM Judge | **3.0337** |
| Preliminary leaderboard | **49 / 53 teams** |

The top 50 models advanced to the AI말평 Arena expert human-evaluation stage.

The official hidden-test RMSE (**0.6213**) was slightly lower than local validation (**0.6384**). This supports that the selected checkpoint transferred reasonably to held-out competition data, but it should not be treated as proof that the model did not overfit because the splits and evaluation conditions differ.

## Arena

- Arena points: **842**
- Rank: **44 / 50 Arena models**

These results are reported as competition outcomes rather than evidence of state-of-the-art performance.

## Final serving artifact

The selected LoRA adapter was merged into the pinned BF16 base model and exported as a standalone Hugging Face artifact.

During final submission validation, repetitive degeneration appeared on a small subset of merged-model generations. The final artifact therefore used:

~~~json
{
  "repetition_penalty": 1.05
}
~~~

The reproducible source of this setting is [../configs/submission_generation.json](../configs/submission_generation.json). The export CLI now loads that file and applies the override to the exported Hugging Face `generation_config.json` before local reload and strict-JSON smoke validation.

## What the results support

The experiments support the following claims:

- supervised QLoRA adaptation substantially improved score calibration relative to zero-shot inference;
- the selected setup outperformed a rounded statistical mean baseline on local validation;
- global class balancing increased extreme-score predictions but slightly worsened overall RMSE and Spearman;
- the final merged BF16 artifact could be validated through the competition serving path.

The experiments do **not** isolate the causal contribution of the weighted token objective because a uniform-loss QLoRA ablation was not run.
