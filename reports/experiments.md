# Experiment Log

All metrics use the official evaluation logic implemented in `src/evaluate.py`. Predicted scores are converted to integers with `ROUND_HALF_UP`, and `score.average` is not evaluated. `undefined` means Spearman correlation is mathematically undefined because the rounded predictions have zero rank variance.

| Date | Experiment | Method | Content RMSE | Organization RMSE | Expression RMSE | Mean RMSE | Content Spearman | Organization Spearman | Expression Spearman | Mean Spearman | Key interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-08-21 | Global Mean Baseline | Apply the train-set mean for each dimension to every validation sample | 0.691990 | 0.916089 | 0.753015 | 0.787031 | undefined | undefined | undefined | undefined | Train means are 3.277750 / 3.337375 / 3.672625; after official rounding, every essay receives 3 / 3 / 4. Spearman is undefined because the predictions are constant. |
| 2026-08-21 | Prompt Mean Baseline | Apply per-`prompt_num` train means; fall back to the global mean for unseen prompts | 0.691990 | 0.916089 | 0.753015 | 0.787031 | undefined | undefined | undefined | undefined | Raw means differ across Q1–Q9, but official rounding maps all of them to 3 / 3 / 4, making this identical to the Global Mean baseline. No fallback cases occurred. |

Detailed global and per-`prompt_num` means are written to `outputs/baselines/baseline_summary.md`. The `outputs/` directory is excluded from Git because it contains reproducible artifacts.

## Qwen3-4B zero-shot baseline (2026-08-23)

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Inference: official competition prompt, NF4 4-bit, BF16 compute, `max_new_tokens=1024`, `do_sample=False`, `batch_size=1`
- Hardware: NVIDIA A100 MIG 1g, approximately 9.5 GiB usable VRAM
- Validation: 399 valid predictions out of 400, 99.75% parse success, 0 truncations, 1 invalid JSON output
- Failure case: the model generated an invalid JSON escape (`\'`). The metrics below exclude that single sample.

### Metrics on 399 valid predictions

| Dimension | RMSE | Spearman |
| --- | ---: | ---: |
| Content | 1.0619648881007777 | 0.2878363668373772 |
| Organization | 1.6904475093758122 | 0.258665621321177 |
| Expression | 0.7311756249261975 | 0.279418387980397 |
| Mean | 1.1611960074675958 | 0.2753067920463171 |

### Calibration

| Dimension | GT Mean | Pred Mean | Pred Distribution |
| --- | ---: | ---: | --- |
| Content | 3.2280701754386003 | 3.892230576441103 | `{2: 18, 3: 73, 4: 242, 5: 66}` |
| Organization | 3.287593984962406 | 4.734335839598997 | `{3: 11, 4: 84, 5: 304}` |
| Expression | 3.676065162907268 | 3.974937343358396 | `{2: 1, 3: 35, 4: 336, 5: 27}` |

### Observations

- The model consistently over-scored essays, with the largest calibration bias in Organization.
- The predictions contained some relative ranking signal, but score-scale calibration was poor.
- The next experiment therefore moved to supervised adaptation with QLoRA.

## Qwen3-4B weighted QLoRA v1 (2026-08-24)

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Training: 2,000 train samples, 1 epoch, NF4 4-bit, BF16 compute, score-focused weighted token loss
- Evaluation: full validation set of 400 samples

### Validation metrics

| Dimension | RMSE | Spearman |
| --- | ---: | ---: |
| Content | 0.5812486559124245 | 0.53258708712125 |
| Organization | 0.750312434922946 | 0.524797525323729 |
| Expression | 0.5837647214417808 | 0.5136796292092026 |
| Mean | 0.6384419374257172 | 0.5236880805513939 |

### Calibration

| Dimension | GT Mean | Pred Mean | Pred Distribution |
| --- | ---: | ---: | --- |
| Content | 3.229 | 3.3475 | `{2: 5, 3: 251, 4: 144}` |
| Organization | 3.286875 | 3.37 | `{2: 5, 3: 242, 4: 153}` |
| Expression | 3.676875 | 3.78 | `{2: 3, 3: 82, 4: 315}` |

### Observations

- The broad over-scoring seen in zero-shot inference was substantially reduced.
- Predictions were compressed around scores 3 and 4; no 1-point or 5-point predictions were produced.
- Ground-truth 2s were usually predicted as 3, while ground-truth 5s were usually predicted as 4, indicating regression toward the mean.
- Rationales became much simpler and followed the rubric-template style used to construct the training targets.
- The next ablation tested mild class weighting derived only from the training-label distribution.

## Qwen3-4B class-balanced QLoRA v1b negative ablation (2026-08-25)

- Evaluation: full validation set of 400 samples
- Change from v1: keep the v1 configuration and add train-label-based class balancing to score-token weights

### Validation metrics

| Dimension | RMSE | Spearman |
| --- | ---: | ---: |
| Content | 0.5944325024761011 | 0.5149457041109735 |
| Organization | 0.7659756849926764 | 0.5075662424977525 |
| Expression | 0.5751358535163671 | 0.5341676427869871 |
| Mean | 0.6451813469950481 | 0.5188931964652377 |

### Prediction distribution

| Dimension | Pred Distribution |
| --- | --- |
| Content | `{2: 11, 3: 242, 4: 147}` |
| Organization | `{2: 13, 3: 234, 4: 147, 5: 6}` |
| Expression | `{2: 6, 3: 87, 4: 301, 5: 6}` |

### Conclusion

- Overall performance was worse than v1.
- Extreme-score predictions became more frequent.
- Expression improved on both RMSE and Spearman.
- Content and Organization both degraded.
- The global class-balancing scheme appears to have over-corrected.
- v1 remained the best overall checkpoint.

## Official competition results (2026-08-25 to 2026-09-23)

Selected submission:

- Model: Qwen3-4B Writing Scoring QLoRA v1
- Official preliminary RMSE: **0.6213**
- Official preliminary Spearman: **0.5697**
- Official preliminary LLM Judge: **3.0337**
- Preliminary leaderboard: **49th of 53 teams**
- Arena qualification: top 50 models advanced
- AI말평 Arena: **842 points, 44th of 50 Arena models**

The selected competition artifact was the merged BF16 Hugging Face model:

https://huggingface.co/slvrfivo/qwen3-4b-writing-eval-v1-merged

During final serving validation, repetitive degeneration was observed on a small subset of merged-model generations. The final Hugging Face artifact therefore set `repetition_penalty=1.05` in `generation_config.json`. The repository records this submission-time override in `configs/submission_generation.json`.

Official references:

- Preliminary leaderboard: https://kli.korean.go.kr/benchmark/taskOrdtm/taskLeaderBoard.do?clCd=ING_TASK&subMenuId=sub04&taskOrdtmId=205
- Arena overview: https://kli.korean.go.kr/taskOrdtm/taskList.do?clCd=END_TASK&subMenuId=sub01&taskOrdtmId=215
