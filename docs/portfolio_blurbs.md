# Portfolio / Resume Descriptions

These descriptions are intentionally limited to claims supported by the repository and recorded competition results.

## Korean — one-line resume version

**국립국어원 AI말평 한국어 논증문 채점 모델 개발** — Qwen3-4B를 4-bit QLoRA로 파인튜닝하고 score-focused weighted token loss, 공식 평가 로직 재현, BF16 모델 병합·vLLM 서빙 검증까지 구현하여 비공개 공식 평가에서 RMSE 0.6213 / Spearman 0.5697을 기록.

## Korean — bullet version

- Qwen3-4B 기반 한국어 논증문 자동 채점 모델을 4-bit NF4 QLoRA로 파인튜닝하고, 내용·구성·표현 점수 및 근거를 strict JSON으로 생성하는 파이프라인 구현.
- 대회 평가 특성을 반영해 score / JSON structure / rationale token에 서로 다른 가중치를 주는 weighted causal LM loss를 구현하고 class-balanced weighting ablation을 수행; local validation에서 mean baseline RMSE 0.7870 대비 0.6384 기록.
- 선택한 LoRA adapter를 standalone BF16 모델로 병합하고 reload/invariant/JSON smoke test 및 vLLM serving을 검증; 비공개 공식 test에서 RMSE 0.6213 / Spearman 0.5697 / LLM Judge 3.0337 기록.

## Korean — portfolio paragraph

2026 국립국어원 AI말평 글쓰기 채점 능력 평가를 위해 Qwen3-4B 기반 한국어 논증문 자동 채점 시스템을 개발했습니다. 공식 반올림·RMSE·Spearman 평가 로직을 먼저 로컬에 재현한 뒤, 약 9.5 GiB VRAM 환경에서 4-bit NF4 QLoRA 파인튜닝을 수행했습니다. 점수 토큰을 더 직접적으로 학습하기 위해 score, JSON structure, rationale token별 가중치를 다르게 적용하는 causal LM loss를 구현했으며, class-balanced weighting은 별도 ablation으로 비교했습니다. 최종 선택 모델은 local validation에서 RMSE 0.6384 / Spearman 0.5237을 기록했고, 비공개 공식 test에서는 RMSE 0.6213 / Spearman 0.5697 / LLM Judge 3.0337을 기록했습니다. 이후 LoRA adapter를 BF16 base model에 병합해 standalone Hugging Face artifact로 export하고, reload/invariant check, strict JSON smoke test와 vLLM serving까지 검증했습니다.

## English — resume version

**Korean argumentative essay scoring with Qwen3-4B** — Fine-tuned Qwen3-4B with 4-bit QLoRA, implemented a score-focused weighted token objective and competition-compatible evaluator, and built a BF16 merge/validation/serving pipeline; achieved **0.6213 RMSE / 0.5697 Spearman** on the hidden official test set.

## English — bullet version

- Fine-tuned Qwen3-4B with 4-bit NF4 QLoRA to generate 1–5 Content, Organization, and Expression scores with Korean rationales in strict JSON.
- Implemented competition-compatible rounding/evaluation and a token-weighted causal LM objective for score, JSON-structure, and rationale tokens; achieved **0.6384 local RMSE** versus a **0.7870 statistical mean baseline** and evaluated a class-balanced negative ablation.
- Merged the selected LoRA adapter into standalone BF16 weights and validated reload invariants, strict-JSON smoke inference, and vLLM serving; the hidden official test scored **0.6213 RMSE / 0.5697 Spearman / 3.0337 LLM Judge**.

## Accuracy notes

Do not describe the weighted objective as the proven cause of the performance improvement: no uniform-loss QLoRA ablation was run.

For competition placement, use the exact recorded results if needed: **49 / 53 teams** in the preliminary leaderboard and **842 points / 44 / 50 models** in the Arena. Avoid wording such as “top finalist.”
