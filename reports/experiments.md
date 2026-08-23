# 실험 기록

모든 지표는 `src/evaluate.py`의 공식 평가 로직을 사용한다. 예측 실수는 `ROUND_HALF_UP`으로 정수화하며, `score.average`는 평가하지 않는다. `undefined`는 공식 정수화 후 예측 순위에 분산이 없어 Spearman이 수학적으로 정의되지 않음을 뜻한다.

| 날짜 | 실험 이름 | 사용한 방법 | Content RMSE | Organization RMSE | Expression RMSE | 평균 RMSE | Content Spearman | Organization Spearman | Expression Spearman | 평균 Spearman | 핵심 해석 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-08-21 | Global Mean Baseline | Train의 영역별 전체 평균을 모든 validation 샘플에 적용 | 0.691990 | 0.916089 | 0.753015 | 0.787031 | undefined | undefined | undefined | undefined | Train 평균은 3.277750/3.337375/3.672625이고 공식 정수화 후 모든 글에 3/3/4를 예측한다. 상수 예측이므로 순위 상관은 정의되지 않는다. |
| 2026-08-21 | Prompt Mean Baseline | Train의 prompt_num별 영역 평균을 적용하고 미관측 prompt는 global mean으로 fallback | 0.691990 | 0.916089 | 0.753015 | 0.787031 | undefined | undefined | undefined | undefined | Q1~Q9의 원시 평균은 다르지만 공식 정수화 후 모두 3/3/4가 되어 Global Mean과 완전히 동일하다. fallback 사용은 0건이다. |

상세 train 평균과 prompt_num별 평균은 `outputs/baselines/baseline_summary.md`에 저장한다. `outputs/`는 재생성 가능한 산출물이므로 Git에서 제외한다.

## Qwen3-4B zero-shot 기준선 (2026-08-23)

- 모델: `Qwen/Qwen3-4B-Instruct-2507`
- revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- 추론 설정: 공식 대회 prompt, NF4 4-bit, BF16 compute, `max_new_tokens=1024`, `do_sample=False`, `batch_size=1`
- 실행 환경: NVIDIA A100 MIG 1g, 사용 가능 VRAM 9.5 GiB
- 검증 결과: 전체 400건 중 유효 예측 399건, parse 성공률 99.75%, truncation 0건, invalid JSON 1건
- 실패 원인: JSON에서 유효하지 않은 escape `\'`가 생성되었다. 아래 진단 지표에서는 이 1건을 제외했다.

### 399건 유효 예측 지표

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

### 관찰

- 전반적으로 over-scoring이 발생했으며, 특히 구성 영역의 calibration 편향이 크다.
- Human score에 대한 상대적 ranking signal은 있으나 scoring scale calibration은 부족하다.
- 다음 실험은 supervised adaptation / QLoRA로 진행한다.
