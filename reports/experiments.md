# 실험 기록

모든 지표는 `src/evaluate.py`의 공식 평가 로직을 사용한다. 예측 실수는 `ROUND_HALF_UP`으로 정수화하며, `score.average`는 평가하지 않는다. `undefined`는 공식 정수화 후 예측 순위에 분산이 없어 Spearman이 수학적으로 정의되지 않음을 뜻한다.

| 날짜 | 실험 이름 | 사용한 방법 | Content RMSE | Organization RMSE | Expression RMSE | 평균 RMSE | Content Spearman | Organization Spearman | Expression Spearman | 평균 Spearman | 핵심 해석 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-08-21 | Global Mean Baseline | Train의 영역별 전체 평균을 모든 validation 샘플에 적용 | 0.691990 | 0.916089 | 0.753015 | 0.787031 | undefined | undefined | undefined | undefined | Train 평균은 3.277750/3.337375/3.672625이고 공식 정수화 후 모든 글에 3/3/4를 예측한다. 상수 예측이므로 순위 상관은 정의되지 않는다. |
| 2026-08-21 | Prompt Mean Baseline | Train의 prompt_num별 영역 평균을 적용하고 미관측 prompt는 global mean으로 fallback | 0.691990 | 0.916089 | 0.753015 | 0.787031 | undefined | undefined | undefined | undefined | Q1~Q9의 원시 평균은 다르지만 공식 정수화 후 모두 3/3/4가 되어 Global Mean과 완전히 동일하다. fallback 사용은 0건이다. |

상세 train 평균과 prompt_num별 평균은 `outputs/baselines/baseline_summary.md`에 저장한다. `outputs/`는 재생성 가능한 산출물이므로 Git에서 제외한다.
