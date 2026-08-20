# 2026 국립국어원 AI말평 글쓰기 채점 능력 평가

2026 국립국어원 AI말평의 **글쓰기 채점 능력 평가**를 위한 프로젝트입니다.

현재 단계에서는 공식 규칙 정리와 train/validation 탐색적 데이터 분석(EDA)까지만 수행합니다. 모델 다운로드, LLM 추론, 파인튜닝 코드는 포함하지 않습니다.

## 대회 핵심 규칙

아래 내용은 `docs/`의 과제 기술서, Docker Image 제출 규정, Hugging Face URL 제출 규정을 기준으로 정리했다. 공식 [AI말평 과제 페이지](https://kli.korean.go.kr/benchmark/taskOrdtm/taskList.do?taskOrdtmId=205)도 함께 참고한다. 문서 간 충돌이 있으면 날짜가 더 최신인 공지를 우선한다.

### 입력

- 한 편의 한국어 논증적 글에 대한 주제(`prompt_text`)와 본문(`essay_text`)이 모델 입력이다.
- 제공 JSONL에서는 각각 `prompt`, `essay` 필드에 해당한다.
- 데이터 규모는 train 2,000건, validation 400건, 비공개 test 400건이다.

### 출력

- `content`(내용), `organization`(구성), `expression`(표현) 세 영역을 모두 평가한다.
- 모델 출력은 코드 블록이나 마크다운이 아닌 JSON 객체 하나여야 한다.
- 영역별 `score`는 1~5 범위의 **정수**이며, `rationale`은 실제 essay에 근거한 한국어 설명이어야 한다.
- 실수 점수를 출력하면 2026-08-06 공지에 따라 사사오입하여 정수로 변환한 뒤 평가한다.
- 영역을 서로 독립적으로 판단하고, 글에 없는 내용을 근거로 만들지 않아야 한다.
- 출력 파싱 실패 시 2회 재시도한 뒤에도 실패하면 0점 처리한다.

```json
{
  "content": {"score": 1, "rationale": "내용 판단 근거"},
  "organization": {"score": 1, "rationale": "구성 판단 근거"},
  "expression": {"score": 1, "rationale": "표현 판단 근거"}
}
```

### 평가 지표

- 점수의 절대 오차: RMSE, 비중 45%
- 점수의 상대 순위: Spearman 순위 상관계수, 비중 45%
- 점수와 근거의 정성 평가: LLM Judge(Qwen3.6-35B-A3B, 4-bit Q4_K_M GGUF), 비중 10%
- 각 지표에서 content·organization·expression 결과를 평균하고, 지표별 순위를 0~1로 정규화한 뒤 비중을 적용한다.
- 모델이 실수 점수를 출력한 경우에는 사사오입으로 정수화된 점수를 기준으로 RMSE와 Spearman을 계산한다.
- 전체 대회 단계는 순위표 기반 정량 평가, 말평 아레나 전문가 상대 평가, 발표 평가로 구성된다.

### 추론 조건

- 고정값: `temperature=0.0`, `top_p=1.0`, `seed=42`
- 평가 서버 설정 기준 최대 생성 길이: `max_tokens=2048`
- stop 문자열: `"Q:"`, `"User:"`
- 긴 입력을 포함한 여러 요청에 안정적으로 응답해야 한다.

### GPU 및 모델 제약

- 전체 추론은 단일 NVIDIA L40S 48GB GPU 1장 안에서 독립적으로 실행 가능해야 한다.
- 14B 이하 모델을 권장한다.
- 여러 모델의 순차 실행이나 앙상블은 허용되지만 동일한 단일 GPU 제약 안에서 완료되어야 한다.
- OpenAI API 같은 외부 API 호출 모델과 비공개 모델은 제출할 수 없다.
- 사용 모델과 코드의 라이선스에 문제가 없어야 한다.

### 제출 방식

- 팀당 예선 모델 제출은 원칙적으로 총 4회이며, 정상 제출 후 다음 제출까지 72시간을 기다려야 한다. 평가 자체가 정상 진행되지 않은 경우에는 횟수를 차감하지 않는다.
- 2026-08-06 점수 처리 방식 변경과 함께 기존 제출 결과를 새 기준으로 재산출하고, 모든 팀의 제출 횟수를 초기화하여 팀별 4회를 새로 부여했다.
- **Hugging Face URL**: 공개 저장소여야 하며, 토큰이나 추가 커스텀 환경 없이 표준 vLLM 평가 환경에서 바로 로드되어야 한다. `config.json`, tokenizer 파일, 실제 컨텍스트 길이 설정이 완전해야 한다.
- **Docker Image**: 추가 명령 인자 없이 `docker run <image>`만으로 서버가 시작되어야 한다. `0.0.0.0:8000`에서 OpenAI 호환 API를 제공하고 `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`를 지원해야 한다.
- Docker 이미지와 Hugging Face 모델 모두 긴 입력에서 컨텍스트 길이 오류나 OOM 없이 동작하는지 제출 전에 확인해야 한다.

### 최신 2026-08-06 점수 반올림 규칙

2026-08-06 국립국어원 공지가 기존 기술서보다 우선한다.

- 모델은 각 영역 점수를 1~5 범위의 정수로 출력해야 한다.
- 실수 점수를 출력하면 **사사오입하여 정수로 변환**한다.
- 변환된 정수를 기준으로 RMSE와 Spearman을 계산한다.
- 기존 제출 결과에도 같은 기준을 적용해 다시 산출한다.
- 처리 방식 변경에 따라 모든 팀의 제출 횟수를 초기화하고 팀별 4회를 새로 부여한다.

학습 라벨에는 정수가 아닌 점수(`content` 최소 0.1 간격, `organization`·`expression` 최소 0.25 간격)가 포함된다. 따라서 연속값을 예측하는 실험을 하더라도 공식 평가 직전에는 위 사사오입 규칙을 그대로 적용한 정수 출력으로 검증해야 한다. 제공된 `average`와 표시된 세 영역 점수의 단순 평균 사이에는 일부 0.01 차이가 있으므로 `average`를 임의로 재계산해 덮어쓰지 않는다.

## 프로젝트 구조

```text
.
├── data/
│   ├── raw/          # 원본 데이터
│   └── processed/    # 전처리된 데이터
├── docs/             # 공식 문서와 최신 공지
├── src/
│   ├── eda.py        # 재사용 가능한 EDA 실행 코드
│   └── evaluate.py   # 공식 반올림 규칙 기반 validation evaluator
├── tests/
│   └── test_evaluate.py
├── notebooks/        # 탐색 및 실험용 노트북
├── outputs/
│   ├── eda_summary.md
│   └── figures/      # EDA 그래프
├── checkpoints/      # 모델 체크포인트
├── .gitignore
├── README.md
└── requirements.txt
```

`data/`, `outputs/`, `checkpoints/`, 모델 가중치 파일과 `.env` 파일은 Git에서 제외됩니다.

## 개발환경

Python 가상환경 사용을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## EDA 실행

기본 경로의 train/validation JSONL을 자동 탐색하여 보고서와 그래프를 생성한다.

```powershell
python src/eda.py
```

경로를 바꾸려면 다음처럼 실행한다.

```powershell
python src/eda.py --data-dir data/raw --output-dir outputs
```

원본 JSONL은 읽기 전용으로 열며 수정하지 않는다.

## Validation 평가

예측 파일은 JSON 객체 배열 또는 JSONL이며, 각 레코드는 validation의 `id`·`document_id` 또는 공식 예시의 `essay_id`로 식별한다. 세 영역을 직접 두거나 공식 `judge` 객체 안에 넣을 수 있다.

```json
[
  {
    "id": "GWGR2300001260",
    "content": {"score": 3.5, "rationale": "내용 근거"},
    "organization": {"score": 3, "rationale": "구성 근거"},
    "expression": {"score": 4, "rationale": "표현 근거"}
  }
]
```

기본 `data/raw`의 validation을 자동 탐색해 평가한다.

```powershell
python src/evaluate.py --predictions predictions.json --require-rationale
```

예측 점수는 먼저 1~5 범위인지 검사하며, 실수이면 `ROUND_HALF_UP`으로 정수화한 뒤 세 영역 RMSE와 Spearman을 계산한다. 범위 밖 값은 clamp하지 않고 오류로 처리하며 `score.average`는 사용하지 않는다.

```powershell
# validation 정답 점수를 예측으로 복사하는 파이프라인 점검
python src/evaluate.py --sanity-check

# 단위 테스트
python -m unittest discover -s tests -v
```
