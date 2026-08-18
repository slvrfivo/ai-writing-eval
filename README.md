# 2026 국립국어원 AI말평 글쓰기 채점 능력 평가

2026 국립국어원 AI말평의 **글쓰기 채점 능력 평가**를 위한 프로젝트입니다.

현재는 개발환경과 기본 디렉터리 구조만 구성되어 있으며, 모델 학습 및 평가 코드는 추후 추가합니다.

## 프로젝트 구조

```text
.
├── data/
│   ├── raw/          # 원본 데이터
│   └── processed/    # 전처리된 데이터
├── src/              # 소스 코드
├── notebooks/        # 탐색 및 실험용 노트북
├── outputs/          # 예측 결과와 실험 산출물
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

아직 외부 패키지 의존성은 없습니다.
