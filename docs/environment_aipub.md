# AIPub GPU 환경

이 문서는 학교 AIPub 서버에서 GPU 실험을 재현하는 데 필요한 환경을 기록한다.
비밀번호, SSH 개인 키, 접근 토큰을 비롯한 비밀 정보는 이 문서에 기록하지 않는다.

## 고정 환경

- 운영체제: Ubuntu 18.04
- Python: 3.10.13
- PyTorch: `2.5.1+cu124`
- PyTorch가 사용하는 CUDA 런타임: 12.4
- GPU: NVIDIA A100 MIG
- 실제 사용 가능한 VRAM: 약 9.5 GiB
- 프로젝트 경로: `/mnt/ai-writing-eval`
- 가상 환경(virtualenv) 경로: `/mnt/venvs/ai-writing-eval`
- Hugging Face 캐시(`HF_HOME`): `/mnt/hf-cache`
- 체크포인트 경로: `/mnt/checkpoints`

PyTorch는 가상 환경에 미리 설치되어 있으므로 `requirements-gpu.txt`에서 의도적으로
제외한다. 기준선 환경을 준비할 때 기존 가상 환경을 다시 만들거나 패키지 전체를
업그레이드하지 않는다. 두 작업 모두 정상 동작하는 PyTorch CUDA 빌드를 다른 버전으로
교체할 수 있다.

## GPU 의존성 설치

Git으로 소스 트리를 갱신한 뒤 기존 가상 환경을 활성화한다. 아래 명령은 모델을
다운로드하거나 추론을 실행하지 않는다.

```bash
cd /mnt/ai-writing-eval
source /mnt/venvs/ai-writing-eval/bin/activate
export HF_HOME=/mnt/hf-cache

python --version
python -c "import torch; assert torch.__version__ == '2.5.1+cu124', torch.__version__; assert torch.version.cuda == '12.4', torch.version.cuda; print(torch.__version__, torch.version.cuda)"
python -m pip install -r requirements-gpu.txt
python -m pip check
```

`requirements-gpu.txt`에 `torch`를 추가하지 말고, 이 환경을 준비할 때 `pip install
--upgrade`를 사용하지 않는다. 설치된 `torch==2.5.1+cu124`는 고정된 Hugging Face
패키지가 선언한 PyTorch 의존성을 충족하므로 `--upgrade` 없는 일반 설치에서는
현재 버전이 유지된다.

## 모델을 로드하지 않는 환경 검증

```bash
python -c "import accelerate, bitsandbytes, transformers; print(transformers.__version__, accelerate.__version__, bitsandbytes.__version__)"
python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0), 'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)})"
```

예상 패키지 버전은 다음과 같다.

```text
transformers 4.55.4
accelerate 1.10.1
bitsandbytes 0.47.0
torch 2.5.1+cu124
```

모델은 `HF_HOME`을 통해 `/mnt/hf-cache`에 다운로드해야 한다. 향후 학습
체크포인트는 `/mnt/checkpoints`에 저장해야 하며, 두 경로 모두 Git에서 관리하지 않는다.
