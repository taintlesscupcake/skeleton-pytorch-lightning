# PyTorch Lightning을 사용한 PyTorch 스켈레톤 코드

이 저장소에는 PyTorch Lightning을 사용하여 딥러닝 모델을 학습하고 테스트하기 위한 스켈레톤 코드가 포함되어 있습니다. 스켈레톤 코드는 `module.py`, `model.py`, `dataloader.py` 등의 여러 모듈로 구성되어 있습니다. 이 코드를 사용자의 특정 사례에 맞게 쉽게 수정할 수 있습니다.

## 목차

- [요구 사항](#요구-사항)
- [프로젝트 구조](#프로젝트-구조)
- [사용법](#사용법)
- [사용자 정의](#사용자-정의)
- [라이선스](#라이선스)

## 요구 사항

- Python 3.7 이상
- PyTorch 1.8.0 이상
- PyTorch Lightning 1.4.0 이상
- torchvision (데이터셋에 필요한 경우 선택적으로 사용)


## 프로젝트 구조

프로젝트에는 다음과 같은 파일이 포함되어 있습니다:

- `train.py`: 모델 학습을 위한 주요 스크립트입니다.
- `test.py`: 모델 테스트를 위한 주요 스크립트입니다.
- `module.py`: `SkeletonModule` 및 `SkeletonDataModule` 클래스가 포함되어 있습니다.
- `model.py`: `SkeletonModel` 클래스가 포함되어 있습니다.
- `dataloader.py`: `SkeletonDataset` 클래스가 포함되어 있습니다.
- `train.sh`: 학습 스크립트를 실행하기 위한 쉘 스크립트입니다.
- `test.sh`: 테스트 스크립트를 실행하기 위한 쉘 스크립트입니다.

## 사용법

먼저 이 저장소를 클론합니다.

```bash
git clone https://github.com/taintlesscupcake/skeleton-pytorch-lightning.git
cd skeleton-pytorch-lightning
```

모델을 학습하려면 다음 명령을 실행합니다:

```bash
./train.sh
# 권한 오류가 발생하는 경우 chmod +x train.sh를 사용합니다.
```

모델을 테스트하려면 다음 명령을 실행합니다:

```bash
./test.sh
# 권한 오류가 발생하는 경우 chmod +x test.sh를 사용합니다.
```


## 사용자 정의

특정 사용 사례에 맞게 코드를 사용자 정의하려면 다음 단계를 수행하세요:

1. **`model.py`의 `SkeletonModel` 업데이트**: `SkeletonModel` 클래스를 수정하여 사용자만의 딥러닝 모델을 정의합니다.

2. **`dataloader.py`의 `SkeletonDataset` 업데이트**: 사용자의 특정 데이터셋을 로드하기 위해 `SkeletonDataset` 클래스를 수정합니다.

3. **`module.py`의 `SkeletonModule` 업데이트**: 손실 함수, 최적화 알고리즘 및 학습률 스케줄링을 포함하도록 `SkeletonModule` 클래스를 수정합니다.

4. **`module.py`의 `SkeletonDataModule` 업데이트**: 데이터 증강 및 데이터 분할을 포함하여 데이터 로딩 파이프라인을 설정하기 위해 `SkeletonDataModule` 클래스를 수정합니다.

5. **`train.py` 및 `test.py` 수정**: 사용자 정의 클래스 및 원하는 학습/테스트 구성과 일치하도록 `train.py`와 `test.py`의 인수와 설정을 조정합니다.

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)에 따라 라이선스가 부여됩니다.
