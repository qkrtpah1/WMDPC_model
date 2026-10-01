# 웨이퍼(WM-811K) 불량 분류 CNN

참고: [Kaggle - WM-811k Wafermap](https://www.kaggle.com/code/ashishpatel26/wm-811k-wafermap)

WM-811K 데이터셋(웨이퍼맵, 0=배경/1=정상/2=불량)에서 라벨링된 서브셋(9개 클래스:
Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none)을 사용해
CNN으로 불량 패턴을 분류합니다.

## 추론 결과 미리보기

라벨 없는 웨이퍼맵 638,507장(WM-811K 중 사람이 분류하지 않은 나머지 전체)에 대해
학습된 모델로 추론한 결과입니다. `none`, `Edge-Ring`, `Center`처럼 학습 데이터가 많았던
클래스는 신뢰도도 높고, `Loc`·`Scratch`는 test셋 평가에서도 약했던 클래스라 신뢰도가
낮게 나옵니다 (`visualize_predictions.py`로 생성, `predict.py` 실행 후 재생성 가능).

![예측 결과 분포 및 클래스별 신뢰도](assets/predictions_overview.png)

## 설치

TensorFlow는 Python 3.14를 아직 지원하지 않으므로, Python 3.11 가상환경(`.venv`)을 사용합니다.

```bash
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## 폴더 구조

- `data/train/` — 학습용 `LSWMD.pkl`을 넣는 곳 ([data/train/README.md](data/train/README.md))
- `data/predict/` — 예측(추론)할 새 웨이퍼맵 `.pkl`을 넣는 곳 ([data/predict/README.md](data/predict/README.md))
- `wafer_model_output/` — 학습/예측 결과물이 저장되는 곳 (git에는 커밋되지 않음)

## 사용법

### 1) 실제 데이터로 학습

[Kaggle "WM-811K wafer map" 데이터셋](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)에서
`LSWMD.pkl`을 내려받아 `data/train/`에 넣은 뒤:

```bash
.venv\Scripts\python wafer_defect_classifier.py --data data/train/LSWMD.pkl --epochs 30 --img-size 48
```

학습이 중간에 끊겨도(`output-dir`에 `last_checkpoint.keras`, `train_state.json`이 남아있으면) 같은 명령을 다시 실행하면
자동으로 마지막 epoch부터 이어서 학습합니다. 처음부터 새로 학습하려면 `--fresh`를 추가하세요.
(재개 시에는 `--data`, `--val-split`, `--test-split`, `--seed`, `--img-size`를 이전 실행과 동일하게 맞춰야 동일한
train/val/test 분할이 재현됩니다.)

### 2) 데이터 없이 파이프라인만 검증 (합성 데이터 자동 생성)

```bash
.venv\Scripts\python wafer_defect_classifier.py --demo --epochs 5
```

### 3) 학습된 모델로 새 웨이퍼맵 예측(추론)

라벨을 모르는 웨이퍼맵 `.pkl`(`waferMap` 컬럼 포함, `LSWMD.pkl`과 동일 형식)을
`data/predict/`에 넣은 뒤:

```bash
.venv\Scripts\python predict.py --data data/predict --model wafer_model_output/wafer_defect_cnn.keras --label-mapping wafer_model_output/label_mapping.json
```

결과는 `wafer_model_output/predictions.csv`에 저장됩니다(`predictedLabel`, `confidence`,
정답을 아는 경우 `trueLabel`도 포함).

이 결과를 맨 위의 차트처럼 시각화하려면:

```bash
.venv\Scripts\python visualize_predictions.py --predictions wafer_model_output/predictions.csv --output assets/predictions_overview.png
```

## 처리 과정

1. `failureType`이 라벨링된(9개 클래스 중 하나) 웨이퍼맵만 필터링
2. 각 웨이퍼맵을 고정 크기로 리사이즈 후 (배경/정상/불량) 3채널 원-핫 이미지로 변환
3. 계층적 train/val/test 분할 + 클래스 불균형 보정(class_weight)
4. Conv2D 3블록 + BatchNorm + Dropout + GAP 구조의 CNN 학습
5. classification report, confusion matrix, 학습 곡선을 `--output-dir`에 저장

## 산출물

- `wafer_defect_cnn.keras` — 학습된 최종 모델
- `best_model.keras` — 학습 중 val_accuracy가 가장 좋았던 시점의 모델
- `last_checkpoint.keras`, `train_state.json` — 이어 학습(resume)용 체크포인트/상태 파일
- `label_mapping.json` — 클래스 인덱스 매핑
- `classification_report.txt`, `confusion_matrix.png`, `training_curves.png`
- `predictions.csv` — `predict.py` 실행 결과
