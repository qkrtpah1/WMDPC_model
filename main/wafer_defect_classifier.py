"""
WM-811K 웨이퍼맵 불량 유형 분류 모델
=====================================

참고: https://www.kaggle.com/code/ashishpatel26/wm-811k-wafermap

WM-811K는 811,457장의 웨이퍼맵 이미지로 구성된 반도체 공정 결함 탐지 공개 데이터셋입니다.
각 웨이퍼맵은 다이(die) 단위로 값이 매겨진 2차원 배열이며,
    0 = 다이 없음(웨이퍼 바깥 영역)
    1 = 정상(pass) 다이
    2 = 불량(fail) 다이
로 구성됩니다. 이 중 약 172,950장에 대해서만 사람이 라벨링한 결함 패턴(failureType)이
존재하며, 총 9개 클래스로 구성됩니다:

    Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none

이 스크립트는 캐글 노트북에서 흔히 쓰이는 전처리(라벨이 존재하는 서브셋만 사용,
웨이퍼맵을 고정 크기로 리사이즈 후 (배경/정상/불량) 3채널 원-핫 이미지로 변환)를
따르고, CNN(합성곱 신경망)으로 9개 클래스를 분류합니다.

사용법
------
1) 실제 데이터로 학습:
   원본 데이터(LSWMD.pkl, Kaggle "WM-811K wafer map" 데이터셋)를 내려받은 뒤:

       python wafer_defect_classifier.py --data LSWMD.pkl --epochs 30

2) 데이터 없이 파이프라인만 검증(합성 데이터 자동 생성):

       python wafer_defect_classifier.py --demo --epochs 3

출력물(모델, 라벨 매핑, 학습 곡선, confusion matrix)은 --output-dir 에 저장됩니다.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 9개 결함 클래스 (WM-811K 표준 라벨)
CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Random", "Scratch", "Near-full", "none",
]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# --------------------------------------------------------------------------
# 1. 데이터 로딩 & 라벨 정제
# --------------------------------------------------------------------------

def _flatten_label(val):
    """failureType/trainTestLabel 컬럼은 numpy object array 안에 중첩 배열/리스트로
    들어있는 경우가 많다 (예: array([['Center']], dtype='<U6')).
    재귀적으로 벗겨서 순수 문자열(또는 라벨 없음이면 None)을 반환한다."""
    v = val
    if isinstance(v, np.ndarray):
        v = v.tolist()
    while isinstance(v, list):
        if len(v) == 0:
            return None
        v = v[0]
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    v = str(v).strip()
    return v if v else None


def _install_legacy_pandas_index_shims():
    """공개 배포된 LSWMD.pkl은 pandas 0.20 이전 버전으로 저장되어, 그 시절 모듈 경로인
    pandas.indexes.*를 pickle에 그대로 담고 있다. 최신 pandas(1.0+)는 이 경로를 더 이상
    인식하지 못하므로, 언피클링 전에 현재 pandas 클래스로 매핑된 가짜 모듈을 등록해준다."""
    import types

    if "pandas.indexes" in sys.modules:
        return

    import pandas.core.indexes.base as idx_base
    import pandas.core.indexes.multi as idx_multi
    import pandas.core.indexes.range as idx_range

    def _new_index(cls, d):
        # 옛 pandas의 Index.__reduce__는 (cls, d)를 넘기고 __new__(cls, **d)로 복원했다.
        try:
            return cls.__new__(cls, **d)
        except TypeError:
            obj = cls.__new__(cls)
            obj.__dict__.update(d)
            return obj

    pkg = types.ModuleType("pandas.indexes")
    pkg.__path__ = []
    sys.modules["pandas.indexes"] = pkg

    for submodule_name, cls_attr, cls in (
        ("base", "Index", idx_base.Index),
        ("range", "RangeIndex", idx_range.RangeIndex),
        ("multi", "MultiIndex", idx_multi.MultiIndex),
    ):
        mod = types.ModuleType(f"pandas.indexes.{submodule_name}")
        setattr(mod, cls_attr, cls)
        mod._new_Index = _new_index
        sys.modules[f"pandas.indexes.{submodule_name}"] = mod


def _read_legacy_pickle(pkl_path):
    """구버전 pandas + Python 2로 저장된 피클을 latin1 인코딩으로 읽는다
    (Python 2의 8비트 문자열을 Python 3에서 안전하게 복원하기 위함)."""
    import pandas.compat.pickle_compat as pickle_compat

    _install_legacy_pandas_index_shims()
    with open(pkl_path, "rb") as f:
        return pickle_compat.Unpickler(f, encoding="latin1").load()


def load_dataframe(pkl_path):
    """LSWMD.pkl을 읽어 라벨(failureType)이 존재하는 서브셋만 정제해서 반환한다."""
    print(f"[1/5] 데이터 로딩: {pkl_path}")
    try:
        df = pd.read_pickle(pkl_path)
    except (ModuleNotFoundError, UnicodeDecodeError, AttributeError) as e:
        print(f"    기본 로딩 실패({e!r}), 구버전 pandas 호환 모드로 재시도")
        df = _read_legacy_pickle(pkl_path)

    print(f"    전체 웨이퍼맵 수: {len(df):,}")

    df["failureLabel"] = df["failureType"].apply(_flatten_label)

    labeled = df[df["failureLabel"].isin(CLASS_NAMES)].copy()
    labeled = labeled.reset_index(drop=True)

    print(f"    라벨링된(9개 클래스) 웨이퍼맵 수: {len(labeled):,}")
    print("    클래스 분포:")
    for name, cnt in labeled["failureLabel"].value_counts().reindex(CLASS_NAMES).items():
        print(f"      - {name:10s}: {int(cnt) if pd.notna(cnt) else 0}")

    return labeled


# --------------------------------------------------------------------------
# 2. 전처리: 웨이퍼맵 -> 고정 크기 3채널(배경/정상/불량) 이미지
# --------------------------------------------------------------------------

def wafer_to_image(wafer_map, img_size=48):
    """웨이퍼맵(값 0/1/2, 가변 크기 2D array)을
    (img_size, img_size, 3) 형태의 원-핫 채널 이미지로 변환.
    최근접 보간(INTER_NEAREST)을 써서 0/1/2 범주값이 섞이지 않게 한다."""
    import cv2

    arr = np.asarray(wafer_map)
    arr = np.clip(arr, 0, 2).astype(np.uint8)
    resized = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    img[..., 0] = (resized == 0).astype(np.float32)  # 다이 없음(배경)
    img[..., 1] = (resized == 1).astype(np.float32)  # 정상 다이
    img[..., 2] = (resized == 2).astype(np.float32)  # 불량 다이
    return img


def build_dataset(df, img_size=48):
    print(f"    {img_size}x{img_size}x3 이미지 변환 ({len(df):,}장)")
    X = np.zeros((len(df), img_size, img_size, 3), dtype=np.float32)
    y = np.zeros((len(df),), dtype=np.int64)

    for i, row in enumerate(df.itertuples(index=False)):
        X[i] = wafer_to_image(row.waferMap, img_size=img_size)
        y[i] = LABEL_TO_IDX[row.failureLabel]
        if (i + 1) % 20000 == 0:
            print(f"    {i + 1:,}/{len(df):,} 변환 완료")

    return X, y


# --------------------------------------------------------------------------
# 3. 모델 정의
# --------------------------------------------------------------------------

def build_cnn(input_shape, num_classes):
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------
# 4. 데모(합성) 데이터 생성 - 실제 LSWMD.pkl 없이 파이프라인 검증용
# --------------------------------------------------------------------------

def _make_synthetic_wafer(pattern, size=40, rng=None):
    """지름 size인 원형 웨이퍼 위에 각 결함 패턴을 흉내내는 합성 웨이퍼맵 생성.
    0=배경, 1=정상, 2=불량."""
    rng = rng or np.random.default_rng()
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = size / 2 - 1

    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[r <= radius] = 1  # 원형 웨이퍼 영역은 기본적으로 정상(pass)

    def add_noise(mask_prob=0.01):
        noise = (rng.random((size, size)) < mask_prob) & (wafer == 1)
        wafer[noise] = 2

    if pattern == "Center":
        wafer[(r <= radius) & (r < radius * 0.35)] = 2
    elif pattern == "Donut":
        wafer[(r <= radius) & (r > radius * 0.35) & (r < radius * 0.65)] = 2
    elif pattern == "Edge-Ring":
        wafer[(r <= radius) & (r > radius * 0.85)] = 2
    elif pattern == "Edge-Loc":
        ang = np.arctan2(yy - cy, xx - cx)
        sector = (ang > -0.6) & (ang < 0.6)
        wafer[(r <= radius) & (r > radius * 0.7) & sector] = 2
    elif pattern == "Loc":
        blob = (yy - cy * 0.6) ** 2 + (xx - cx * 0.6) ** 2 < (radius * 0.2) ** 2
        wafer[(r <= radius) & blob] = 2
    elif pattern == "Scratch":
        for t in np.linspace(-radius, radius, 200):
            yi = int(cy + t * 0.6)
            xi = int(cx + t)
            if 0 <= yi < size and 0 <= xi < size and r[yi, xi] <= radius:
                wafer[max(0, yi - 1):yi + 1, max(0, xi - 1):xi + 1] = 2
    elif pattern == "Near-full":
        wafer[(r <= radius)] = rng.choice([1, 2], size=wafer[(r <= radius)].shape, p=[0.15, 0.85])
    elif pattern == "Random":
        add_noise(mask_prob=0.12)
    elif pattern == "none":
        pass  # 결함 없음, 정상 다이만

    add_noise(mask_prob=0.01)  # 약간의 배경 잡음(모든 클래스 공통)
    return wafer


def make_demo_dataframe(n_per_class=120, size=40, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for cls in CLASS_NAMES:
        for _ in range(n_per_class):
            rows.append({
                "waferMap": _make_synthetic_wafer(cls, size=size, rng=rng),
                "failureType": np.array([[cls]], dtype=object),
                "trainTestLabel": np.array([["Training"]], dtype=object),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. 학습 & 평가
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WM-811K 웨이퍼맵 불량 분류 CNN")
    parser.add_argument("--data", type=str, default=None, help="LSWMD.pkl 경로")
    parser.add_argument("--demo", action="store_true", help="합성 데이터로 파이프라인만 검증")
    parser.add_argument("--demo-per-class", type=int, default=120, help="데모 모드: 클래스당 샘플 수")
    parser.add_argument("--img-size", type=int, default=48, help="리사이즈할 웨이퍼맵 크기")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="./wafer_model_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh", action="store_true",
                         help="output-dir에 이어할 체크포인트가 있어도 무시하고 처음부터 새로 학습")
    args = parser.parse_args()

    if not args.demo and not args.data:
        parser.error("--data LSWMD.pkl 경로를 지정하거나, 파이프라인만 확인하려면 --demo 를 사용하세요.")

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # 1) 데이터 로드
    if args.demo:
        print("[데모 모드] 합성 웨이퍼맵 데이터를 생성합니다 (실제 데이터셋 미사용).")
        df = make_demo_dataframe(n_per_class=args.demo_per_class, size=40, seed=args.seed)
        df["failureLabel"] = df["failureType"].apply(_flatten_label)
    else:
        df = load_dataframe(args.data)

    # 2) train/val/test 분할 (계층적 샘플링) — 이미지로 변환하기 전에 인덱스만 나눈다.
    #    전체를 먼저 이미지 배열로 만든 뒤 분할하면, 원본 X와 분할된 복사본이
    #    한꺼번에 메모리에 떠 있게 되어(대략 2배) 대용량 데이터셋에서 메모리 부족이
    #    나기 쉽다. 인덱스만 분할하고 각 split을 따로 이미지로 변환하면 이를 피한다.
    from sklearn.model_selection import train_test_split

    y_all = df["failureLabel"].map(LABEL_TO_IDX).to_numpy()
    idx_all = np.arange(len(df))

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all, y_all, test_size=args.val_split + args.test_split,
        stratify=y_all, random_state=args.seed,
    )
    rel_test = args.test_split / (args.val_split + args.test_split)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=args.seed,
    )
    print(f"[2/5] 데이터 분할 -> train {len(idx_train):,} / val {len(idx_val):,} / test {len(idx_test):,}")

    # 3) 전처리 (split별로 개별 변환하여 메모리 사용량을 낮게 유지)
    print("[3/5] 웨이퍼맵 -> 이미지 변환")
    X_train, y_train = build_dataset(df.iloc[idx_train], img_size=args.img_size)
    X_val, y_val = build_dataset(df.iloc[idx_val], img_size=args.img_size)
    X_test, y_test = build_dataset(df.iloc[idx_test], img_size=args.img_size)

    import gc

    del df
    gc.collect()

    # 4) 클래스 불균형 보정 가중치
    from sklearn.utils.class_weight import compute_class_weight

    present_classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(present_classes, weights)}

    # 5) 모델 구성 & 학습
    from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.models import load_model

    class _EpochStateSaver(Callback):
        """매 epoch이 끝날 때마다 완료된 epoch 번호를 파일에 기록한다.
        재시작 시 이 값으로 model.fit(initial_epoch=...)를 넘겨 이어서 학습한다."""

        def __init__(self, state_path):
            super().__init__()
            self.state_path = state_path

        def on_epoch_end(self, epoch, logs=None):
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({"last_epoch": epoch}, f)

    best_ckpt_path = os.path.join(args.output_dir, "best_model.keras")
    last_ckpt_path = os.path.join(args.output_dir, "last_checkpoint.keras")
    state_path = os.path.join(args.output_dir, "train_state.json")

    initial_epoch = 0
    if not args.fresh and os.path.exists(last_ckpt_path) and os.path.exists(state_path):
        print(f"[4/5] 기존 체크포인트에서 이어 학습: {last_ckpt_path}")
        model = load_model(last_ckpt_path)
        with open(state_path, "r", encoding="utf-8") as f:
            initial_epoch = json.load(f)["last_epoch"] + 1
        print(f"    epoch {initial_epoch + 1}부터 재개 (목표 {args.epochs} epoch)")
        if initial_epoch >= args.epochs:
            print(f"    이미 목표 epoch({args.epochs})에 도달했습니다. --epochs를 늘리거나 --fresh로 새로 시작하세요.")
    else:
        print("[4/5] 모델 학습")
        model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(CLASS_NAMES))
        model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ModelCheckpoint(best_ckpt_path, monitor="val_accuracy", save_best_only=True),
        ModelCheckpoint(last_ckpt_path, save_best_only=False),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        _EpochStateSaver(state_path),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # 6) 평가
    print("[5/5] 테스트셋 평가")
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=[CLASS_NAMES[i] for i in present_labels],
        digits=3,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=present_labels)

    # 결과물 저장
    model.save(os.path.join(args.output_dir, "wafer_defect_cnn.keras"))

    with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL_TO_IDX, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 학습 곡선
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(history.history["loss"], label="train")
        axes[0].plot(history.history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(history.history["accuracy"], label="train")
        axes[1].plot(history.history["val_accuracy"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)
        plt.close(fig)

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(present_labels)))
        ax.set_yticks(range(len(present_labels)))
        ax.set_xticklabels([CLASS_NAMES[i] for i in present_labels], rotation=45, ha="right")
        ax.set_yticklabels([CLASS_NAMES[i] for i in present_labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib 미설치: 시각화 파일 생성을 건너뜁니다.")

    print(f"\n완료. 결과물 저장 위치: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
