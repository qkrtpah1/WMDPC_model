"""
학습된 모델로 새 웨이퍼맵의 불량 유형을 예측(추론)하는 스크립트
================================================================

사용법
------
    python predict.py --data data/predict \
        --model wafer_model_output/wafer_defect_cnn.keras \
        --label-mapping wafer_model_output/label_mapping.json

`--data` 폴더(하위 폴더 포함) 안의 모든 `.pkl` 파일을 읽습니다. 각 pkl은
`LSWMD.pkl`과 동일한 형식으로 최소한 `waferMap` 컬럼을 포함해야 하며,
`failureType` 컬럼이 있으면(정답을 아는 경우) 예측 정확도도 함께 출력합니다.

결과는 `--output-dir`에 `predictions.csv`로 저장됩니다.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from wafer_defect_classifier import _flatten_label, wafer_to_image


def predict_in_chunks(wafer_maps, model, img_size, chunk_size):
    """전체를 한 번에 이미지 배열로 만들면 수십만 장 규모에서 메모리 부족이 나므로,
    청크 단위로 변환 -> 예측 -> 버리기를 반복해 메모리 사용량을 낮게 유지한다."""
    n = len(wafer_maps)
    pred_idx = np.empty(n, dtype=np.int64)
    confidence = np.empty(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        X_chunk = np.stack([wafer_to_image(w, img_size=img_size) for w in wafer_maps[start:end]])
        probs = model.predict(X_chunk, verbose=0)
        idx = np.argmax(probs, axis=1)
        pred_idx[start:end] = idx
        confidence[start:end] = probs[np.arange(len(probs)), idx]
        print(f"    {end:,}/{n:,} 예측 완료")

    return pred_idx, confidence


def load_predict_dataframe(data_dir):
    pkl_paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True))
    if not pkl_paths:
        raise FileNotFoundError(f"{data_dir} 안에 .pkl 파일이 없습니다.")

    frames = []
    for path in pkl_paths:
        df = pd.read_pickle(path)
        if "waferMap" not in df.columns:
            print(f"    건너뜀 (waferMap 컬럼 없음): {path}")
            continue
        df = df.copy()
        df["sourceFile"] = os.path.basename(path)
        df["trueLabel"] = df["failureType"].apply(_flatten_label) if "failureType" in df.columns else None
        frames.append(df)

    if not frames:
        raise ValueError("waferMap 컬럼을 가진 pkl 파일을 찾지 못했습니다.")

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="웨이퍼맵 불량 유형 예측(추론)")
    parser.add_argument("--data", type=str, default="./data/predict", help="예측할 웨이퍼맵(.pkl)이 들어있는 폴더")
    parser.add_argument("--model", type=str, default="./wafer_model_output/wafer_defect_cnn.keras", help="학습된 모델 경로")
    parser.add_argument("--label-mapping", type=str, default="./wafer_model_output/label_mapping.json", help="라벨 매핑 json 경로")
    parser.add_argument("--img-size", type=int, default=48, help="학습 시 사용한 --img-size와 동일해야 함")
    parser.add_argument("--chunk-size", type=int, default=10000, help="한 번에 이미지로 변환/예측할 웨이퍼맵 수")
    parser.add_argument("--output-dir", type=str, default="./wafer_model_output", help="예측 결과(csv) 저장 위치")
    parser.add_argument("--output-name", type=str, default="predictions.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.label_mapping, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    print(f"[1/3] 예측 데이터 로딩: {args.data}")
    df = load_predict_dataframe(args.data)
    print(f"    웨이퍼맵 {len(df):,}장")

    print(f"[2/3] 모델 로딩: {args.model}")
    from tensorflow.keras.models import load_model

    model = load_model(args.model)

    print("[3/3] 예측 수행 (청크 단위)")
    pred_idx, confidence = predict_in_chunks(
        df["waferMap"].to_numpy(), model, args.img_size, args.chunk_size,
    )
    pred_label = [idx_to_label[i] for i in pred_idx]

    result = pd.DataFrame({
        "sourceFile": df["sourceFile"],
        "predictedLabel": pred_label,
        "confidence": confidence,
        "trueLabel": df["trueLabel"],
    })

    known = result["trueLabel"].notna()
    if known.any():
        acc = (result.loc[known, "trueLabel"] == result.loc[known, "predictedLabel"]).mean()
        print(f"    정답 라벨이 있는 {int(known.sum()):,}장 기준 정확도: {acc:.3f}")

    out_path = os.path.join(args.output_dir, args.output_name)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n완료. 예측 결과 저장 위치: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
