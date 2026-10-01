"""
LSWMD.pkl에서 라벨(failureType)이 없는 웨이퍼맵만 뽑아 별도 pkl로 저장한다.

WM-811K 전체 811,457장 중 172,950장만 9개 클래스로 라벨링되어 있고,
나머지 약 638,000장은 실제로 분류되지 않은 "라벨 없는" 웨이퍼맵이다.
predict.py가 상정하는 "불량 유형을 모르는 새 웨이퍼맵"과 정확히 같은 데이터라서,
이 서브셋을 data/predict/로 옮겨 추론용으로 쓴다.
"""

import os

import pandas as pd

from wafer_defect_classifier import CLASS_NAMES, _flatten_label, load_dataframe


def main():
    src = "data/train/LSWMD.pkl"
    dst = "data/predict/LSWMD_unlabeled.pkl"

    # load_dataframe 내부와 동일한 구버전 pandas 호환 로더를 쓰기 위해
    # 비공개 헬퍼를 직접 가져온다.
    import wafer_defect_classifier as wdc

    print(f"[1/3] 전체 데이터 로딩: {src}")
    try:
        df = pd.read_pickle(src)
    except (ModuleNotFoundError, UnicodeDecodeError, AttributeError):
        df = wdc._read_legacy_pickle(src)
    print(f"    전체 웨이퍼맵 수: {len(df):,}")

    print("[2/3] 라벨 없는 웨이퍼맵 필터링")
    flat_label = df["failureType"].apply(_flatten_label)
    unlabeled = df[~flat_label.isin(CLASS_NAMES)].copy()
    print(f"    라벨 있는 웨이퍼맵: {flat_label.isin(CLASS_NAMES).sum():,}")
    print(f"    라벨 없는 웨이퍼맵: {len(unlabeled):,}")

    print(f"[3/3] 저장: {dst}")
    unlabeled.to_pickle(dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"완료. {len(unlabeled):,}장, {size_mb:.1f} MB -> {os.path.abspath(dst)}")


if __name__ == "__main__":
    main()
