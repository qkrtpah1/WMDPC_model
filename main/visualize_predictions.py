"""
predict.py 결과(predictions.csv)를 README에 넣을 차트 한 장으로 요약한다.

사용법
------
    python visualize_predictions.py --predictions wafer_model_output/predictions.csv \
        --output assets/predictions_overview.png
"""

import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    parser = argparse.ArgumentParser(description="예측 결과 분포/신뢰도 차트 생성")
    parser.add_argument("--predictions", type=str, default="wafer_model_output/predictions.csv")
    parser.add_argument("--output", type=str, default="assets/predictions_overview.png")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)

    counts = df["predictedLabel"].value_counts().sort_values(ascending=True)
    mean_conf = df.groupby("predictedLabel")["confidence"].mean().reindex(counts.index)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.barh(counts.index, counts.values, color="#4C72B0")
    ax1.set_xscale("log")
    ax1.set_xlabel("예측 건수 (log scale)")
    ax1.set_title(f"클래스별 예측 분포 (전체 {len(df):,}장)")
    for bar, value in zip(bars, counts.values):
        ax1.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                  f"{value:,} ({value / len(df) * 100:.1f}%)", va="center", fontsize=8)

    colors = ["#C44E52" if c < 0.6 else "#55A868" for c in mean_conf.values]
    ax2.barh(mean_conf.index, mean_conf.values, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("평균 신뢰도(confidence)")
    ax2.set_title("클래스별 평균 신뢰도")
    for i, value in enumerate(mean_conf.values):
        ax2.text(value + 0.02, i, f"{value:.3f}", va="center", fontsize=8)

    fig.suptitle("학습된 모델의 라벨 없는 웨이퍼맵 추론 결과", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"저장 완료: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
