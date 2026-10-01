# 학습용 데이터 폴더

이 폴더에 라벨링된 웨이퍼맵 데이터셋 `LSWMD.pkl` (Kaggle "WM-811K wafer map" 데이터셋)을 넣으세요.

학습 실행:

```bash
python wafer_defect_classifier.py --data data/train/LSWMD.pkl --epochs 30 --img-size 48
```
