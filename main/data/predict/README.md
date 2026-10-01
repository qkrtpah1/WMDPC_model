# 예측(추론)용 데이터 폴더

라벨을 모르는 새 웨이퍼맵을 예측하려면, `LSWMD.pkl`과 동일한 형식(`waferMap` 컬럼 필수, `failureType`은 없어도 됨)의
`.pkl` 파일을 이 폴더(하위 폴더 포함)에 넣으세요.

예측 실행:

```bash
python predict.py --data data/predict --model wafer_model_output/wafer_defect_cnn.keras --label-mapping wafer_model_output/label_mapping.json
```

결과는 `wafer_model_output/predictions.csv`에 저장됩니다.
