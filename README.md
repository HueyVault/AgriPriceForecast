
# AGRIPRICEFORECAST (농산물 가격 예측 시스템)

## 프로젝트 개요

AGRIPRICEFORECAST는 다양한 머신러닝 모델을 사용하여 농산물 가격을 예측하는 시스템입니다. 
이 프로젝트는 2018년부터 2021년까지의 데이터를 기반으로 미래 농산물 가격을 예측합니다.

## 프로젝트 구조
```
\AGRIPRICEFORECAST
│  .gitignore
│  config.ini.sample
│  README.md
│  requirements.txt
│
├─agripriceforecast
│      config_reader.py
│      graph_plotter.py
│      main.py
│      __init__.py
│
├─data
│  │  sample_submission.csv       
│  │
│  ├─test
│  │  │  TEST_00.csv
│  │  │  TEST_01.csv
│  │  │  TEST_02.csv
│  │  │  TEST_03.csv
│  │  │  TEST_04.csv
│  │  │  TEST_05.csv
│  │  │  TEST_06.csv
│  │  │  TEST_07.csv
│  │  │  TEST_08.csv
│  │  │  TEST_09.csv
│  │  │  TEST_10.csv
│  │  │  TEST_11.csv
│  │  │  TEST_12.csv
│  │  │  TEST_13.csv
│  │  │  TEST_14.csv
│  │  │  TEST_15.csv
│  │  │  TEST_16.csv
│  │  │  TEST_17.csv
│  │  │  TEST_18.csv
│  │  │  TEST_19.csv
│  │  │  TEST_20.csv
│  │  │  TEST_21.csv
│  │  │  TEST_22.csv
│  │  │  TEST_23.csv
│  │  │  TEST_24.csv
│  │  │
│  │  └─meta
│  │          TEST_산지공판장_00.csv
│  │          TEST_산지공판장_01.csv
│  │          TEST_산지공판장_02.csv
│  │          TEST_산지공판장_03.csv
│  │          TEST_산지공판장_04.csv
│  │          TEST_산지공판장_05.csv
│  │          TEST_산지공판장_06.csv
│  │          TEST_산지공판장_07.csv
│  │          TEST_산지공판장_08.csv
│  │          TEST_산지공판장_09.csv
│  │          TEST_산지공판장_10.csv
│  │          TEST_산지공판장_11.csv
│  │          TEST_산지공판장_12.csv
│  │          TEST_산지공판장_13.csv
│  │          TEST_산지공판장_14.csv
│  │          TEST_산지공판장_15.csv
│  │          TEST_산지공판장_16.csv
│  │          TEST_산지공판장_17.csv
│  │          TEST_산지공판장_18.csv
│  │          TEST_산지공판장_19.csv
│  │          TEST_산지공판장_20.csv
│  │          TEST_산지공판장_21.csv
│  │          TEST_산지공판장_22.csv
│  │          TEST_산지공판장_23.csv
│  │          TEST_산지공판장_24.csv
│  │          TEST_전국도매_00.csv
│  │          TEST_전국도매_01.csv
│  │          TEST_전국도매_02.csv
│  │          TEST_전국도매_03.csv
│  │          TEST_전국도매_04.csv
│  │          TEST_전국도매_05.csv
│  │          TEST_전국도매_06.csv
│  │          TEST_전국도매_07.csv
│  │          TEST_전국도매_08.csv
│  │          TEST_전국도매_09.csv
│  │          TEST_전국도매_10.csv
│  │          TEST_전국도매_11.csv
│  │          TEST_전국도매_12.csv
│  │          TEST_전국도매_13.csv
│  │          TEST_전국도매_14.csv
│  │          TEST_전국도매_15.csv
│  │          TEST_전국도매_16.csv
│  │          TEST_전국도매_17.csv
│  │          TEST_전국도매_18.csv
│  │          TEST_전국도매_19.csv
│  │          TEST_전국도매_20.csv
│  │          TEST_전국도매_21.csv
│  │          TEST_전국도매_22.csv
│  │          TEST_전국도매_23.csv
│  │          TEST_전국도매_24.csv
│  │
│  └─train
│      │  train.csv
│      │
│      └─meta
│              TRAIN_산지공판장_2018-2021.csv
│              TRAIN_전국도매_2018-2021.csv
│
├─dataloader
│      csv_reader.py
│      __init__.py
│
└─models
        common.py
        lightgbm_price_predictor.py
        LSTM_price_predicator.py  
        LSTM_price_predicator_multi.py
        prophet_price_predictor.py        
        xgboost_price_predictor.py        
        __init__.py
```

## 주요 구성 요소

1. **agripriceforecast**: 주요 애플리케이션 코드
   - `config_reader.py`: 설정 파일 읽기
   - `graph_plotter.py`: 그래프 생성
   - `main.py`: 메인 실행 파일

2. **data**: 훈련 및 테스트 데이터셋
   - `test`: 테스트 데이터 및 메타 데이터
   - `train`: 훈련 데이터 (2018-2021) 및 메타 데이터

3. **dataloader**: 데이터 로딩 유틸리티
   - `csv_reader.py`: CSV 파일 읽기 기능

4. **models**: 다양한 예측 모델 구현
   - LightGBM
   - LSTM (단일 및 다중)
   - Prophet
   - XGBoost

## 설정 파일

`config.ini` 파일에서 다양한 설정을 조정할 수 있습니다. 예를 들어, 모델 선택, 데이터 경로 및 기타 매개변수를 설정할 수 있습니다.
```
[Paths]
train_data = ./data/train/train.csv
test_data_dir = ./data/test/
output_dir = ./data/output/train_data

[Fonts]
korean_font = /System/Library/Fonts/AppleSDGothicNeo.ttc
```

## 실행 방법

1. 프로젝트 디렉토리로 이동합니다.
2. 필요한 모듈을 설치합니다.

```
pip install -r requirements.txt
```

3. 프로젝트를 실행합니다.

```
python main.py
```

