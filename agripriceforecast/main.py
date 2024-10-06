
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict, Union

import dataloader.csv_reader as dl
import pandas as pd
from graph_plotter import plot_multiple_products
import models.prophet_price_predictor as pp

def convert_to_date(date_str):
    year = int(date_str[:4])
    month = int(date_str[4:6])
    part = date_str[6:]
    
    if part == '상순':
        day = 5
    elif part == '중순':
        day = 15
    else:  # 하순
        day = 25
    
    return pd.to_datetime(f'{year}-{month:02d}-{day:02d}')

def train_and_predict(filtered_dfs, products_info, base_output_dir):
    # Prophet 모델 학습
    models = pp.train_multiple_prophet_models(filtered_dfs, list(products_info.keys()), "시점", "평균가격(원)")
    
    print("학습된 모델:")
    for product, model in models.items():
        print(f"- {product}")
    
    # 미래 가격 예측
    predictions = pp.predict_multiple_products(models, periods=30)  # 30일 후까지 예측

    # 예측 결과 출력
    for product, forecast in predictions.items():
        print(f"\n{product} 가격 예측 결과:")
        print(forecast.tail())  # 마지막 5일의 예측 결과만 출력

    # 결과 시각화
    # 예측 결과 시각화
    save_graphs = None # input("그래프를 파일로 저장하시겠습니까? (y/n): ").lower() == 'y'
    
    if save_graphs:
        forecast_output_dir = os.path.join(base_output_dir, "forecasts")
        os.makedirs(forecast_output_dir, exist_ok=True)
    else:
        forecast_output_dir = None
    
    # 실제 데이터를 딕셔너리 형태로 변환
    actual_data = {product: df for product, df in zip(products_info.keys(), filtered_dfs)}
    
    pp.plot_forecast(predictions, actual_data, "시점", "평균가격(원)", forecast_output_dir)

    return models, predictions

def train_and_predict_combined(filtered_dfs, products_info, base_output_dir):
    # 데이터 전처리 및 결합
    combined_df = pp.prepare_combined_data(filtered_dfs, list(products_info.keys()), "시점", "평균가격(원)")
    
    # 모델 학습
    model = pp.train_combined_prophet_model(combined_df)
    
    print("학습된 모델:")
    print(model)
    
    # 예측
    predictions = pp.predict_multiple_products2(model, combined_df, periods=30)  # 30일 후까지 예측

    # 예측 결과 출력
    for product, forecast in predictions.items():
        print(f"\n{product} 가격 예측 결과:")
        print(forecast.tail(30))  # 마지막 30일의 예측 결과만 출력

    # 결과 시각화
    forecast_output_dir = os.path.join(base_output_dir, "forecasts")
    os.makedirs(forecast_output_dir, exist_ok=True)
    
    # 실제 데이터를 딕셔너리 형태로 변환
    actual_data = {product: df for product, df in zip(products_info.keys(), filtered_dfs)}
    
    pp.plot_forecast2(predictions, actual_data, "시점", "평균가격(원)", None)

    return model, predictions


def main():
 
    # 필요한 품목 및 세부 정보
    products_info: Dict[str, Dict[str, Union[str, List[str]]]] = {
        '건고추': {'품종명': '화건', '거래단위': '30 kg', '등급': '상품'},
        '사과': {'품종명': ['홍로', '후지'], '거래단위': '10 개', '등급': '상품'},
        '감자': {'품종명': '감자 수미', '거래단위': '20키로상자', '등급': '상'},
        '배': {'품종명': '신고', '거래단위': '10 개', '등급': '상품'},
        '깐마늘(국산)': {'품종명': '깐마늘(국산)', '거래단위': '20 kg', '등급': '상품'},
        '무': {'품종명': '무', '거래단위': '20키로상자', '등급': '상'},
        '상추': {'품종명': '청', '거래단위': '100 g', '등급': '상품'},
        '배추': {'품종명': '배추', '거래단위': '10키로망대', '등급': '상'},
        '양파': {'품종명': '양파', '거래단위': '1키로', '등급': '상'},
        '대파': {'품종명': '대파(일반)', '거래단위': '1키로단', '등급': '상'}
    }
    
    file_path = "./data/train/train.csv" # CSV 파일 경로를 지정하세요
    
    # 데이터 로드
    filtered_dfs = dl.filter_agri_products(file_path, products_info)

    # 안쓰고 싶은 데이터 삭제
    # del filtered_dfs[3]

    # 각 DataFrame 출력
    for i, df in enumerate(filtered_dfs):
        df['시점'] = df['시점'].apply(convert_to_date)
        print(f"\n{list(products_info.keys())[i]} DataFrame:")
        print(df)
    
    # # 데이터 시각화
    # # 여러 제품 비교 그래프
    # plot_multiple_products(filtered_dfs, list(products_info.keys()), "시점", "평균가격(원)", "제품 가격 비교")

    # # 데이터 전처리

    # # 모델에 넣을 데이터 미리 저장
    # # filtered_dfs를 CSV 파일로 저장
    base_output_dir = "./data/output/train_data"  # CSV 파일을 저장할 디렉토리 경로
    # dl.save_dataframes_to_csv(filtered_dfs, list(products_info.keys()), base_output_dir)

    # 모델 학습
    # 데이터 전처리 및 결합
    model, predictions = train_and_predict_combined(filtered_dfs, products_info, base_output_dir)

    # Prophet 모델 학습
    # 모델 학습 및 예측
    #models, predictions = train_and_predict(filtered_dfs, products_info, base_output_dir)

    # 모델 평가

    # 평가 결과 저장

    # 예측 결과 저장

if __name__ == "__main__":
    main()