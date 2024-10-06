import pandas as pd
from prophet import Prophet
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from agripriceforecast.config_reader import read_config

config = read_config()
font_path = config['Fonts']['korean_font']
# 한글 폰트 설정
# font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # macOS 기본 한글 폰트
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def train_prophet_model(df: pd.DataFrame, date_column: str, price_column: str) -> Prophet:
    """
    Prophet 모델을 학습하는 함수

    :param df: 학습에 사용할 DataFrame
    :param date_column: 날짜 열의 이름
    :param price_column: 가격 열의 이름
    :return: 학습된 Prophet 모델
    """
    # Prophet 모델은 'ds'와 'y' 열을 사용합니다
    model_df = df.rename(columns={date_column: 'ds', price_column: 'y'})
    
    # Prophet 모델 초기화 및 학습
    model = Prophet()
    model.fit(model_df)
    
    return model

def train_multiple_prophet_models(filtered_dfs: List[pd.DataFrame], 
                                  product_names: List[str], 
                                  date_column: str, 
                                  price_column: str) -> Dict[str, Prophet]:
    """
    여러 제품에 대해 Prophet 모델을 학습하는 함수

    :param filtered_dfs: 제품별로 필터링된 DataFrame 리스트
    :param product_names: 제품 이름 리스트
    :param date_column: 날짜 열의 이름
    :param price_column: 가격 열의 이름
    :return: 제품 이름을 키로, 학습된 Prophet 모델을 값으로 하는 딕셔너리
    """
    models = {}
    for df, product_name in zip(filtered_dfs, product_names):
        print(f"{product_name} 모델 학습 중...")
        model = train_prophet_model(df, date_column, price_column)
        models[product_name] = model
        print(f"{product_name} 모델 학습 완료")
    
    return models


def predict_future_prices(model: Prophet, periods: int = 30) -> pd.DataFrame:
    """
    학습된 Prophet 모델을 사용하여 미래 가격을 예측하는 함수

    :param model: 학습된 Prophet 모델
    :param periods: 예측할 기간 (일 단위)
    :return: 예측 결과가 담긴 DataFrame
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def predict_multiple_products(models: Dict[str, Prophet], periods: int = 30) -> Dict[str, pd.DataFrame]:
    """
    여러 제품에 대해 미래 가격을 예측하는 함수

    :param models: 제품 이름을 키로, 학습된 Prophet 모델을 값으로 하는 딕셔너리
    :param periods: 예측할 기간 (일 단위)
    :return: 제품 이름을 키로, 예측 결과 DataFrame을 값으로 하는 딕셔너리
    """
    predictions = {}
    for product_name, model in models.items():
        print(f"{product_name} 가격 예측 중...")
        forecast = predict_future_prices(model, periods)
        predictions[product_name] = forecast
        print(f"{product_name} 가격 예측 완료")
    
    return predictions

def plot_forecast(forecasts: Dict[str, pd.DataFrame], actual_data: Dict[str, pd.DataFrame], 
                  date_column: str, price_column: str, output_dir: Optional[str] = None):
    """
    실제 데이터와 예측 결과를 함께 그래프로 표시합니다.

    :param forecasts: 제품 이름을 키로, 예측 결과 DataFrame을 값으로 하는 딕셔너리
    :param actual_data: 제품 이름을 키로, 실제 데이터 DataFrame을 값으로 하는 딕셔너리
    :param date_column: 날짜 열의 이름
    :param price_column: 가격 열의 이름
    :param output_dir: 그래프를 저장할 디렉토리 경로 (선택적)
    """
    for product, forecast in forecasts.items():
        plt.figure(figsize=(12, 6))
        
        # 실제 데이터 플롯
        actual = actual_data[product]
        plt.plot(actual[date_column], actual[price_column], label='실제 가격', color='blue')
        
        # 예측 데이터 플롯
        plt.plot(forecast['ds'], forecast['yhat'], label='예측 가격', color='red')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                         color='red', alpha=0.2, label='예측 구간')
        
        plt.title(f'{product} 가격 예측')
        plt.xlabel('날짜')
        plt.ylabel('가격')
        plt.legend()
        
        # x축 레이블 회전
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if output_dir:
            # 그래프 저장
            plt.savefig(f'{output_dir}/{product}_forecast.png')
            print(f"{product} 예측 그래프가 {output_dir} 폴더에 저장되었습니다.")
        else:
            plt.show()
        
        plt.close()

    if output_dir:
        print(f"모든 예측 그래프가 {output_dir} 폴더에 저장되었습니다.")
    else:
        print("모든 예측 그래프가 화면에 표시되었습니다.")

def prepare_combined_data(filtered_dfs: List[pd.DataFrame], product_names: List[str], date_column: str, price_column: str) -> pd.DataFrame:
    combined_df = pd.DataFrame()
    for df, product in zip(filtered_dfs, product_names):
        temp_df = df[[date_column, price_column]].copy()
        temp_df['product'] = product
        combined_df = pd.concat([combined_df, temp_df])
    
    combined_df = combined_df.rename(columns={date_column: 'ds', price_column: 'y'})
    return combined_df

def train_combined_prophet_model(combined_df: pd.DataFrame) -> Prophet:
    model = Prophet()
    for product in combined_df['product'].unique():
        model.add_regressor(f'product_{product}')
    
    for product in combined_df['product'].unique():
        combined_df[f'product_{product}'] = (combined_df['product'] == product).astype(int)
    
    model.fit(combined_df)
    return model


def predict_multiple_products2(model: Prophet, combined_df: pd.DataFrame, periods: int = 30) -> Dict[str, pd.DataFrame]:
    predictions = {}
    last_date = combined_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    
    for product in combined_df['product'].unique():
        future = pd.DataFrame({'ds': future_dates})
        for other_product in combined_df['product'].unique():
            future[f'product_{other_product}'] = 1 if other_product == product else 0
        
        forecast = model.predict(future)
        predictions[product] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    return predictions

def plot_forecast2(predictions: Dict[str, pd.DataFrame], actual_data: Dict[str, pd.DataFrame], date_column: str, price_column: str, output_dir: Optional[str] = None):
    n_products = len(predictions)
    n_cols = 2
    n_rows = (n_products + 1) // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axs = axs.flatten()

    for i, (product, forecast) in enumerate(predictions.items()):
        ax = axs[i]
        actual = actual_data[product]
        
        # 마지막 30일의 실제 데이터만 표시
        last_30_days_actual = actual.tail(30)
        ax.plot(last_30_days_actual[date_column], last_30_days_actual[price_column], label='실제 가격', color='blue')
        
        ax.plot(forecast['ds'], forecast['yhat'], label='예측 가격', color='red')
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2, label='예측 구간')
        
        ax.set_title(f'{product} 가격 예측 (마지막 30일)')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f'{output_dir}/combined_forecast.png')
        print(f"예측 그래프가 {output_dir}/combined_forecast.png 에 저장되었습니다.")
    else:
        plt.show()

    plt.close()