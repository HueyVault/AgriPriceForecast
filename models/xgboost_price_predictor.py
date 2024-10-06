import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib.dates as mdates

print(xgb.__version__)

def convert_to_date(date_str):
    if isinstance(date_str, pd.Timestamp):
        return date_str
    
    if not isinstance(date_str, str):
        return pd.NaT
    
    if date_str.startswith('T-'):
        days_ago = int(date_str[2:-1])
        return datetime.now().date() - timedelta(days=days_ago*10)
    
    if len(date_str) == 8 and date_str[:6].isdigit():
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
    
    return pd.NaT



def create_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    시계열 데이터에 대한 특성을 생성합니다.
    """
    df['date'] = df[date_column].apply(convert_to_date)
    df = df.dropna(subset=['date'])  # NaT 값을 가진 행 제거
    
    # date 열이 datetime 형식인지 확인
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # 변환 실패한 행 제거
    
    if len(df) == 0:
        raise ValueError("모든 날짜 데이터가 유효하지 않습니다.")
    
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    
    return df

def train_xgboost_model(df: pd.DataFrame, price_column: str, test_size: float = 0.2) -> tuple:
    """
    XGBoost 모델을 학습합니다.
    """
    features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    X = df[features]
    y = df[price_column]
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        random_state=42,
        early_stopping_rounds=50
    )
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    return model, X_test, y_test

def predict_prices(model: XGBRegressor, future_features: pd.DataFrame, scaler: MinMaxScaler = None) -> np.ndarray:
    """
    학습된 모델을 사용하여 미래 가격을 예측합니다.
    """
    predictions = model.predict(future_features)
    
    if scaler:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    return predictions

def train_and_predict_xgboost(filtered_dfs: List[pd.DataFrame], test_dataframes: List[Dict[str, pd.DataFrame]], 
                              products_info: Dict[str, Dict[str, Union[str, List[str]]]], 
                              date_column: str, price_column: str) -> Dict[str, List[pd.DataFrame]]:
    """
    여러 제품에 대해 XGBoost 모델을 학습하고 예측합니다.
    """
    predictions = {product: [] for product in products_info.keys()}
    models = {}
    scalers = {}
    
    for product, df in zip(products_info.keys(), filtered_dfs):
        print(f"{product} 모델 학습 중...")
        
        df = df.copy()  # 복사본 생성
        df = df.rename(columns={date_column: 'date', price_column: 'price'})
        try:
            df = create_features(df, 'date')
        except ValueError as e:
            print(f"{product} 데이터 처리 실패: {str(e)}")
            continue
        
        scaler = MinMaxScaler()
        df['price'] = scaler.fit_transform(df[['price']])
        
        model, X_test, y_test = train_xgboost_model(df, 'price')
        
        # 모델 평가
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{product} 모델 MSE: {mse}")
        
        models[product] = model
        scalers[product] = scaler
    
    # 테스트 데이터에 대한 예측
    for test_df_dict in test_dataframes:
        product_predictions = {}
        for product, test_df in test_df_dict.items():
            if product not in models:
                print(f"{product}에 대한 모델이 없습니다. 건너뜁니다.")
                continue
            test_df = test_df.copy()  # 복사본 생성
            test_df = test_df.rename(columns={date_column: 'date', price_column: 'price'})
            try:
                test_df = create_features(test_df, 'date')
            except ValueError as e:
                print(f"{product} 테스트 데이터 처리 실패: {str(e)}")
                continue
            
            features = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
            X_test = test_df[features]
            
            predictions_values = predict_prices(models[product], X_test, scalers[product])
            
            product_predictions[product] = pd.DataFrame({
                'ds': test_df['date'],
                'yhat': predictions_values
            })
        
        for product in products_info.keys():
            if product in product_predictions:
                predictions[product].append(product_predictions[product])
            else:
                print(f"{product}에 대한 예측이 생성되지 않았습니다.")
    
    return predictions

def plot_xgboost_forecast(predictions: Dict[str, List[pd.DataFrame]], actual_data: List[Dict[str, pd.DataFrame]], 
                          date_column: str, price_column: str, output_dir: str = None):
    """
    XGBoost 모델의 예측 결과와 실제 데이터를 비교하여 그래프로 시각화합니다.
    """
    n_products = len(predictions)
    n_cols = 2
    n_rows = (n_products + 1) // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axs = axs.flatten()

    for i, (product, pred_list) in enumerate(predictions.items()):
        ax = axs[i]
        
        # 모든 테스트 데이터셋에 대한 예측을 플롯
        for j, pred_df in enumerate(pred_list):
            ax.plot(pred_df['ds'].astype(str), pred_df['yhat'], label=f'예측 {j+1}', alpha=0.7)
        
        # 실제 데이터 플롯 (첫 번째 테스트 데이터셋 사용)
        actual = actual_data[0][product]
        ax.plot(actual[date_column].astype(str), actual[price_column], label='실제 가격', color='black', linewidth=2)
        
        ax.set_title(f'{product} 가격 예측 (XGBoost)')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        ax.legend()
        
        # x축 레이블 포맷 설정
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f'{output_dir}/xgboost_forecast.png')
        print(f"XGBoost 예측 그래프가 {output_dir}/xgboost_forecast.png 에 저장되었습니다.")
    else:
        plt.show()

    plt.close()