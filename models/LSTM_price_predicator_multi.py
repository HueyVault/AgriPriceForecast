import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
#from tensorflow import Sequential, LSTM, Dense, Adam
import matplotlib.pyplot as plt

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def train_lstm_model(data, product_name, time_step=25, epochs=100, batch_size=32):
    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data.reshape(-1,1))
    
    # 데이터셋 생성
    train_size = int(len(data_scaled) * 0.8)  # 80%를 훈련 데이터로 사용
    train_data, test_data = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]
    
    X_train, y_train = create_dataset(train_data, time_step)
    
    # 테스트 데이터 생성 (충분한 데이터가 있는 경우에만)
    if len(test_data) > time_step:
        X_test, y_test = create_dataset(test_data, time_step)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    else:
        print(f"Warning: Not enough test data for {product_name}. Using training data for validation.")
        X_test, y_test = X_train, y_train
    
    # LSTM 입력을 위해 데이터 형태 변경
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # LSTM 모델 생성
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    # 모델 학습
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=epochs, batch_size=batch_size, verbose=1)
    
    print(f"train_data shape: {train_data.shape}")
    print(f"test_data shape: {test_data.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    #print(f"train_predict shape: {train_predict.shape}")
    #print(f"test_predict shape: {test_predict.shape}")
    # 예측
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # 스케일 역변환
    test_predict = scaler.inverse_transform(test_predict)
    
    # 결과 시각화
    plt.figure(figsize=(12,6))
    plt.plot(data, label='Actual Price')
    plt.plot(range(time_step, len(train_predict)+time_step), train_predict, label='Train Predict')
    if len(test_data) > time_step:
        # 테스트 예측 결과의 시작점을 조정합니다
        test_predict_start = len(train_data) + time_step
        plt.plot(range(test_predict_start, test_predict_start + len(test_predict)), test_predict, label='Test Predict')
    plt.title(f'{product_name} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    #plt.savefig(f'{product_name}_lstm_prediction.png')
    plt.close() 
    
    return model, scaler

def predict_future(model, scaler, last_sequence, future_steps=30):
    future_predictions = []
    current_sequence = last_sequence.reshape((1, last_sequence.shape[0], 1))
    
    for _ in range(future_steps):
        next_pred = model.predict(current_sequence)[0]
        future_predictions.append(next_pred[0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

def train_multiple_lstm_models(filtered_dfs, products_info, date_column, price_column):
    """
    여러 농산물 품목에 대해 개별적인 LSTM 모델을 학습시킵니다.

    :param filtered_dfs: 품목별로 필터링된 데이터프레임 리스트
    :param products_info: 품목 정보 딕셔너리
    :param date_column: 날짜 열 이름
    :param price_column: 가격 열 이름
    :return: 학습된 모델과 스케일러의 딕셔너리
    """
    models = {}
    scalers = {}
    for product, df in zip(products_info.keys(), filtered_dfs):
        print(f"Training LSTM model for {product}...")
        data = df[price_column].values
        model, scaler = train_lstm_model(data, product)
        models[product] = model
        scalers[product] = scaler
        print(f"Finished training LSTM model for {product}")
    
    return models, scalers

def predict_multiple_products_lstm(models, scalers, filtered_dfs, products_info, date_column, price_column, future_steps=30):

    
    predictions = {}
    for product, df in zip(products_info.keys(), filtered_dfs):
        data = df[price_column].values
        last_sequence = data[-60:]  # 마지막 60개 데이터 포인트 사용
        last_sequence = scalers[product].transform(last_sequence.reshape(-1, 1))
        future_pred = predict_future(models[product], scalers[product], last_sequence, future_steps)
        
        last_date = df[date_column].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=future_steps+1)[1:]
        predictions[product] = pd.DataFrame({'ds': future_dates, 'yhat': future_pred.flatten()})
    
    return predictions

def plot_lstm_forecast(predictions, actual_data, date_column, price_column, output_dir=None):
    n_products = len(predictions)
    n_cols = 2
    n_rows = (n_products + 1) // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axs = axs.flatten()

    for i, (product, forecast) in enumerate(predictions.items()):
        ax = axs[i]
        actual = actual_data[product]
        
        ax.plot(actual[date_column], actual[price_column], label='실제 가격', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='예측 가격', color='red')
        
        ax.set_title(f'{product} 가격 예측 (LSTM)')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f'{output_dir}/lstm_forecast.png')
        print(f"LSTM 예측 그래프가 {output_dir}/lstm_forecast.png 에 저장되었습니다.")
    else:
        plt.show()

    plt.close()

def predict_multiple_products_lstm_test(models, scalers, test_dfs, products_info, date_column, price_column, index=0):
    """
    학습된 LSTM 모델을 사용하여 테스트 데이터셋에 대한 예측을 수행합니다.

    :param models: 품목별 학습된 LSTM 모델 딕셔너리
    :param scalers: 품목별 스케일러 딕셔너리
    :param test_dfs: 테스트 데이터프레임 리스트
    :param products_info: 품목 정보 딕셔너리
    :param date_column: 날짜 열 이름
    :param price_column: 가격 열 이름
    :param index: 사용할 테스트 데이터프레임의 인덱스
    :return: 품목별 예측 결과 딕셔너리
    """
    
    predictions = {}
    test_df = test_dfs[index]  # 지정된 인덱스의 테스트 데이터프레임만 사용
    
    for product, product_df in test_df.items():
        if product not in products_info:
            continue  # products_info에 없는 제품은 건너뜁니다.
        
        print(f"Predicting for {product} (test data)...")
        data = product_df[price_column].values
        dates = product_df[date_column].values
        predicted_prices = []
        print('1111data:'+str(len(data)))
        for i in range(len(data)):
            if i < 60:  # 처음 60개 데이터 포인트는 예측에 사용
                predicted_prices.append(data[i])
            else:
                last_sequence = data[i-60:i]
                last_sequence_scaled = scalers[product].transform(last_sequence.reshape(-1, 1))
                predicted_price = predict_next_price(models[product], scalers[product], last_sequence_scaled)
                predicted_prices.append(predicted_price[0][0])  # 2D 배열에서 단일 값 추출

        predictions[product] = pd.DataFrame({'ds': dates, 'yhat': predicted_prices})
        print(f"Finished predicting for {product} (test data)")
    
    return predictions

def predict_next_price(model, scaler, last_sequence):
    """
    주어진 시퀀스에 대해 다음 가격을 예측합니다.

    :param model: 학습된 LSTM 모델
    :param scaler: 데이터 스케일러
    :param last_sequence: 마지막 시퀀스 데이터
    :return: 예측된 다음 가격
    """
    # 마지막 시퀀스를 사용하여 다음 가격 예측
    X = last_sequence.reshape((1, last_sequence.shape[0], 1))
    predicted = model.predict(X)
    return scaler.inverse_transform(predicted)

def plot_lstm_forecast_with_test(predictions, actual_data, date_column, price_column, output_dir=None):
    """
    LSTM 모델의 예측 결과와 실제 데이터를 비교하여 그래프로 시각화합니다.

    :param predictions: 예측 결과 딕셔너리
    :param actual_data: 실제 데이터 리스트
    :param date_column: 날짜 열 이름
    :param price_column: 가격 열 이름
    :param output_dir: 그래프 저장 디렉토리 (선택적)
    """
    n_products = len(predictions)
    n_cols = 2
    n_rows = (n_products + 1) // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axs = axs.flatten()

    for i, (product, forecast) in enumerate(predictions.items()):
        ax = axs[i]
        actual = actual_data[i][product]  # 실제 데이터에서 해당 제품의 데이터프레임 선택

        # 상대적 시점을 숫자로 변환
        relative_dates = np.arange(len(actual))
        
        ax.plot(relative_dates, actual[price_column], label='실제 가격', color='blue')
        ax.plot(relative_dates, forecast['yhat'], label='예측 가격', color='red')
        
        ax.set_title(f'{product} 가격 예측')
        ax.set_xlabel('상대적 시점')
        ax.set_ylabel('가격')
        ax.legend()

        # x축 레이블 설정
        x_ticks = np.linspace(0, len(actual) - 1, 5, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([actual[date_column].iloc[i] for i in x_ticks])

    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f'{output_dir}/lstm_forecast_test.png')
        print(f"LSTM 테스트 데이터 예측 그래프가 {output_dir}/lstm_forecast_test.png 에 저장되었습니다.")
    else:
        plt.show()

    plt.close()