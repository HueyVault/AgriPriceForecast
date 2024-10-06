import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

def create_dataset(dataset, time_step=1):
    """
    시계열 데이터셋을 생성합니다.

    :param dataset: 원본 데이터셋
    :param time_step: 시퀀스 길이
    :return: 입력 시퀀스와 타겟 값
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i+time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, :])
    return np.array(dataX), np.array(dataY)

def train_lstm_model(data, product_names, time_step=60, epochs=100, batch_size=32):
    """
    여러 품목의 데이터를 하나의 LSTM 모델로 학습합니다.

    :param data: 모든 품목의 가격 데이터가 포함된 2D 배열
    :param product_names: 품목 이름 리스트
    :param time_step: 시퀀스 길이
    :param epochs: 학습 에포크 수
    :param batch_size: 배치 크기
    :return: 학습된 모델, 스케일러, 학습 히스토리
    """
    # 데이터 전처리
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    
    # 데이터셋 생성
    X, y = create_dataset(data_scaled, time_step)
    
    # 학습 데이터와 검증 데이터 분리
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # LSTM 모델 생성
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(time_step, len(product_names))),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(len(product_names))
    ])
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    # 조기 종료 콜백 추가
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 모델 학습
    for epoch in range(epochs):
        history = model.fit(X_train, y_train, 
                            epochs=1, 
                            batch_size=batch_size, 
                            validation_data=(X_val, y_val),
                            verbose=0)
        
        # 각 에포크마다 손실 값 확인
        train_loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"Warning: 에포크 {epoch+1}에서 NaN 손실 발생. 학습 중단.")
            break
        
        # print(f"에포크 {epoch+1}/{epochs}, 훈련 손실: {train_loss:.4f}, 검증 손실: {val_loss:.4f}")

    
    # 학습 결과 평가
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # # 학습 곡선 그리기
    # plt.figure(figsize=(12, 6))
    # plt.plot(history.history['loss'], label='Train Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    
    return model, scaler, history

def predict_prices(model, scaler, last_sequence):
    """
    학습된 모델을 사용하여 다음 가격을 예측합니다.

    :param model: 학습된 LSTM 모델
    :param scaler: 데이터 스케일러
    :param last_sequence: 마지막 시퀀스 데이터
    :return: 예측된 다음 가격
    """
    X = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    predicted = model.predict(X)
    print(predicted)
    return scaler.inverse_transform(predicted)[0]

def train_and_predict(filtered_dfs, products_info, test_dfs, date_column, price_column, test_index=0):
    """
    모든 품목의 데이터를 하나의 LSTM 모델로 학습하고 특정 테스트 데이터에 대한 예측을 수행합니다.

    :param filtered_dfs: 학습용 데이터프레임 리스트
    :param products_info: 품목 정보 딕셔너리
    :param test_dfs: 테스트 데이터프레임 리스트
    :param date_column: 날짜 열 이름
    :param price_column: 가격 열 이름
    :param test_index: 사용할 테스트 데이터의 인덱스
    :return: 예측 결과 딕셔너리
    """
    # 모든 품목의 가격 데이터를 하나의 2D 배열로 결합
    combined_data = np.column_stack([df[price_column].values for df in filtered_dfs])
    product_names = list(products_info.keys())

    # LSTM 모델 학습
    model, scaler, history = train_lstm_model(combined_data, product_names)

    # 특정 테스트 데이터에 대한 예측
    predictions = {product: [] for product in product_names}
    dates = []

    test_df = test_dfs[test_index]
    test_data = np.column_stack([test_df[product][price_column].values for product in product_names])
    test_data_scaled = scaler.transform(test_data)
    for i in range(len(test_data_scaled)):
        if i < 60:
            last_sequence = np.pad(test_data_scaled[:i+1], ((60-i-1, 0), (0, 0)), mode='edge')
        else:
            last_sequence = test_data_scaled[i-60:i]
        
        predicted_price = predict_prices(model, scaler, last_sequence)
        for j, product in enumerate(product_names):
            predictions[product].append(predicted_price[j])
        
        dates.append(test_df[product_names[0]][date_column].iloc[i])

    # 예측 결과를 DataFrame으로 변환
    for product in product_names:
        predictions[product] = pd.DataFrame({'ds': dates, 'yhat': predictions[product]})
        # NaN 또는 Inf 값 확인
        invalid_values = predictions[product]['yhat'].isnull() | np.isinf(predictions[product]['yhat'])
        if invalid_values.any():
            print(f"Warning: {product}에 대한 예측에 {invalid_values.sum()}개의 유효하지 않은 값이 포함되어 있습니다.")

    return predictions

def plot_lstm_forecast_with_test(predictions, test_dfs, date_column, price_column, test_index=0, output_dir=None):
    """
    LSTM 모델의 예측 결과와 실제 데이터를 비교하여 그래프로 시각화합니다.

    :param predictions: 예측 결과 딕셔너리
    :param test_dfs: 테스트 데이터프레임 리스트
    :param date_column: 날짜 열 이름
    :param price_column: 가격 열 이름
    :param test_index: 사용할 테스트 데이터의 인덱스
    :param output_dir: 그래프 저장 디렉토리 (선택적)
    """
    n_products = len(predictions)
    n_cols = 2
    n_rows = (n_products + 1) // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axs = axs.flatten()

    test_df = test_dfs[test_index]  # 지정된 인덱스의 테스트 데이터프레임 사용

    for i, (product, forecast) in enumerate(predictions.items()):
        ax = axs[i]
        actual = test_df[product]

        ax.plot(actual[date_column], actual[price_column], label='실제 가격', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='예측 가격', color='red')
        
        ax.set_title(f'{product} 가격 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f'{output_dir}/lstm_forecast_test_{test_index}.png')
        print(f"LSTM 테스트 데이터 예측 그래프가 {output_dir}/lstm_forecast_test_{test_index}.png 에 저장되었습니다.")
    else:
        plt.show()

    plt.close()


def plot_lstm_forecast_only(predictions, date_column, price_column, output_dir=None):
    """
    LSTM 모델의 예측 결과만을 그래프로 시각화합니다.

    :param predictions: 예측 결과 딕셔너리
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

        ax.plot(forecast['ds'], forecast['yhat'], label='예측 가격', color='red')
        
        ax.set_title(f'{product} 가격 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        # y축 범위 설정
        y_min = forecast['yhat'].min() * 0.9  # 최소값의 90%
        y_max = forecast['yhat'].max() * 1.1  # 최대값의 110%
        ax.set_ylim(y_min, y_max)

    # 사용하지 않는 서브플롯 제거
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f'{output_dir}/lstm_forecast_only.png')
        print(f"LSTM 예측 결과 그래프가 {output_dir}/lstm_forecast_only.png 에 저장되었습니다.")
    else:
        plt.show()

    plt.close()