import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
# import tensorflow_probability as tfp
# import tf_keras

# tfd = tfp.distributions
# tfpl = tfp.layers

print(f"TensorFlow version: {tf.__version__}")
# print(f"TensorFlow Probability version: {tfp.__version__}")
# print(f"TF-Keras version: {tf_keras.__version__}")

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

def create_mc_dropout_lstm_model(time_step, n_features, dropout_rate=0.2):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(time_step, n_features)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(n_features)
    ])
    return model

def train_lstm_model(data, product_names, time_step=60, epochs=100, batch_size=32, dropout_rate=0.2, learning_rate=0.001):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    
    X, y = create_dataset(data_scaled, time_step)
    
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    model = create_mc_dropout_lstm_model(time_step, len(product_names), dropout_rate)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping],
                        verbose=1)
    
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    return model, scaler, history

def predict_prices_with_uncertainty(model, scaler, last_sequence, num_samples=100):
    X = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    predictions = []
    for _ in range(num_samples):
        prediction = model(X, training=True)  # Enable dropout during prediction
        predictions.append(prediction)
    
    predictions = np.array(predictions).squeeze()
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)
    
    mean_unscaled = scaler.inverse_transform(mean_prediction.reshape(1, -1))
    std_unscaled = scaler.inverse_transform(std_prediction.reshape(1, -1))
    
    return mean_unscaled[0], std_unscaled[0]

def predict_prices(model, scaler, last_sequence, num_samples=100):
    X = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    predictions = model(X)
    samples = predictions.sample(num_samples)
    mean_prediction = tf.reduce_mean(samples, axis=0)
    std_prediction = tf.math.reduce_std(samples, axis=0)
    
    mean_unscaled = scaler.inverse_transform(mean_prediction.numpy().reshape(1, -1))
    std_unscaled = scaler.inverse_transform(std_prediction.numpy().reshape(1, -1))
    
    return mean_unscaled[0], std_unscaled[0]

def train_and_predict(filtered_dfs, products_info, test_dfs, date_column, price_column, test_index=0, lstm_hyperparams = None):
    combined_data = np.column_stack([df[price_column].values for df in filtered_dfs])
    product_names = list(products_info.keys())

    if lstm_hyperparams is None:
        lstm_hyperparams = {}
    
    model, scaler, history = train_lstm_model(combined_data, product_names, **lstm_hyperparams)
    
    predictions = {product: {'mean': [], 'std': []} for product in product_names}
    dates = []

    test_df = test_dfs[test_index]
    test_data = np.column_stack([test_df[product][price_column].values for product in product_names])
    test_data_scaled = scaler.transform(test_data)
    
    for i in range(len(test_data_scaled)):
        if i < 60:
            last_sequence = np.pad(test_data_scaled[:i+1], ((60-i-1, 0), (0, 0)), mode='edge')
        else:
            last_sequence = test_data_scaled[i-60:i]
        
        predicted_mean, predicted_std = predict_prices_with_uncertainty(model, scaler, last_sequence)
        for j, product in enumerate(product_names):
            predictions[product]['mean'].append(predicted_mean[j])
            predictions[product]['std'].append(predicted_std[j])
        
        dates.append(test_df[product_names[0]][date_column].iloc[i])

    for product in product_names:
        predictions[product] = pd.DataFrame({
            'ds': dates, 
            'yhat': predictions[product]['mean'],
            'yhat_lower': np.array(predictions[product]['mean']) - 2 * np.array(predictions[product]['std']),
            'yhat_upper': np.array(predictions[product]['mean']) + 2 * np.array(predictions[product]['std'])
        })
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
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2, label='95% 신뢰 구간')
        
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
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2, label='95% 신뢰 구간')
        
        ax.set_title(f'{product} 가격 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        # y축 범위 설정
        y_min = forecast['yhat_lower'].min() * 0.9
        y_max = forecast['yhat_upper'].max() * 1.1
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