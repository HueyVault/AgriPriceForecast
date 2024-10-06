
import numpy as np

def normalized_mean_absolute_error(y_true, y_pred):
    """
    Normalized Mean Absolute Error를 계산합니다.
    
    Parameters:
    y_true (array-like): 실제 값
    y_pred (array-like): 예측 값
    
    Returns:
    float: NMAE 값
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 실제 값의 범위 계산
    range_y = np.max(y_true) - np.min(y_true)
    
    # 0으로 나누는 것을 방지
    if range_y == 0:
        return 0
    
    # MAE 계산
    mae = np.mean(np.abs(y_true - y_pred))
    
    # NMAE 계산
    nmae = mae / range_y
    
    return nmae