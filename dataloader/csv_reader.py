

import pandas as pd
import os
from typing import List, Dict, Union


'''
    품목명        품종명        거래단위      등급
1   건고추        화건          30 kg        상품
2   사과          홍로,후지      10 개        상품
3   감자          감자 수미      20키로상자    상
4   배            신고          10 개        상품
5   깐마늘(국산)   깐마늘(국산)   20 kg        상품
6   무            무            20키로상자    상
7   상추          청            100 g        상품
8   배추          배추          10키로망대    상
9   양파          양파          1키로         상
10  대파          대파(일반)     1키로단       상
'''

def read_train_csv_file_all(file_path: str, ) -> pd.DataFrame:
    return pd.read_csv(file_path)


def filter_agri_products(csv_file_path: str, products_info: Dict[str, Dict[str, Union[str, List[str]]]]) -> List[pd.DataFrame]:
    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)
    
    # 각 품목별 DataFrame을 저장할 리스트
    dataframes: List[pd.DataFrame] = []
    
    for product, info in products_info.items():
        # 품목명으로 필터링
        filtered_df = df[df['품목명'] == product]
        
        # 품종명 필터링 (리스트인 경우 or 조건 적용)
        if isinstance(info['품종명'], list):
            filtered_df = filtered_df[filtered_df['품종명'].isin(info['품종명'])]
        else:
            filtered_df = filtered_df[filtered_df['품종명'] == info['품종명']]
        
        # 거래단위와 등급으로 필터링
        filtered_df = filtered_df[
            (filtered_df['거래단위'] == info['거래단위']) & 
            (filtered_df['등급'] == info['등급'])
        ]
        
        # 필요한 열만 선택
        # filtered_df = filtered_df[['품목명', '품종명', '거래단위', '등급']]
        
        # 인덱스 재설정
        filtered_df = filtered_df.reset_index(drop=True)
        filtered_df.index += 1  # 인덱스를 1부터 시작하도록 설정
        
        dataframes.append(filtered_df)
    
    return dataframes
