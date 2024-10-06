
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict, Union

import dataloader.csv_reader as dl
import pandas as pd
from graph_plotter import plot_multiple_products


def main():
 
    # 필요한 품목 및 세부 정보
    products_info: Dict[str, Dict[str, Union[str, List[str]]]] = {
        # '건고추': {'품종명': '화건', '거래단위': '30 kg', '등급': '상품'},
        # '사과': {'품종명': ['홍로', '후지'], '거래단위': '10 개', '등급': '상품'},
        # '감자': {'품종명': '감자 수미', '거래단위': '20키로상자', '등급': '상'},
        # '배': {'품종명': '신고', '거래단위': '10 개', '등급': '상품'},
        # '깐마늘(국산)': {'품종명': '깐마늘(국산)', '거래단위': '20 kg', '등급': '상품'},
        '무': {'품종명': '무', '거래단위': '20키로상자', '등급': '상'},
        # '상추': {'품종명': '청', '거래단위': '100 g', '등급': '상품'},
        '배추': {'품종명': '배추', '거래단위': '10키로망대', '등급': '상'},
        # '양파': {'품종명': '양파', '거래단위': '1키로', '등급': '상'},
        # '대파': {'품종명': '대파(일반)', '거래단위': '1키로단', '등급': '상'}
    }
    
    file_path = "./data/train/train.csv" # CSV 파일 경로를 지정하세요
    
    # 데이터 로드
    result = dl.filter_agri_products(file_path, products_info)

    # 안쓰고 싶은 데이터 삭제
    # del result[3]

    # 각 DataFrame 출력
    for i, df in enumerate(result):
        print(f"\n{list(products_info.keys())[i]} DataFrame:")
        print(df)
    
    # 데이터 시각화
    # 여러 제품 비교 그래프
    plot_multiple_products(result, list(products_info.keys()), "시점", "평균가격(원)", "제품 가격 비교")


    # 데이터 전처리

    # 모델에 넣을 데이터 미리 저장

    # 모델 학습

    # 예측

    # 결과 시각화

    # 모델 평가

    # 평가 결과 저장

    # 예측 결과 저장

if __name__ == "__main__":
    main()