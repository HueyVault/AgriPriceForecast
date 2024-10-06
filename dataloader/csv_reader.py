

import pandas as pd
import os
from typing import List, Dict, Union
from datetime import datetime

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


def save_dataframes_to_csv(dataframes: List[pd.DataFrame], product_names: List[str], base_output_dir: str):
    """
    DataFrame 리스트를 각각의 CSV 파일로 저장합니다.
    실행 날짜와 시간으로 폴더를 생성합니다.

    :param dataframes: 저장할 DataFrame 리스트
    :param product_names: 각 DataFrame에 해당하는 제품 이름 리스트
    :param base_output_dir: 기본 출력 디렉토리 경로
    :return: 생성된 폴더의 경로
    """
    # 현재 날짜와 시간으로 폴더명 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, current_time)

    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    for df, product_name in zip(dataframes, product_names):
        # 파일명에 사용할 수 없는 문자 제거 또는 대체
        safe_product_name = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in product_name)
        file_path = os.path.join(output_dir, f"{safe_product_name}.csv")
        
        # DataFrame을 CSV 파일로 저장
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"{product_name} 데이터가 {file_path}에 저장되었습니다.")


def read_test_files(directory_path: str, products_info: Dict[str, Dict[str, Union[str, List[str]]]], num_files: int = 25) -> List[Dict[str, pd.DataFrame]]:
    """
    지정된 디렉토리에서 TEST_XX.csv 파일들을 읽고, 각 파일에서 지정된 품목의 데이터프레임을 추출합니다.
    
    :param directory_path: CSV 파일들이 있는 디렉토리 경로
    :param products_info: 품목별 필터링 조건을 포함하는 딕셔너리
    :param num_files: 읽을 파일의 수 (기본값: 25)
    :return: 각 파일별로 지정된 품목의 데이터프레임을 포함하는 리스트
    """
    all_dataframes = []

    for filenum in range(num_files):
        filename = f'TEST_{filenum:02d}.csv'
        file_path = os.path.join(directory_path, filename)
        try:
            # CSV 파일 읽기
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # 각 품목별로 데이터프레임 추출
            file_dataframes = {}
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
                
                # 인덱스 재설정
                filtered_df = filtered_df.reset_index(drop=True)
                filtered_df.index += 1  # 인덱스를 1부터 시작하도록 설정
                
                file_dataframes[product] = filtered_df
            
            all_dataframes.append(file_dataframes)
            print(f"{filename} 파일을 성공적으로 읽었습니다.")
        except Exception as e:
            print(f"{filename} 파일 읽기 실패: {str(e)}")
            # 파일 읽기에 실패한 경우 빈 딕셔너리를 추가하여 순서 유지
            all_dataframes.append({})
    
    return all_dataframes