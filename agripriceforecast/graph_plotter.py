import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional
import matplotlib.font_manager as fm
from agripriceforecast.config_reader import read_config

# 한글 폰트 설정
config = read_config()
font_path = config['Fonts']['korean_font']
# font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # macOS 기본 한글 폰트
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def plot_multiple_products(dfs: List[pd.DataFrame], product_names: List[str], 
                           x_column: str, y_column: str, title: str, 
                           save_path: Optional[str] = None):
    """
    여러 제품의 데이터를 하나의 그래프에 그립니다.
    
    :param dfs: 데이터프레임 리스트
    :param product_names: 제품 이름 리스트
    :param x_column: x축에 사용할 열 이름
    :param y_column: y축에 사용할 열 이름
    :param title: 그래프 제목
    :param save_path: 그래프를 저장할 경로 (옵션)
    """
    plt.figure(figsize=(12, 6))
    for df, name in zip(dfs, product_names):
        sns.lineplot(data=df, x=x_column, y=y_column, label=name)
    
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
