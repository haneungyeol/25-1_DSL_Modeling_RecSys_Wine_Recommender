import pandas as pd
from kgat_config import WINE_DATA_PATH

def load_wine_data(path: str = WINE_DATA_PATH) -> pd.DataFrame:
    """
    와인 데이터셋을 로드하고 DataFrame으로 반환합니다.
    """
    wine_data = pd.read_csv(path)
    print("와인 데이터 미리보기:")
    print(wine_data.head())
    print("Wine dataset shape:", wine_data.shape)
    return wine_data

def build_kg(wine_data: pd.DataFrame, selected_columns: list = None):
    """
    각 와인을 아이템으로 보고, 선택한 속성들에 대해 KG 이웃 정보를 구성합니다.
    Returns:
      - kg_neighbors: 각 아이템에 연결된 이웃 엔티티 id 리스트
      - attribute_to_id: 속성값을 고유 엔티티 id에 매핑한 딕셔너리
    """
    if selected_columns is None:
        selected_columns = ["Flavor Group 1", "Current Price_Quintile", "Region", "Winery", "Wine style"]
    
    num_items = wine_data.shape[0]
    kg_neighbors = []         # 각 아이템에 대한 이웃 엔티티 id 리스트
    attribute_to_id = {}      # 속성값을 고유 엔티티 id에 매핑
    current_entity_id = num_items  # 아이템 인덱스 이후부터 엔티티 id 할당
    
    for idx, row in wine_data.iterrows():
        neighbors = []
        for col in selected_columns:
            attr_value = row[col]
            if pd.isnull(attr_value):
                continue
            # "속성명:값" 형식의 key 생성
            key = f"{col}:{attr_value}"
            if key not in attribute_to_id:
                attribute_to_id[key] = current_entity_id
                current_entity_id += 1
            neighbors.append(attribute_to_id[key])
        kg_neighbors.append(neighbors)
    
    print("전체 아이템 개수:", num_items)
    print("생성된 속성(엔티티) 수:", current_entity_id - num_items)
    return kg_neighbors, attribute_to_id
