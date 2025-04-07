from kgat_data import load_wine_data, build_kg
from cf_data import load_cf_data
from kgat_model import KGATModel
from bpr_dataset import BPRDataset
from train import train_model
from kgat_config import EMBEDDING_DIM, device, POSITIVE_THRESHOLD
import pandas as pd

def main():
    # 1. 와인 데이터 로딩 및 KG 구축
    wine_data = load_wine_data()
    kg_neighbors, attribute_to_id = build_kg(wine_data)
    num_items = wine_data.shape[0]
    num_entities = num_items + len(attribute_to_id)
    print(f"전체 아이템 수: {num_items}, 전체 엔티티 수: {num_entities}")
    
    # 2. CF 데이터 로딩 (train, val, test)
    train_data, val_data, test_data = load_cf_data()
    # train_data에는 'user', 'item', 'rating' 컬럼이 있다고 가정합니다.
    user_item_interactions = {}
    for _, row in train_data.iterrows():
        user = row["user"]
        item = row["item"]
        rating = row["rating"]
        if user not in user_item_interactions:
            user_item_interactions[user] = []
        user_item_interactions[user].append((item, rating))
    
    # 3. BPRDataset 생성 (실제 사용자-아이템 상호작용 데이터를 사용)
    dataset = BPRDataset(user_item_interactions, num_items=num_items, positive_threshold=POSITIVE_THRESHOLD)
    
    # 4. KGAT 모델 인스턴스 생성
    model = KGATModel(num_items=num_items, num_entities=num_entities, embedding_dim=EMBEDDING_DIM)
    model.to(device)
    
    # 5. 모델 학습 (BPR loss 기반 학습)
    trained_model = train_model(model, dataset)
    
    # 6. (선택 사항) 테스트 데이터 평가
    # test_data를 이용한 평가 코드를 추가할 수 있습니다.
    print("모델 학습이 완료되었습니다.")

if __name__ == "__main__":
    main()
