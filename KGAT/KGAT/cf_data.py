import pandas as pd
from kgat_config import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH

def load_cf_data(train_path: str = TRAIN_DATA_PATH, 
                 val_path: str = VAL_DATA_PATH, 
                 test_path: str = TEST_DATA_PATH):
    """
    CF 데이터셋(사용자–아이템–평점)을 로드합니다.
    Returns:
      - train: 학습 데이터셋 DataFrame
      - val: 검증 데이터셋 DataFrame
      - test: 테스트 데이터셋 DataFrame
    """
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    
    print("Train data shape:", train.shape)
    print("Validation data shape:", val.shape)
    print("Test data shape:", test.shape)
    
    return train, val, test

if __name__ == "__main__":
    load_cf_data()
