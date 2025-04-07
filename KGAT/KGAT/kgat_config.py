import torch

# 하이퍼파라미터 설정
NUM_NEIGHBORS = 5         # 각 아이템에 연결할 이웃(속성) 수
EMBEDDING_DIM = 64        # 임베딩 차원
DROPOUT = 0.5             # 드롭아웃 비율
NUM_USERS = 466           # 전체 고유 사용자 수 (train, val, test에 등장하는 모든 유저)
LEARNING_RATE = 0.001     # 학습률
EPOCHS = 30               # 학습 에폭 수
BATCH_SIZE = 128          # 배치 사이즈
PAD_TOKEN = -1            # 이웃이 부족할 경우 패딩 토큰
POSITIVE_THRESHOLD = 14   # 평점이 이 값 이상이면 긍정 상호작용으로 간주
K = 20                    # 평가 시 상위 K 추천 아이템
N_CANDIDATES = 3          # 각 긍정 샘플 당 부정 후보 수 (hard negative 샘플링용)
MARGIN = 0.3              # margin ranking loss에 사용할 마진 값
LAMBDA_MARGIN = 0.1       # BPR loss와 margin loss의 가중치 조절
WEIGHT_DECAY = 1e-5       # Adam 옵티마이저의 weight decay (L2 정규화)

# 디바이스 설정 (GPU 사용 가능 시 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 파일 경로 설정
WINE_DATA_PATH = "wine_info_processed_quintiles.csv"
TRAIN_DATA_PATH = "train_data.csv"
VAL_DATA_PATH = "val_data.csv"
TEST_DATA_PATH = "test_data.csv"
