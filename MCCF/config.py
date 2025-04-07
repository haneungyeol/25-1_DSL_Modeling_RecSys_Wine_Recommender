# 환경 설정

!pip install sentence-transformers

!pip install torch torchvision torchaudio torch-geometric pandas scikit-learn numpy

import os
import torch
import random
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# 데이터 로드
train_df = pd.read_csv("/content/drive/MyDrive/25-1 DSL Modeling/modelling data/filtered_train_data.csv")
val_df   = pd.read_csv("/content/drive/MyDrive/25-1 DSL Modeling/modelling data/filtered_val_data.csv")
test_df  = pd.read_csv("/content/drive/MyDrive/25-1 DSL Modeling/modelling data/filtered_test_data.csv")

# 와이드 포맷을 롱 포맷으로 변환
def wide_to_long(df):
    df = df.rename(columns={df.columns[0]: "user"})
    melted = df.melt(id_vars="user", var_name="wine", value_name="rating")
    melted.dropna(subset=["rating"], inplace=True)
    melted.reset_index(drop=True, inplace=True)
    return melted

train_long_df = wide_to_long(train_df)
val_long_df   = wide_to_long(val_df)
test_long_df  = wide_to_long(test_df)

# 사용자 및 와인 ID 인코딩
user_encoder = LabelEncoder()
wine_encoder = LabelEncoder()

all_users = pd.concat([train_long_df["user"], val_long_df["user"], test_long_df["user"]]).unique()
all_wines = pd.concat([train_long_df["wine"], val_long_df["wine"], test_long_df["wine"]]).unique()

user_encoder.fit(all_users)
wine_encoder.fit(all_wines)

for df in [train_long_df, val_long_df, test_long_df]:
    df["user_id"] = user_encoder.transform(df["user"])
    df["wine_id"] = wine_encoder.transform(df["wine"])

# 검증 및 테스트 데이터 분할 (각 사용자별 70%-30%)
def split_data(df):
    train_indices, holdout_indices = [], []
    for user in df["user"].unique():
        user_data = df[df["user"] == user]
        if len(user_data) < 2:
            train_indices.extend(user_data.index)
        else:
            chosen = np.random.choice(user_data.index, size=int(len(user_data) * 0.7), replace=False)
            holdout = list(set(user_data.index) - set(chosen))
            train_indices.extend(chosen)
            holdout_indices.extend(holdout)
    return df.loc[train_indices], df.loc[holdout_indices]

val_train_df, val_test_df = split_data(val_long_df)
test_train_df, test_test_df = split_data(test_long_df)

# MCCF에서 요구하는 리스트 형식으로 변환 (rating * 10 적용)
def df_to_mccf_format(df, log_transform=False, rating_scale_factor=10):
    ratings = df["rating"].astype(float)
    if log_transform:
        ratings = np.log1p(ratings) * rating_scale_factor  # log(rating + 1) * 10
    return list(zip(
        df["user_id"].astype(int),
        df["wine_id"].astype(int),
        ratings
    ))

train_data = df_to_mccf_format(train_long_df)
val_train_data = df_to_mccf_format(val_train_df)
val_test_data = df_to_mccf_format(val_test_df)
test_train_data = df_to_mccf_format(test_train_df)
test_test_data = df_to_mccf_format(test_test_df)

# 사용자 및 아이템 수 계산
num_total_users = max(train_long_df["user_id"].max(), val_long_df["user_id"].max(), test_long_df["user_id"].max()) + 1
num_total_items = max(train_long_df["wine_id"].max(), val_long_df["wine_id"].max(), test_long_df["wine_id"].max()) + 1

# 인접 리스트 생성 (MCCF 방식)
u_adj = {i: [] for i in range(num_total_users)}
i_adj = {i: [] for i in range(num_total_items)}

for user_id, wine_id, rating in train_data:
    u_adj[user_id].append((wine_id, rating))
    i_adj[wine_id].append((user_id, rating))

# 사용자 및 아이템 특성 벡터 초기화
ufeature = np.zeros((num_total_users, num_total_items), dtype=np.float32)
ifeature = np.zeros((num_total_items, num_total_users), dtype=np.float32)

for user_id, items in u_adj.items():
    for wine_id, rating in items:
        ufeature[user_id, wine_id] = rating

for wine_id, users in i_adj.items():
    for user_id, rating in users:
        ifeature[wine_id, user_id] = rating

# PyTorch 임베딩 변환
ufeature_tensor = torch.tensor(ufeature, dtype=torch.float32)
ifeature_tensor = torch.tensor(ifeature, dtype=torch.float32)

u2e = nn.Embedding.from_pretrained(ufeature_tensor, freeze=False)
i2e = nn.Embedding.from_pretrained(ifeature_tensor, freeze=False)

# 데이터 저장 (MCCF 모델용)
output_path = "mccf_data.p"
with open(output_path, "wb") as meta:
    pickle.dump((u2e, i2e, train_data, val_train_data, val_test_data, test_train_data, test_test_data, u_adj, i_adj), meta)

print(f"✅ MCCF 데이터 전처리 완료! 저장 위치: {output_path}")

import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# CSV 파일 로드
df = pd.read_csv("/content/drive/MyDrive/25-1 DSL Modeling/wine item data/Final_Merged_Wine_Data.csv")

# 1. 맛·향 관련: Flavor Group 및 Keywords 처리
flavor_cols = ["Flavor Group 1", "Keywords 1", "Flavor Group 2", "Keywords 2", "Flavor Group 3", "Keywords 3"]
df["flavor_text"] = df[flavor_cols].fillna("").agg(" ".join, axis=1)

# 2. 생산 및 스타일 관련: Grapes, Region, Wine style, Winery, Alcohol content 처리
prod_cols = ["Grapes", "Region", "Wine style", "Winery", "Alcohol content"]
df["prod_text"] = df[prod_cols].fillna("").agg(" ".join, axis=1)

# 3. Food Pairing 처리
df["food_pairing_text"] = df["Food Pairing"].fillna("")

# BERT 모델 로드 (예: all-MiniLM-L6-v2, 임베딩 차원 384)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# BERT 임베딩 계산 함수
def get_bert_embeddings(text_list, target_dim=64):
    # text_list: 리스트 형태의 텍스트 (문장)들
    embeddings = bert_model.encode(text_list, show_progress_bar=True)
    # embeddings: (num_samples, 384)
    # PCA로 target_dim 차원으로 축소
    pca = PCA(n_components=target_dim, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return reduced  # (num_samples, target_dim)

# 각 영역별 BERT 임베딩 계산 (목표: 64차원)
flavor_emb = get_bert_embeddings(df["flavor_text"].tolist(), target_dim=64)
prod_emb = get_bert_embeddings(df["prod_text"].tolist(), target_dim=64)
food_emb = get_bert_embeddings(df["food_pairing_text"].tolist(), target_dim=64)

# 4. 사이드 정보 텐서 구성
# 각 와인에 대해: 토큰 1 = 맛·향 (flavor_emb), 토큰 2 = 생산/스타일 (prod_emb), 토큰 3 = Food Pairing (food_emb)
wine_side_features_np = np.stack([flavor_emb, prod_emb, food_emb], axis=1)  # shape: (num_items, 3, 64)
wine_side_features = torch.tensor(wine_side_features_np, dtype=torch.float)

# MCCF에 맞게 데이터 변환
def df_to_mccf_format(df):
    return list(zip(df["user_id"].astype(int), df["wine_id"].astype(int), df["rating"].astype(float)))

train_data = df_to_mccf_format(train_long_df)
val_train_data = df_to_mccf_format(val_train_df)
val_test_data = df_to_mccf_format(val_test_df)
test_train_data = df_to_mccf_format(test_train_df)
test_test_data = df_to_mccf_format(test_test_df)

num_total_users = train_long_df["user_id"].nunique()
num_total_items = train_long_df["wine_id"].nunique()

# ✅ (num_items, 3, 64) → (num_items, 192)
wine_side_features_reshaped = wine_side_features.view(wine_side_features.shape[0], -1)
#print("✅ Reshaped wine_side_features:", wine_side_features_reshaped.shape)

# PyTorch Embedding 업데이트 (아이템 정보 반영)
i2e = nn.Embedding.from_pretrained(wine_side_features_reshaped, freeze=False)

# 데이터 저장 (MCCF 모델용)
output_path = "/content/drive/MyDrive/25-1 DSL Modeling/mccf_data.p"
with open(output_path, "wb") as meta:
    pickle.dump((i2e, train_data, val_train_data, val_test_data, test_train_data, test_test_data), meta)

print(f"✅ MCCF 데이터 전처리 완료! 저장 위치: {output_path}")

print("wine_side_features shape:", wine_side_features.shape)

from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F

class MCCF(nn.Module):
    def __init__(self, user_embedding, item_embedding, embed_dim, wine_side_info=None, N=30000, dropout_rate=0.5, beta_ema=0.999):
        super(MCCF, self).__init__()

        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.embed_dim = embed_dim
        self.N = N
        self.dropout_rate = dropout_rate
        self.beta_ema = beta_ema
        self.criterion = nn.MSELoss()

        # ✅ 와인 부가 정보 추가 (리스트일 경우 변환)
        if wine_side_info is not None:
            if isinstance(wine_side_info, list):
                wine_side_info = torch.tensor(wine_side_info, dtype=torch.float32)
            self.wine_side_info = wine_side_info
            wine_feature_dim = self.wine_side_info.shape[1]
        else:
            self.wine_side_info = None
            wine_feature_dim = 0

        print(f"✅ 와인 부가 정보 사용 여부: {'Yes' if wine_feature_dim > 0 else 'No'} | 차원: {wine_feature_dim}")
        #print(f"✅ 아이템 임베딩 차원: {self.item_embedding.embedding_dim}")

        # ✅ 임베딩 차원(469) + wine 부가 정보 차원(3) = 총 입력 차원
        item_input_dim = self.item_embedding.embedding_dim + wine_feature_dim
        self.item_layer1 = nn.Linear(item_input_dim, self.embed_dim)
        self.item_layer2 = nn.Linear(self.embed_dim, self.embed_dim)

        # 사용자-아이템 상호작용 학습 MLP
        self.interaction_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.interaction_layer2 = nn.Linear(self.embed_dim, 1)



        #print("🔎 item_embedding.embedding_dim:", self.item_embedding.embedding_dim)


    def forward(self, user_ids, item_ids, wine_features=None):
        #print("▶️ forward() 호출됨")  # ✅ 제일 앞에 위치

        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)

        if self.wine_side_info is not None and wine_features is None:
            wine_features = self.wine_side_info[item_ids]

        if wine_features is not None:
            #print(f"[Debug] wine_features shape: {wine_features.shape}")
            item_embedded = torch.cat((item_embedded, wine_features), dim=1)

        #print(f"[Debug] item_embedded shape: {item_embedded.shape}")

        item_hidden = F.relu(self.item_layer1(item_embedded))
        item_hidden = self.item_layer2(item_hidden)

        interaction_input = torch.cat((user_embedded, item_hidden), dim=1)
        #print(f"🧩 interaction_input shape: {interaction_input.shape}")
        scores = self.interaction_layer2(F.relu(self.interaction_layer1(interaction_input)))


        #print(f"📦 item_embedded shape before concat: {item_embedded.shape}")


        return scores.squeeze()  # ✅ return은 마지막에


    def compute_loss(self, user_ids, item_ids, ratings, wine_features=None):
        """
        모델 손실 함수 계산

        Args:
        - user_ids (torch.Tensor): 사용자 ID 텐서
        - item_ids (torch.Tensor): 아이템(와인) ID 텐서
        - ratings (torch.Tensor): 실제 평점 텐서
        - wine_features (torch.Tensor, optional): 아이템의 부가적 와인 정보

        Returns:
        - loss (torch.Tensor): MSE 손실 값
        """
        predicted_scores = self.forward(user_ids, item_ids, wine_features)
        loss = self.criterion(predicted_scores, ratings)
        return loss

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def test(model, test_loader, device):
    """
    MCCF 모델 테스트 함수 (RMSE, MAE 평가)

    Args:
    - model (MCCF): 학습된 MCCF 모델
    - test_loader (DataLoader): 테스트 데이터 배치 로더
    - device (torch.device): 학습 장치 (GPU/CPU)

    Returns:
    - rmse (float): Root Mean Squared Error (RMSE)
    - mae (float): Mean Absolute Error (MAE)
    """
    model.eval()
    pred, ground_truth = [], []

    with torch.no_grad():  # ✅ No gradient computation for inference
        for test_u, test_i, test_ratings, test_wine_features in test_loader:
            # ✅ 데이터 GPU/CPU로 이동
            test_u = test_u.to(device)
            test_i = test_i.to(device)
            test_ratings = test_ratings.to(device)
            test_wine_features = test_wine_features.to(device) if test_wine_features is not None else None

            # ✅ 예측 수행
            scores = model(test_u, test_i)

            # ✅ NumPy 변환 후 리스트에 저장
            pred.append(scores.cpu().numpy())
            ground_truth.append(test_ratings.cpu().numpy())

    # ✅ 리스트를 NumPy 배열로 변환
    pred = np.concatenate(pred)
    ground_truth = np.concatenate(ground_truth)

    # ✅ RMSE & MAE 계산
    rmse = mean_squared_error(ground_truth, pred)  # ✅ squared=False 옵션 사용
    mae = mean_absolute_error(ground_truth, pred)

    print(f"✅ Test 결과 - RMSE: {rmse:.5f}, MAE: {mae:.5f}")
    return rmse, mae

import numpy as np
import torch

def recall_at_k(y_true, y_pred, k=20):
    y_pred = y_pred[:k]
    num_relevant = len(set(y_true) & set(y_pred))
    return num_relevant / len(y_true) if len(y_true) > 0 else 0

def precision_at_k(y_true, y_pred, k=20):
    y_pred = y_pred[:k]
    num_relevant = len(set(y_true) & set(y_pred))
    return num_relevant / k

def ndcg_at_k(y_true, y_pred, k=20):
    y_pred = y_pred[:k]
    dcg = 0.0
    for idx, item in enumerate(y_pred):
        if item in y_true:
            dcg += 1 / np.log2(idx + 2)  # log2(rank + 1)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate_mccf(model, test_data, k=20):
    recall_list, ndcg_list, precision_list = [], [], []

    users = list(set([u for u, _, _ in test_data]))

    for user in users:
        user_actual_items = set([i for u, i, r in test_data if u == user and r > 0])

        with torch.no_grad():
            item_scores = {i: model.forward(torch.tensor([user]), torch.tensor([i])).item()
                           for _, i, _ in test_data}

        sorted_items = sorted(item_scores, key=item_scores.get, reverse=True)[:k]

        recall_list.append(recall_at_k(user_actual_items, sorted_items, k))
        precision_list.append(precision_at_k(user_actual_items, sorted_items, k))
        ndcg_list.append(ndcg_at_k(user_actual_items, sorted_items, k))

    recall_avg = np.mean(recall_list)
    precision_avg = np.mean(precision_list)
    ndcg_avg = np.mean(ndcg_list)

    print(f"\U0001F4CA Evaluation Results - Recall@{k}: {recall_avg:.5f}, Precision@{k}: {precision_avg:.5f}, NDCG@{k}: {ndcg_avg:.5f}")
    return recall_avg, precision_avg, ndcg_avg

import argparse
from torch.utils.data import TensorDataset

def main():
    import argparse
    import pickle
    import torch
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    parser = argparse.ArgumentParser(description="MCCF")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--droprate', type=float, default=0.3)

    args, unknown = parser.parse_known_args()
    print('-------------------- Hyperparams --------------------')
    print(f"Learning rate: {args.lr}, Embedding Dimension: {args.embed_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Load data
    dataset_path = '/content/mccf_data.p'
    with open(dataset_path, "rb") as f:
        loaded_data = pickle.load(f)

    if len(loaded_data) < 5:
        raise ValueError("❌ 데이터셋 형식이 올바르지 않습니다. 최소 5개의 요소가 필요합니다.")

    user_embedding, item_embedding, train_data, test_data, wine_side_info = loaded_data[:5]
    print(f"✅ Train 데이터 크기: {len(train_data)}, Test 데이터 크기: {len(test_data)}")

    # ✅ user embedding 차원 줄이기 (64차원만 사용)
    user_embedding_matrix = user_embedding.weight.data[:, :args.embed_dim]
    user_embedding = nn.Embedding.from_pretrained(user_embedding_matrix.clone(), freeze=False)

    item_embedding_matrix = item_embedding.weight.data[:, :args.embed_dim]
    item_embedding = nn.Embedding.from_pretrained(item_embedding_matrix.clone(), freeze=False)

    print("✅ user_embedding type:", type(user_embedding))

    if isinstance(wine_side_info, torch.Tensor) and wine_side_info.dim() == 3:
        wine_side_info = wine_side_info.view(wine_side_info.shape[0], -1)

    wine_side_info = torch.FloatTensor(wine_side_info)
    num_items = wine_side_info.shape[0]

    # ✅ 전처리: 유효한 item 인덱스만 유지
    train_data = [x for x in train_data if x[1] < num_items]
    test_data = [x for x in test_data if x[1] < num_items]

    # 🔁 다시 인덱스 추출
    item_indices_train = torch.LongTensor([x[1] for x in train_data])
    item_indices_test = torch.LongTensor([x[1] for x in test_data])

    trainset = TensorDataset(
        torch.LongTensor([x[0] for x in train_data]),
        torch.LongTensor([x[1] for x in train_data]),
        torch.FloatTensor([x[2] for x in train_data]),
        wine_side_info.index_select(0, item_indices_train)
    )

    testset = TensorDataset(
        torch.LongTensor([x[0] for x in test_data]),
        torch.LongTensor([x[1] for x in test_data]),
        torch.FloatTensor([x[2] for x in test_data]),
        wine_side_info.index_select(0, item_indices_test)
    )

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    # ✅ 모델 초기화
    model = MCCF(
        user_embedding,
        item_embedding,
        args.embed_dim,
        wine_side_info=wine_side_info,
        dropout_rate=args.droprate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_rmse, best_mae = np.inf, np.inf
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, device)
        rmse, mae = test(model, test_loader, device)

        print(f"<Test> RMSE: {rmse:.5f}, MAE: {mae:.5f}")

        if endure_count > 30:
            break

    # ✅ Top-K 평가
    print("🔍 Running Top-K Evaluation...")
    recall, precision, ndcg = evaluate_mccf(model, test_data, k=20)

    print(f"Best RMSE/MAE: {best_rmse:.5f} / {best_mae:.5f}")
    print(f"Top-K Metrics — Recall@20: {recall:.5f}, Precision@20: {precision:.5f}, NDCG@20: {ndcg:.5f}")

