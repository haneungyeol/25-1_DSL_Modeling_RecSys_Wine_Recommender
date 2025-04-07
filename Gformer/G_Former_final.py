#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install torch torchvision torchaudio torch-geometric pandas scikit-learn numpy')


# In[5]:


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

print("wine_side_features shape:", wine_side_features.shape)


# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, TransformerConv
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

#########################################
# 1. 데이터 로드 및 전처리
#########################################
train_df = pd.read_csv("filtered_train_data.csv")
val_df   = pd.read_csv("filtered_val_data.csv")
test_df  = pd.read_csv("filtered_test_data.csv")


def wide_to_long(df):
    df = df.rename(columns={df.columns[0]: "user"})
    melted = df.melt(id_vars="user", var_name="wine", value_name="rating")
    melted.dropna(subset=["rating"], inplace=True)
    melted.reset_index(drop=True, inplace=True)
    return melted

train_long_df = wide_to_long(train_df)
val_long_df   = wide_to_long(val_df)
test_long_df  = wide_to_long(test_df)

user_encoder = LabelEncoder()
wine_encoder = LabelEncoder()

all_users = pd.concat([train_long_df["user"], val_long_df["user"], test_long_df["user"]]).unique()
all_wines = pd.concat([train_long_df["wine"], val_long_df["wine"], test_long_df["wine"]]).unique()
user_encoder.fit(all_users)
wine_encoder.fit(all_wines)

for df in [train_long_df, val_long_df, test_long_df]:
    df["user_id"] = user_encoder.transform(df["user"])
    df["wine_id"] = wine_encoder.transform(df["wine"])

#########################################
# 2. 검증 및 테스트 데이터 분할 (각 사용자별 70%-30%)
#########################################
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

num_total_users = max(train_long_df["user_id"].max(), val_long_df["user_id"].max(), test_long_df["user_id"].max()) + 1
num_total_items = max(train_long_df["wine_id"].max(), val_long_df["wine_id"].max(), test_long_df["wine_id"].max()) + 1

#########################################
# 3. Edge Augmentation 관련 함수 (Dropout + Prediction)
#########################################
def edge_dropout(edge_index, dropout_rate=0.2):
    num_edges = edge_index.shape[1]
    keep_mask = torch.rand(num_edges) > dropout_rate
    return edge_index[:, keep_mask]

class EdgePredictor(nn.Module):
    def __init__(self, feature_dim):
        super(EdgePredictor, self).__init__()
        # feature_dim * 2 입력 (사용자와 와인 임베딩 concat)
        self.fc = nn.Linear(feature_dim * 2, 1)

    def forward(self, user_emb, wine_emb):
        interaction = torch.cat([user_emb, wine_emb], dim=1)
        score = torch.sigmoid(self.fc(interaction))
        return score

#########################################
# 4. Dataset 정의
#########################################
class WineDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.wines = torch.tensor(df["wine_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.wines[idx], self.ratings[idx]

#########################################
# 5. Attention 기반 융합 모듈: WineSideFusion
#########################################
class WineSideFusion(nn.Module):
    def __init__(self, base_dim, side_dim, fusion_dim, num_side_tokens):
        """
        base_dim: 투영된 와인 임베딩 차원 (예: fusion_dim, 256)
        side_dim: 사이드 정보의 각 토큰 차원 (예: 64)
        fusion_dim: 융합 후 출력 차원 (보통 base_dim와 동일)
        num_side_tokens: 사이드 정보 시퀀스 길이 (예: 4)
        """
        super(WineSideFusion, self).__init__()
        self.side_proj = nn.Linear(side_dim, base_dim)
        self.attn = nn.MultiheadAttention(embed_dim=base_dim, num_heads=1, batch_first=True)
        self.fc = nn.Linear(base_dim * 2, fusion_dim)

    def forward(self, base, side_seq):
        # side_seq: [batch, num_side_tokens, side_dim]
        side_proj = self.side_proj(side_seq)  # [batch, num_side_tokens, base_dim]
        query = base.unsqueeze(1)              # [batch, 1, base_dim]
        attn_output, _ = self.attn(query, side_proj, side_proj)
        attn_output = attn_output.squeeze(1)   # [batch, base_dim]
        fused = torch.cat([base, attn_output], dim=1)
        fused = self.fc(fused)
        return fused  # [batch, fusion_dim]

#########################################
# 6. (선택적) 사이드 정보 인코더: WineFeatureEncoder
#########################################
# 예시로 MLP 기반 인코더 (실제로는 BERT 등 전처리된 결과 사용 가능)
class WineFeatureEncoder(nn.Module):
    def __init__(self, numeric_dim, flavor_vocab_size, flavor_embed_dim, cat_vocab_sizes, cat_embed_dim, food_pairing_dim, output_dim, num_side_tokens):
        super(WineFeatureEncoder, self).__init__()
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, output_dim // 4),
            nn.ReLU()
        )
        self.flavor_encoder = nn.Sequential(
            nn.Linear(flavor_vocab_size, output_dim // 4),
            nn.ReLU()
        )
        self.grapes_emb = nn.Embedding(cat_vocab_sizes[0], cat_embed_dim)
        self.region_emb = nn.Embedding(cat_vocab_sizes[1], cat_embed_dim)
        self.wine_style_emb = nn.Embedding(cat_vocab_sizes[2], cat_embed_dim)
        self.winery_emb = nn.Embedding(cat_vocab_sizes[3], cat_embed_dim)
        self.cat_linear = nn.Sequential(
            nn.Linear(cat_embed_dim * 4, output_dim // 4),
            nn.ReLU()
        )
        self.alcohol_linear = nn.Sequential(
            nn.Linear(1, output_dim // 8),
            nn.ReLU()
        )
        self.food_pairing_linear = nn.Sequential(
            nn.Linear(food_pairing_dim, output_dim // 8),
            nn.ReLU()
        )
        total_dim = output_dim // 4 + output_dim // 4 + output_dim // 4 + output_dim // 8 + output_dim // 8
        self.fc = nn.Linear(total_dim, output_dim)
        self.num_side_tokens = num_side_tokens

    def forward(self, numeric_feats, flavor_multihot, grapes_idx, region_idx, wine_style_idx, winery_idx, alcohol, food_pairing):
        num_out = self.numeric_encoder(numeric_feats)
        flavor_out = self.flavor_encoder(flavor_multihot)
        grapes_emb = self.grapes_emb(grapes_idx)
        region_emb = self.region_emb(region_idx)
        wine_style_emb = self.wine_style_emb(wine_style_idx)
        winery_emb = self.winery_emb(winery_idx)
        cat_concat = torch.cat([grapes_emb, region_emb, wine_style_emb, winery_emb], dim=1)
        cat_out = self.cat_linear(cat_concat)
        alcohol_out = self.alcohol_linear(alcohol)
        food_pairing_out = self.food_pairing_linear(food_pairing)
        combined = torch.cat([num_out, flavor_out, cat_out, alcohol_out, food_pairing_out], dim=1)
        side_embedding = self.fc(combined)
        token_dim = side_embedding.size(1) // self.num_side_tokens
        side_seq = side_embedding.view(side_embedding.size(0), self.num_side_tokens, token_dim)
        return side_seq

#########################################
# 7. G-Former 모델에 사이드 정보 통합: GFormerWithSide
#########################################
class GFormerWithSide(nn.Module):
    def __init__(self, num_users, num_items, wine_side_features, side_fusion_module,
                 embedding_dim=64, num_heads=4, mask_ratio=0.2, dropout_rate=0.2):
        """
        wine_side_features: [num_items, num_side_tokens, side_token_dim] - 미리 전처리된 사이드 정보
        side_fusion_module: WineSideFusion 모듈
        """
        super(GFormerWithSide, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.wine_side_features = wine_side_features  # 미리 전처리된 사이드 정보
        self.side_fusion = side_fusion_module

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.wine_embedding = nn.Embedding(num_items, embedding_dim)
        # 와인 임베딩을 fusion 차원으로 투영 (예: 64 -> 256)
        self.fusion_dim = embedding_dim * num_heads  # 예: 64*4 = 256
        self.wine_proj = nn.Linear(embedding_dim, self.fusion_dim)
        # 사용자 임베딩도 같은 fusion_dim으로 투영
        self.user_proj = nn.Linear(embedding_dim, self.fusion_dim)

        # 변경: GCNConv와 TransformerConv의 입력/출력 차원을 fusion_dim으로 설정
        self.gcn = GCNConv(self.fusion_dim, self.fusion_dim)
        self.transformer = TransformerConv(self.fusion_dim, self.fusion_dim, heads=num_heads, concat=False)
        self.fc = nn.Linear(self.fusion_dim * 2, 1)

        self.mask_ratio = mask_ratio
        self.dropout_rate = dropout_rate
        self.node_autoencoder = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.edge_predictor = EdgePredictor(self.fusion_dim)

    def forward(self, user, wine, edge_index, mask_autoencoding=True):
        edge_index = edge_dropout(edge_index, dropout_rate=self.dropout_rate)

        # 사용자 임베딩 투영
        user_all = self.user_embedding.weight          # [num_users, 64]
        user_all_proj = self.user_proj(user_all)         # [num_users, fusion_dim]

        # 와인 임베딩 투영 및 사이드 정보 융합
        wine_all = self.wine_embedding.weight            # [num_items, 64]
        wine_all_proj = self.wine_proj(wine_all)           # [num_items, fusion_dim]
        fused_wine = self.side_fusion(wine_all_proj, self.wine_side_features)  # [num_items, fusion_dim]

        # 전체 노드 임베딩: 사용자와 융합된 와인 임베딩 concat
        node_features = torch.cat([user_all_proj, fused_wine], dim=0)  # [num_users+num_items, fusion_dim]

        node_features = self.gcn(node_features, edge_index)
        node_features = self.transformer(node_features, edge_index)

        recon_loss = 0
        if mask_autoencoding:
            num_nodes = node_features.size(0)
            mask = torch.rand(num_nodes, device=node_features.device) < self.mask_ratio
            target_features = node_features[mask]
            masked_features = node_features.clone()
            masked_features[mask] = 0
            recon_features = self.node_autoencoder(masked_features[mask])
            recon_loss = F.mse_loss(recon_features, target_features)

        predicted_edge_scores = self.edge_predictor(node_features[user],
                                                     node_features[wine + self.user_embedding.weight.size(0)])
        edge_pred_loss = F.mse_loss(predicted_edge_scores, torch.ones_like(predicted_edge_scores))

        user_out = node_features[user]
        wine_out = node_features[wine + self.user_embedding.weight.size(0)]
        interaction = torch.cat([user_out, wine_out], dim=1)
        score = self.fc(interaction).squeeze()
        score = torch.sigmoid(score) * 4 + 1

        return score, recon_loss, edge_pred_loss

#########################################
# 8. 최종 학습, 검증 및 테스트 파이프라인
#########################################
# 8-1. 학습 데이터: 전체 edge_index 구성 (Edge Augmentation 적용)
train_edge_index = torch.tensor(
    np.array([train_long_df["user_id"].to_numpy(), train_long_df["wine_id"].to_numpy()]),
    dtype=torch.long
)

train_dataset = WineDataset(train_long_df)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# 미리 전처리된 와인 사이드 정보: (num_items, num_side_tokens, side_token_dim)
# 여기서는 예시로 임의의 텐서를 생성 (예: num_total_items x 4 x 64)
num_side_tokens = 4
side_token_dim = 64
wine_side_features = torch.randn(num_total_items, num_side_tokens, side_token_dim)

# Attention 기반 융합 모듈 생성: fusion_dim = embedding_dim * num_heads = 64*4 = 256
fusion_dim = 256
side_fusion_module = WineSideFusion(base_dim=fusion_dim, side_dim=side_token_dim,
                                      fusion_dim=fusion_dim, num_side_tokens=num_side_tokens)

# G-Former 모델 (사이드 정보 포함) 생성
model = GFormerWithSide(num_total_users, num_total_items, wine_side_features, side_fusion_module,
                        embedding_dim=64, num_heads=4, mask_ratio=0.2, dropout_rate=0.2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
recon_weight = 0.5      # Reconstruction loss 가중치
edge_pred_weight = 0.5  # Edge Prediction loss 가중치

num_train_epochs = 10
print("===== TRAINING with Edge Augmentation + Side Information =====")
for epoch in range(num_train_epochs):
    model.train()
    total_loss = 0
    for user_batch, wine_batch, rating_batch in train_loader:
        optimizer.zero_grad()
        score, recon_loss, edge_pred_loss = model(user_batch, wine_batch, train_edge_index, mask_autoencoding=True)
        sup_loss = criterion(score, rating_batch)
        loss = sup_loss + recon_weight * recon_loss + edge_pred_weight * edge_pred_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training Epoch {epoch+1}/{num_train_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# 8-2. 검증 데이터 Fine-tuning (70% 데이터)
val_ft_edge_index = torch.tensor(
    np.array([val_train_df["user_id"].to_numpy(), val_train_df["wine_id"].to_numpy()]),
    dtype=torch.long
)
val_train_dataset = WineDataset(val_train_df)
val_train_loader = DataLoader(val_train_dataset, batch_size=1024, shuffle=True)

num_ft_epochs = 3
print("\n===== VALIDATION FINE-TUNING =====")
for epoch in range(num_ft_epochs):
    model.train()
    total_loss = 0
    for user_batch, wine_batch, rating_batch in val_train_loader:
        optimizer.zero_grad()
        score, recon_loss, edge_pred_loss = model(user_batch, wine_batch, val_ft_edge_index, mask_autoencoding=True)
        sup_loss = criterion(score, rating_batch)
        loss = sup_loss + recon_weight * recon_loss + edge_pred_weight * edge_pred_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Validation Fine-tuning Epoch {epoch+1}/{num_ft_epochs}, Loss: {total_loss / len(val_train_loader):.4f}")

# 8-3. 검증 Hold-out 데이터 평가 (30% 데이터)
print("\n===== VALIDATION EVALUATION =====")
with torch.no_grad():
    val_user_tensor = torch.tensor(val_test_df["user_id"].values, dtype=torch.long)
    val_wine_tensor = torch.tensor(val_test_df["wine_id"].values, dtype=torch.long)
    val_eval_edge_index = torch.tensor(
        np.array([val_test_df["user_id"].to_numpy(), val_test_df["wine_id"].to_numpy()]),
        dtype=torch.long
    )
    score, _, _ = model(val_user_tensor, val_wine_tensor, val_eval_edge_index, mask_autoencoding=False)
    actual_val_ratings = torch.tensor(val_test_df["rating"].values, dtype=torch.float)
    mse_val = mean_squared_error(actual_val_ratings.numpy(), score.numpy())
    rmse_val = np.sqrt(mse_val)
    print(f"Validation After Fine-tuning - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}")

# 8-4. 테스트 데이터 평가 (Edge Augmentation + Side 정보 적용된 모델 검증)
print("\n===== TEST EVALUATION =====")
with torch.no_grad():
    test_user_tensor = torch.tensor(test_test_df["user_id"].values, dtype=torch.long)
    test_wine_tensor = torch.tensor(test_test_df["wine_id"].values, dtype=torch.long)
    test_eval_edge_index = torch.tensor(
        np.array([test_test_df["user_id"].to_numpy(), test_test_df["wine_id"].to_numpy()]),
        dtype=torch.long
    )
    score, _, _ = model(test_user_tensor, test_wine_tensor, test_eval_edge_index, mask_autoencoding=False)
    actual_test_ratings = torch.tensor(test_test_df["rating"].values, dtype=torch.float)
    mse_test = mean_squared_error(actual_test_ratings.numpy(), score.numpy())
    rmse_test = np.sqrt(mse_test)
    print(f"Test Evaluation - MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}")


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, TransformerConv
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

#########################################
# NDCG 계산 함수 정의
#########################################
def ndcg_at_k(r, k):
    """
    r: relevance scores (정렬된 실제 평점 배열)
    k: top-k까지의 NDCG 계산
    """
    r = np.asfarray(r)[:k]
    if r.size == 0:
        return 0.
    dcg = np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
    ideal_r = np.sort(r)[::-1]
    idcg = np.sum((2**ideal_r - 1) / np.log2(np.arange(2, ideal_r.size + 2)))
    return dcg / idcg if idcg > 0 else 0.

def compute_ndcg(user_ids, predictions, actuals, k=10):
    """
    user_ids: 예측마다 해당하는 사용자 id (numpy array)
    predictions: 예측 평점 (numpy array)
    actuals: 실제 평점 (numpy array)
    k: NDCG@k 계산
    """
    # 사용자별로 인덱스 그룹화
    user_to_indices = {}
    for i, u in enumerate(user_ids):
        user_to_indices.setdefault(u, []).append(i)
    ndcg_scores = []
    for u, indices in user_to_indices.items():
        preds = predictions[indices]
        acts = actuals[indices]
        sorted_idx = np.argsort(-preds)
        sorted_acts = acts[sorted_idx]
        ndcg_scores.append(ndcg_at_k(sorted_acts, k))
    return np.mean(ndcg_scores)


#########################################
# 1. 데이터 로드 및 전처리
#########################################
train_df = pd.read_csv("/content/drive/MyDrive/25-1 DSL Modeling/modelling data/filtered_train_data.csv")
val_df   = pd.read_csv("/content/drive/MyDrive/25-1 DSL Modeling/modelling data/filtered_val_data.csv")
test_df  = pd.read_csv("/content/drive/MyDrive/25-1 DSL Modeling/modelling data/filtered_test_data.csv")

def wide_to_long(df):
    df = df.rename(columns={df.columns[0]: "user"})
    melted = df.melt(id_vars="user", var_name="wine", value_name="rating")
    melted.dropna(subset=["rating"], inplace=True)
    melted.reset_index(drop=True, inplace=True)
    return melted

train_long_df = wide_to_long(train_df)
val_long_df   = wide_to_long(val_df)
test_long_df  = wide_to_long(test_df)

# 로그 변환 및 10배 스케일링 추가 (예: 원래 평점이 1~5라고 가정)
for df in [train_long_df, val_long_df, test_long_df]:
    df["rating"] = np.log(df["rating"]) * 10

user_encoder = LabelEncoder()
wine_encoder = LabelEncoder()

all_users = pd.concat([train_long_df["user"], val_long_df["user"], test_long_df["user"]]).unique()
all_wines = pd.concat([train_long_df["wine"], val_long_df["wine"], test_long_df["wine"]]).unique()
user_encoder.fit(all_users)
wine_encoder.fit(all_wines)

for df in [train_long_df, val_long_df, test_long_df]:
    df["user_id"] = user_encoder.transform(df["user"])
    df["wine_id"] = wine_encoder.transform(df["wine"])

#########################################
# 2. 검증 및 테스트 데이터 분할 (각 사용자별 70%-30%)
#########################################
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

num_total_users = max(train_long_df["user_id"].max(), val_long_df["user_id"].max(), test_long_df["user_id"].max()) + 1
num_total_items = max(train_long_df["wine_id"].max(), val_long_df["wine_id"].max(), test_long_df["wine_id"].max()) + 1

#########################################
# 3. Edge Augmentation 관련 함수 (Dropout + Prediction)
#########################################
def edge_dropout(edge_index, dropout_rate=0.2):
    num_edges = edge_index.shape[1]
    keep_mask = torch.rand(num_edges) > dropout_rate
    return edge_index[:, keep_mask]

class EdgePredictor(nn.Module):
    def __init__(self, feature_dim):
        super(EdgePredictor, self).__init__()
        # 입력: feature_dim * 2 (사용자와 와인 임베딩 concat)
        self.fc = nn.Linear(feature_dim * 2, 1)

    def forward(self, user_emb, wine_emb):
        interaction = torch.cat([user_emb, wine_emb], dim=1)
        score = torch.sigmoid(self.fc(interaction))
        return score

#########################################
# 4. Dataset 정의
#########################################
class WineDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.wines = torch.tensor(df["wine_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.wines[idx], self.ratings[idx]

#########################################
# 5. Attention 기반 융합 모듈: WineSideFusion
#########################################
class WineSideFusion(nn.Module):
    def __init__(self, base_dim, side_dim, fusion_dim, num_side_tokens):
        """
        base_dim: 투영된 와인 임베딩 차원 (예: fusion_dim, 256)
        side_dim: 사이드 정보의 각 토큰 차원 (예: 64)
        fusion_dim: 융합 후 출력 차원 (보통 base_dim와 동일)
        num_side_tokens: 사이드 정보 시퀀스 길이 (예: 4)
        """
        super(WineSideFusion, self).__init__()
        self.side_proj = nn.Linear(side_dim, base_dim)
        self.attn = nn.MultiheadAttention(embed_dim=base_dim, num_heads=1, batch_first=True)
        self.fc = nn.Linear(base_dim * 2, fusion_dim)

    def forward(self, base, side_seq):
        # side_seq: [batch, num_side_tokens, side_dim]
        side_proj = self.side_proj(side_seq)  # [batch, num_side_tokens, base_dim]
        query = base.unsqueeze(1)              # [batch, 1, base_dim]
        attn_output, _ = self.attn(query, side_proj, side_proj)
        attn_output = attn_output.squeeze(1)   # [batch, base_dim]
        fused = torch.cat([base, attn_output], dim=1)
        fused = self.fc(fused)
        return fused  # [batch, fusion_dim]

#########################################
# 6. (선택적) 사이드 정보 인코더: WineFeatureEncoder
#########################################
# 여기서는 예시로 MLP 기반 인코더를 구성합니다.
class WineFeatureEncoder(nn.Module):
    def __init__(self, numeric_dim, flavor_vocab_size, flavor_embed_dim, cat_vocab_sizes, cat_embed_dim, food_pairing_dim, output_dim, num_side_tokens):
        super(WineFeatureEncoder, self).__init__()
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, output_dim // 4),
            nn.ReLU()
        )
        self.flavor_encoder = nn.Sequential(
            nn.Linear(flavor_vocab_size, output_dim // 4),
            nn.ReLU()
        )
        self.grapes_emb = nn.Embedding(cat_vocab_sizes[0], cat_embed_dim)
        self.region_emb = nn.Embedding(cat_vocab_sizes[1], cat_embed_dim)
        self.wine_style_emb = nn.Embedding(cat_vocab_sizes[2], cat_embed_dim)
        self.winery_emb = nn.Embedding(cat_vocab_sizes[3], cat_embed_dim)
        self.cat_linear = nn.Sequential(
            nn.Linear(cat_embed_dim * 4, output_dim // 4),
            nn.ReLU()
        )
        self.alcohol_linear = nn.Sequential(
            nn.Linear(1, output_dim // 8),
            nn.ReLU()
        )
        self.food_pairing_linear = nn.Sequential(
            nn.Linear(food_pairing_dim, output_dim // 8),
            nn.ReLU()
        )
        total_dim = output_dim // 4 + output_dim // 4 + output_dim // 4 + output_dim // 8 + output_dim // 8
        self.fc = nn.Linear(total_dim, output_dim)
        self.num_side_tokens = num_side_tokens

    def forward(self, numeric_feats, flavor_multihot, grapes_idx, region_idx, wine_style_idx, winery_idx, alcohol, food_pairing):
        num_out = self.numeric_encoder(numeric_feats)
        flavor_out = self.flavor_encoder(flavor_multihot)
        grapes_emb = self.grapes_emb(grapes_idx)
        region_emb = self.region_emb(region_idx)
        wine_style_emb = self.wine_style_emb(wine_style_idx)
        winery_emb = self.winery_emb(winery_idx)
        cat_concat = torch.cat([grapes_emb, region_emb, wine_style_emb, winery_emb], dim=1)
        cat_out = self.cat_linear(cat_concat)
        alcohol_out = self.alcohol_linear(alcohol)
        food_pairing_out = self.food_pairing_linear(food_pairing)
        combined = torch.cat([num_out, flavor_out, cat_out, alcohol_out, food_pairing_out], dim=1)
        side_embedding = self.fc(combined)
        token_dim = side_embedding.size(1) // self.num_side_tokens
        side_seq = side_embedding.view(side_embedding.size(0), self.num_side_tokens, token_dim)
        return side_seq

#########################################
# 7. G-Former 모델에 사이드 정보 통합: GFormerWithSide
#########################################
class GFormerWithSide(nn.Module):
    def __init__(self, num_users, num_items, wine_side_features, side_fusion_module,
                 embedding_dim=64, num_heads=4, mask_ratio=0.2, dropout_rate=0.2):
        """
        wine_side_features: [num_items, num_side_tokens, side_token_dim] - 미리 전처리된 사이드 정보
        side_fusion_module: WineSideFusion 모듈
        """
        super(GFormerWithSide, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.wine_side_features = wine_side_features  # 미리 전처리된 사이드 정보
        self.side_fusion = side_fusion_module

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.wine_embedding = nn.Embedding(num_items, embedding_dim)
        # 와인 임베딩을 fusion 차원으로 투영 (예: 64 -> 256)
        self.fusion_dim = embedding_dim * num_heads  # 예: 64*4 = 256
        self.wine_proj = nn.Linear(embedding_dim, self.fusion_dim)
        # 사용자 임베딩도 같은 fusion_dim으로 투영
        self.user_proj = nn.Linear(embedding_dim, self.fusion_dim)

        # GCNConv와 TransformerConv의 입력/출력 차원을 fusion_dim으로 설정
        self.gcn = GCNConv(self.fusion_dim, self.fusion_dim)
        self.transformer = TransformerConv(self.fusion_dim, self.fusion_dim, heads=num_heads, concat=False)
        self.fc = nn.Linear(self.fusion_dim * 2, 1)

        self.mask_ratio = mask_ratio
        self.dropout_rate = dropout_rate
        self.node_autoencoder = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.edge_predictor = EdgePredictor(self.fusion_dim)

    def forward(self, user, wine, edge_index, mask_autoencoding=True):
        edge_index = edge_dropout(edge_index, dropout_rate=self.dropout_rate)

        # 사용자 임베딩 투영
        user_all = self.user_embedding.weight          # [num_users, 64]
        user_all_proj = self.user_proj(user_all)         # [num_users, fusion_dim]

        # 와인 임베딩 투영 및 사이드 정보 융합
        wine_all = self.wine_embedding.weight            # [num_items, 64]
        wine_all_proj = self.wine_proj(wine_all)           # [num_items, fusion_dim]
        fused_wine = self.side_fusion(wine_all_proj, self.wine_side_features)  # [num_items, fusion_dim]

        # 전체 노드 임베딩: 사용자와 융합된 와인 임베딩 concat
        node_features = torch.cat([user_all_proj, fused_wine], dim=0)  # [num_users+num_items, fusion_dim]

        node_features = self.gcn(node_features, edge_index)
        node_features = self.transformer(node_features, edge_index)

        recon_loss = 0
        if mask_autoencoding:
            num_nodes = node_features.size(0)
            mask = torch.rand(num_nodes, device=node_features.device) < self.mask_ratio
            target_features = node_features[mask]
            masked_features = node_features.clone()
            masked_features[mask] = 0
            recon_features = self.node_autoencoder(masked_features[mask])
            recon_loss = F.mse_loss(recon_features, target_features)

        predicted_edge_scores = self.edge_predictor(node_features[user],
                                                     node_features[wine + self.user_embedding.weight.size(0)])
        edge_pred_loss = F.mse_loss(predicted_edge_scores, torch.ones_like(predicted_edge_scores))

        user_out = node_features[user]
        wine_out = node_features[wine + self.user_embedding.weight.size(0)]
        interaction = torch.cat([user_out, wine_out], dim=1)
        score = self.fc(interaction).squeeze()
        score = torch.sigmoid(score) * 4 + 1

        return score, recon_loss, edge_pred_loss

#########################################
# 8. 최종 학습, 검증 및 테스트 파이프라인
#########################################
# 8-1. 학습 데이터: 전체 edge_index 구성 (Edge Augmentation 적용)
train_edge_index = torch.tensor(
    np.array([train_long_df["user_id"].to_numpy(), train_long_df["wine_id"].to_numpy()]),
    dtype=torch.long
)

train_dataset = WineDataset(train_long_df)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# 미리 전처리된 와인 사이드 정보: (num_items, num_side_tokens, side_token_dim)
# 여기서는 예시로 임의의 텐서를 생성 (예: num_total_items x 4 x 64)
num_side_tokens = 4
side_token_dim = 64
wine_side_features = torch.randn(num_total_items, num_side_tokens, side_token_dim)

# Attention 기반 융합 모듈 생성: fusion_dim = embedding_dim * num_heads = 64*4 = 256
fusion_dim = 256
side_fusion_module = WineSideFusion(base_dim=fusion_dim, side_dim=side_token_dim,
                                      fusion_dim=fusion_dim, num_side_tokens=num_side_tokens)

# G-Former 모델 (사이드 정보 포함) 생성
model = GFormerWithSide(num_total_users, num_total_items, wine_side_features, side_fusion_module,
                        embedding_dim=64, num_heads=4, mask_ratio=0.2, dropout_rate=0.2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
recon_weight = 0.5      # Reconstruction loss 가중치
edge_pred_weight = 0.5  # Edge Prediction loss 가중치

num_train_epochs = 10
print("===== TRAINING with Edge Augmentation + Side Information =====")
for epoch in range(num_train_epochs):
    model.train()
    total_loss = 0
    for user_batch, wine_batch, rating_batch in train_loader:
        optimizer.zero_grad()
        score, recon_loss, edge_pred_loss = model(user_batch, wine_batch, train_edge_index, mask_autoencoding=True)
        sup_loss = criterion(score, rating_batch)
        loss = sup_loss + recon_weight * recon_loss + edge_pred_weight * edge_pred_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training Epoch {epoch+1}/{num_train_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# 8-2. 검증 데이터 Fine-tuning (70% 데이터)
val_ft_edge_index = torch.tensor(
    np.array([val_train_df["user_id"].to_numpy(), val_train_df["wine_id"].to_numpy()]),
    dtype=torch.long
)
val_train_dataset = WineDataset(val_train_df)
val_train_loader = DataLoader(val_train_dataset, batch_size=1024, shuffle=True)

num_ft_epochs = 3
print("\n===== VALIDATION FINE-TUNING =====")
for epoch in range(num_ft_epochs):
    model.train()
    total_loss = 0
    for user_batch, wine_batch, rating_batch in val_train_loader:
        optimizer.zero_grad()
        score, recon_loss, edge_pred_loss = model(user_batch, wine_batch, val_ft_edge_index, mask_autoencoding=True)
        sup_loss = criterion(score, rating_batch)
        loss = sup_loss + recon_weight * recon_loss + edge_pred_weight * edge_pred_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Validation Fine-tuning Epoch {epoch+1}/{num_ft_epochs}, Loss: {total_loss / len(val_train_loader):.4f}")

# 8-3. 검증 Hold-out 데이터 평가 (30% 데이터) + NDCG 계산
print("\n===== VALIDATION EVALUATION =====")
with torch.no_grad():
    val_user_tensor = torch.tensor(val_test_df["user_id"].values, dtype=torch.long)
    val_wine_tensor = torch.tensor(val_test_df["wine_id"].values, dtype=torch.long)
    val_eval_edge_index = torch.tensor(
        np.array([val_test_df["user_id"].to_numpy(), val_test_df["wine_id"].to_numpy()]),
        dtype=torch.long
    )
    pred_score, _, _ = model(val_user_tensor, val_wine_tensor, val_eval_edge_index, mask_autoencoding=False)
    actual_val_ratings = torch.tensor(val_test_df["rating"].values, dtype=torch.float)

    pred_score_np = pred_score.cpu().numpy()
    actual_val_np = actual_val_ratings.cpu().numpy()
    user_ids_np = val_test_df["user_id"].values  # numpy array

    mse_val = mean_squared_error(actual_val_np, pred_score_np)
    rmse_val = np.sqrt(mse_val)
    ndcg_val = compute_ndcg(user_ids_np, pred_score_np, actual_val_np, k=10)
    print(f"Validation After Fine-tuning - MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, NDCG@10: {ndcg_val:.4f}")

# 8-4. 테스트 데이터 평가 (Edge Augmentation + Side 정보 적용된 모델 검증) + NDCG 계산
print("\n===== TEST EVALUATION =====")
with torch.no_grad():
    test_user_tensor = torch.tensor(test_test_df["user_id"].values, dtype=torch.long)
    test_wine_tensor = torch.tensor(test_test_df["wine_id"].values, dtype=torch.long)
    test_eval_edge_index = torch.tensor(
        np.array([test_test_df["user_id"].to_numpy(), test_test_df["wine_id"].to_numpy()]),
        dtype=torch.long
    )
    pred_score, _, _ = model(test_user_tensor, test_wine_tensor, test_eval_edge_index, mask_autoencoding=False)
    actual_test_ratings = torch.tensor(test_test_df["rating"].values, dtype=torch.float)

    pred_score_np = pred_score.cpu().numpy()
    actual_test_np = actual_test_ratings.cpu().numpy()
    user_ids_np = test_test_df["user_id"].values

    mse_test = mean_squared_error(actual_test_np, pred_score_np)
    rmse_test = np.sqrt(mse_test)
    ndcg_test = compute_ndcg(user_ids_np, pred_score_np, actual_test_np, k=10)
    print(f"Test Evaluation - MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, NDCG@10: {ndcg_test:.4f}")


# In[4]:


#########################################
# 6-5. NDCG@20 평가 코드 (수정됨)
#########################################
def dcg_at_k(ratings, k):
    """ DCG@k 계산 (Discounted Cumulative Gain) """
    ratings = np.asarray(ratings, dtype=np.float64)[:k]
    if ratings.size:
        return np.sum(ratings / np.log2(np.arange(2, ratings.size + 2)))
    return 0.0

def ndcg_at_k(actual_ratings, predicted_ratings, k=20):
    """ NDCG@20 계산 """
    ideal_ratings = np.sort(actual_ratings)[::-1]  # 이상적인 정렬 (내림차순)
    dcg = dcg_at_k(predicted_ratings, k)
    idcg = dcg_at_k(ideal_ratings, k)
    return dcg / idcg if idcg > 0 else 0

print("\n===== TEST NDCG@20 EVALUATION =====")
ndcg_scores = []

with torch.no_grad():
    # 테스트 데이터에 대한 예측 수행
    test_user_tensor = torch.tensor(test_test_df["user_id"].values, dtype=torch.long)
    test_wine_tensor = torch.tensor(test_test_df["wine_id"].values, dtype=torch.long)
    test_eval_edge_index = torch.tensor(
        np.array([test_test_df["user_id"].to_numpy(), test_test_df["wine_id"].to_numpy()]),
        dtype=torch.long
    )
    predicted_scores, _, _ = model(test_user_tensor, test_wine_tensor, test_eval_edge_index, mask_autoencoding=False)
    test_test_df["predicted_rating"] = predicted_scores.numpy()

    for user in test_test_df["user_id"].unique():
        user_data = test_test_df[test_test_df["user_id"] == user]
        actual_ratings = user_data["rating"].values
        predicted_ratings = user_data["predicted_rating"].values

        # 예측 점수 내림차순 정렬 후 상위 20개 항목 선택
        sorted_indices = np.argsort(predicted_ratings)[::-1]
        actual_ratings_sorted = actual_ratings[sorted_indices]

        ndcg = ndcg_at_k(actual_ratings_sorted, actual_ratings, k=20)
        ndcg_scores.append(ndcg)

mean_ndcg_20 = np.mean(ndcg_scores)
print(f"Test Evaluation - NDCG@20: {mean_ndcg_20:.4f}")

#########################################
# 6-6. Recall@20 평가 코드 (추가됨)
#########################################
print("\n===== TEST Recall@20 EVALUATION =====")
recall_scores = []
threshold = 3.0  # 예: 평점이 3 이상이면 'relevant'하다고 가정 (필요에 따라 조정)

with torch.no_grad():
    # 위와 같이 테스트 데이터에 대한 예측 수행
    test_user_tensor = torch.tensor(test_test_df["user_id"].values, dtype=torch.long)
    test_wine_tensor = torch.tensor(test_test_df["wine_id"].values, dtype=torch.long)
    test_eval_edge_index = torch.tensor(
        np.array([test_test_df["user_id"].to_numpy(), test_test_df["wine_id"].to_numpy()]),
        dtype=torch.long
    )
    predicted_scores, _, _ = model(test_user_tensor, test_wine_tensor, test_eval_edge_index, mask_autoencoding=False)
    test_test_df["predicted_rating"] = predicted_scores.numpy()

    for user in test_test_df["user_id"].unique():
        user_data = test_test_df[test_test_df["user_id"] == user]
        actual_ratings = user_data["rating"].values
        predicted_ratings = user_data["predicted_rating"].values

        # 이진 relevance: 평점이 threshold 이상이면 1, 그렇지 않으면 0
        relevant_mask = (actual_ratings >= threshold).astype(int)
        # 예측 점수를 기준으로 내림차순 정렬 후 상위 20개 항목 선택
        sorted_indices = np.argsort(predicted_ratings)[::-1]
        top_k_indices = sorted_indices[:20]

        total_relevant = np.sum(relevant_mask)
        if total_relevant == 0:
            recall = 0.0
        else:
            retrieved_relevant = np.sum(relevant_mask[top_k_indices])
            recall = retrieved_relevant / total_relevant

        recall_scores.append(recall)

mean_recall_20 = np.mean(recall_scores)
print(f"Test Evaluation - Recall@20: {mean_recall_20:.4f}")


# In[ ]:


#########################################
# TEST Precision@20 평가 코드 (추가됨)
#########################################
print("\n===== TEST Precision@13 EVALUATION =====")
precision_scores = []
threshold = 13.0  # 예: 평점이 3 이상이면 'relevant'하다고 가정 (필요에 따라 조정)

with torch.no_grad():
    # 테스트 데이터에 대한 예측 수행
    test_user_tensor = torch.tensor(test_test_df["user_id"].values, dtype=torch.long)
    test_wine_tensor = torch.tensor(test_test_df["wine_id"].values, dtype=torch.long)
    test_eval_edge_index = torch.tensor(
        np.array([test_test_df["user_id"].to_numpy(), test_test_df["wine_id"].to_numpy()]),
        dtype=torch.long
    )
    predicted_scores, _, _ = model(test_user_tensor, test_wine_tensor, test_eval_edge_index, mask_autoencoding=False)
    test_test_df["predicted_rating"] = predicted_scores.numpy()

    for user in test_test_df["user_id"].unique():
        user_data = test_test_df[test_test_df["user_id"] == user]
        actual_ratings = user_data["rating"].values
        predicted_ratings = user_data["predicted_rating"].values

        # 이진 relevance: 평점이 threshold 이상이면 1, 그렇지 않으면 0
        relevant_mask = (actual_ratings >= threshold).astype(int)

        # 예측 점수를 기준으로 내림차순 정렬 후 상위 20개 항목 선택
        sorted_indices = np.argsort(predicted_ratings)[::-1]
        top_k_indices = sorted_indices[:20]

        # Precision@20은 상위 20개 추천 항목 중 relevant 항목의 비율
        precision = np.sum(relevant_mask[top_k_indices]) / 20.0
        precision_scores.append(precision)

mean_precision_20 = np.mean(precision_scores)
print(f"Test Evaluation - Precision@20: {mean_precision_20:.4f}")


# In[ ]:




