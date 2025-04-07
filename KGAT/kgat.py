# %% [code]
# 1. 하이퍼파라미터 설정
NUM_NEIGHBORS = 5         # 각 아이템에 연결할 이웃(속성) 수
EMBEDDING_DIM = 64        # 임베딩 차원
DROPOUT = 0.5             # 드롭아웃 비율
NUM_USERS = 466          # 전체 고유 사용자 수 (train, val, test에 등장하는 모든 유저)
LEARNING_RATE = 0.001     # 학습률
EPOCHS = 30                # 학습 에폭 수
BATCH_SIZE = 128           # 배치 사이즈
PAD_TOKEN = -1            # 이웃이 부족할 경우 패딩 토큰
POSITIVE_THRESHOLD = 14   # 평점이 이 값 이상이면 긍정 상호작용으로 간주
K = 20                    # 평가 시 상위 K 추천 아이템
N_CANDIDATES = 3          # 각 긍정 샘플 당 부정 후보 수 (hard negative 샘플링용)
MARGIN = 0.3              # margin ranking loss에 사용할 마진 값
LAMBDA_MARGIN = 0.1       # BPR loss와 margin loss의 가중치 조절
WEIGHT_DECAY = 1e-5       # Adam 옵티마이저의 weight decay (L2 정규화)

# %% [code]
# 2. 와인 정보 데이터 로딩 및 KG_adj 구성
import torch
import pandas as pd
import random

# 디바이스 설정 (GPU 사용 가능 시 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 와인 정보 데이터셋 로드 (파일 경로에 맞게 수정)
wine_data = pd.read_csv("wine_info_processed_quintiles.csv")
print("와인 데이터 미리보기:")
print(wine_data.head())
print("Wine dataset shape:", wine_data.shape)

# 사용할 속성 선택 (예: Country, Variety, Winery)
selected_columns = ["Flavor Group 1", "Current Price_Quintile", "Region", "Winery", "Wine style"]

# 각 행(와인)을 하나의 아이템으로 간주
num_items = wine_data.shape[0]

# 각 아이템에 연결될 속성(엔티티) 리스트 생성
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

# 각 아이템마다 고정 개수(NUM_NEIGHBORS)의 이웃 구성 (부족 시 PAD_TOKEN으로 채움)
kg_adj_list = []
for neighbors in kg_neighbors:
    if len(neighbors) >= NUM_NEIGHBORS:
        selected = random.sample(neighbors, NUM_NEIGHBORS)
    else:
        selected = neighbors + [PAD_TOKEN] * (NUM_NEIGHBORS - len(neighbors))
    kg_adj_list.append(selected)

# kg_adj: (num_items x NUM_NEIGHBORS) 텐서
kg_adj = torch.tensor(kg_adj_list, device=device)
print("kg_adj shape:", kg_adj.shape)

# 전체 엔티티 수 = 아이템 수 + 속성 엔티티 수
num_entities = current_entity_id
print("총 엔티티 수:", num_entities)


# %% [code]
# 3. CF(사용자–아이템–평점) 데이터 로딩 (train, val, test)
# 파일은 탭(\t) 구분 텍스트 파일이며, 컬럼 순서는 user, item, rating이라고 가정합니다.
train_cf = pd.read_csv("train_cf.txt", sep="\t", header=None, names=["user", "item", "rating"])
val_cf = pd.read_csv("val_cf.txt", sep="\t", header=None, names=["user", "item", "rating"])
test_cf = pd.read_csv("test_cf.txt", sep="\t", header=None, names=["user", "item", "rating"])

print("Train CF 데이터 미리보기:")
print(train_cf.head())
print("Train CF shape:", train_cf.shape)

print("Validation CF 데이터 미리보기:")
print(val_cf.head())
print("Val CF shape:", val_cf.shape)

print("Test CF 데이터 미리보기:")
print(test_cf.head())
print("Test CF shape:", test_cf.shape)


# %% [code]
# 4. KGAT 모델 정의 (추가 네트워크 계층 및 개선된 KG 통합)
import torch.nn as nn
import torch.nn.functional as F

class KGATEnhanced(nn.Module):
    def __init__(self, num_users, num_items, num_entities, embedding_dim, dropout, num_neighbors):
        super(KGATEnhanced, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.num_neighbors = num_neighbors
        
        # 기본 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        
        # 기존 어텐션 계층
        self.attention_linear = nn.Linear(embedding_dim * 2, 1)
        
        # 추가: KG 정보 정제를 위한 MLP (두 층)
        self.kg_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 추가: 아이템 임베딩과 정제된 KG 정보를 결합하는 MLP (출력 차원: embedding_dim)
        self.combined_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 추가: 사용자와 최종 아이템 표현을 결합해 점수를 산출하는 출력 계층
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 1)
        )
        
    def forward(self, user_indices, item_indices, kg_adj_batch):
        # 사용자 및 아이템 임베딩 조회
        user_emb = self.user_embedding(user_indices)    # (B, embedding_dim)
        item_emb = self.item_embedding(item_indices)      # (B, embedding_dim)
        
        # KG: 인접 리스트를 통한 이웃 임베딩 조회
        kg_adj_clamped = kg_adj_batch.clamp(min=0)         # (B, num_neighbors)
        neighbor_emb = self.entity_embedding(kg_adj_clamped) # (B, num_neighbors, embedding_dim)
        
        # 어텐션 메커니즘: 아이템 임베딩을 각 이웃에 대해 확장 후 결합
        item_rep_exp = item_emb.unsqueeze(1).expand(-1, self.num_neighbors, -1)  # (B, num_neighbors, embedding_dim)
        attn_input = torch.cat([item_rep_exp, neighbor_emb], dim=-1)              # (B, num_neighbors, 2*embedding_dim)
        attn_scores = self.attention_linear(attn_input)                           # (B, num_neighbors, 1)
        attn_scores = F.leaky_relu(attn_scores, negative_slope=0.2)
        attn_weights = F.softmax(attn_scores, dim=1)                               # (B, num_neighbors, 1)
        
        # 가중합: 이웃 임베딩의 어텐션 기반 집계
        neighbor_attn = (attn_weights * neighbor_emb).sum(dim=1)                  # (B, embedding_dim)
        
        # 추가: KG MLP를 통해 이웃 정보 정제
        refined_kg = self.kg_mlp(neighbor_attn)                                   # (B, embedding_dim)
        
        # 아이템 임베딩과 정제된 KG 정보를 결합
        combined_item = torch.cat([item_emb, refined_kg], dim=1)                  # (B, 2*embedding_dim)
        final_item_rep = self.combined_mlp(combined_item)                         # (B, embedding_dim)
        
        # 최종: 사용자 임베딩과 최종 아이템 표현 결합 후, 출력 계층을 통해 점수 산출
        combined_user_item = torch.cat([user_emb, final_item_rep], dim=1)         # (B, 2*embedding_dim)
        score = self.output_layer(combined_user_item).squeeze(1)                  # (B,)
        return score

# 모델 생성 및 옵티마이저 설정 (weight decay 적용)
model = KGATEnhanced(NUM_USERS, num_items, num_entities, EMBEDDING_DIM, DROPOUT, NUM_NEIGHBORS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
print("KGATEnhanced 모델 생성 완료!")


# %% [code]
# 5. BPRDataset 정의 (Hard Negative Sampling 개선)
from torch.utils.data import Dataset

class BPRDataset(Dataset):
    def __init__(self, train_df, num_items, pos_threshold, n_candidates=5):
        # 긍정적 상호작용 필터링 (rating >= pos_threshold)
        self.train_df = train_df[train_df['rating'] >= pos_threshold]
        self.num_items = num_items
        self.n_candidates = n_candidates
        # 사용자별 상호작용 아이템 집합 구축
        self.user2items = {}
        for row in train_df.itertuples(index=False):
            user, item, rating = row
            if user not in self.user2items:
                self.user2items[user] = set()
            self.user2items[user].add(item)
        self.data = list(self.train_df.itertuples(index=False))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user, pos_item, rating = self.data[idx]
        candidate_negatives = []
        for _ in range(self.n_candidates):
            while True:
                neg_item = random.randint(0, self.num_items - 1)
                if neg_item not in self.user2items.get(user, set()):
                    candidate_negatives.append(neg_item)
                    break
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(candidate_negatives, dtype=torch.long))  # shape: (n_candidates,)


# %% [code]
# 6. 평가 함수 정의: precision@K, recall@K, ndcg@K 계산
import numpy as np

def precision_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    if len(recommended_k) == 0:
        return 0.0
    return len(set(recommended_k) & set(ground_truth)) / k

def recall_at_k(recommended, ground_truth, k):
    if len(ground_truth) == 0:
        return 0.0
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(ground_truth)) / len(ground_truth)

def ndcg_at_k(recommended, ground_truth, k):
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, rec in enumerate(recommended_k):
        if rec in ground_truth:
            dcg += 1 / np.log2(i + 2)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum([1 / np.log2(i + 2) for i in range(ideal_hits)])
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(model, val_df, kg_adj, num_items, pos_threshold, k):
    model.eval()
    user_group = val_df.groupby("user")
    precision_list, recall_list, ndcg_list = [], [], []
    
    with torch.no_grad():
        for user, group in user_group:
            gt_items = group[group['rating'] >= pos_threshold]['item'].tolist()
            if len(gt_items) == 0:
                continue
            
            candidate_items = torch.arange(0, num_items, dtype=torch.long, device=device)
            user_tensor = torch.full((num_items,), user, dtype=torch.long, device=device)
            candidate_kg_adj = kg_adj[candidate_items]
            
            scores = model(user_tensor, candidate_items, candidate_kg_adj)
            scores = scores.cpu().numpy()
            recommended = np.argsort(-scores).tolist()
            
            prec = precision_at_k(recommended, gt_items, k)
            rec = recall_at_k(recommended, gt_items, k)
            ndcg = ndcg_at_k(recommended, gt_items, k)
            precision_list.append(prec)
            recall_list.append(rec)
            ndcg_list.append(ndcg)
    
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    return avg_precision, avg_recall, avg_ndcg


# %% [code]
# 7. BPR loss 기반 학습 (Hard Negative Mining + Margin Ranking Loss) 및 최고 성능 기록
from torch.utils.data import DataLoader

def bpr_loss(pos_score, neg_score, eps=1e-8):
    return -torch.log(torch.sigmoid(pos_score - neg_score) + eps).mean()

def margin_ranking_loss(pos_score, neg_score, margin):
    return torch.relu(margin - (pos_score - neg_score)).mean()

bpr_dataset = BPRDataset(train_cf, num_items, POSITIVE_THRESHOLD, n_candidates=N_CANDIDATES)
bpr_loader = DataLoader(bpr_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 최고 성능 기록을 위한 변수 초기화 (여기서는 주로 recall을 기준으로 함)
best_val_recall = 0.0
best_val_precision = 0.0
best_val_ndcg = 0.0
best_loss = float('inf')
best_epoch = -1

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in bpr_loader:
        batch_user, batch_pos_item, batch_candidate_negatives = [x.to(device) for x in batch]
        
        pos_kg_adj = kg_adj[batch_pos_item]
        pos_score = model(batch_user, batch_pos_item, pos_kg_adj)
        
        B, n_candidates = batch_candidate_negatives.shape
        flat_candidate_neg = batch_candidate_negatives.view(-1)
        repeated_user = batch_user.unsqueeze(1).expand(-1, n_candidates).reshape(-1)
        
        neg_kg_adj = kg_adj[flat_candidate_neg]
        neg_scores_flat = model(repeated_user, flat_candidate_neg, neg_kg_adj)
        neg_scores = neg_scores_flat.view(B, n_candidates)
        
        # Hard negative: 후보 중 가장 높은 점수를 가진 부정 샘플 선택
        hard_neg_score, _ = neg_scores.max(dim=1)
        
        loss_bpr = bpr_loss(pos_score, hard_neg_score)
        loss_margin = margin_ranking_loss(pos_score, hard_neg_score, MARGIN)
        loss = loss_bpr + LAMBDA_MARGIN * loss_margin
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(bpr_loader)
    
    # 매 epoch마다 Validation 평가 진행
    val_precision, val_recall, val_ndcg = evaluate_model(model, val_cf, kg_adj, num_items, POSITIVE_THRESHOLD, K)
    
    # 최고 성능 갱신 시 기록 (여기서는 val recall를 기준으로 함)
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_val_precision = val_precision
        best_val_ndcg = val_ndcg
        best_loss = avg_loss
        best_epoch = epoch + 1
        with open("best_performance.txt", "w") as f:
            f.write(f"Best performance at epoch: {best_epoch}\n")
            f.write(f"Loss: {best_loss:.4f}\n")
            f.write(f"Val Precision@{K}: {best_val_precision:.4f}\n")
            f.write(f"Val Recall@{K}: {best_val_recall:.4f}\n")
            f.write(f"Val NDCG@{K}: {best_val_ndcg:.4f}\n")
            f.write("Hyperparameters:\n")
            f.write(f"NUM_NEIGHBORS: {NUM_NEIGHBORS}\n")
            f.write(f"EMBEDDING_DIM: {EMBEDDING_DIM}\n")
            f.write(f"DROPOUT: {DROPOUT}\n")
            f.write(f"NUM_USERS: {NUM_USERS}\n")
            f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
            f.write(f"EPOCHS: {EPOCHS}\n")
            f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
            f.write(f"PAD_TOKEN: {PAD_TOKEN}\n")
            f.write(f"POSITIVE_THRESHOLD: {POSITIVE_THRESHOLD}\n")
            f.write(f"K: {K}\n")
            f.write(f"N_CANDIDATES: {N_CANDIDATES}\n")
            f.write(f"MARGIN: {MARGIN}\n")
            f.write(f"LAMBDA_MARGIN: {LAMBDA_MARGIN}\n")
            f.write(f"WEIGHT_DECAY: {WEIGHT_DECAY}\n")
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, "
                 f"Val Precision@{K}: {val_precision:.4f}, "
                 f"Val Recall@{K}: {val_recall:.4f}, "
                 f"Val NDCG@{K}: {val_ndcg:.4f}")


# %% [code]
# 8. (선택 사항) Test 데이터 평가
test_precision, test_recall, test_ndcg = evaluate_model(model, test_cf, kg_adj, num_items, POSITIVE_THRESHOLD, K)
print(f"Test Precision@{K}: {test_precision:.4f}")
print(f"Test Recall@{K}: {test_recall:.4f}")
print(f"Test NDCG@{K}: {test_ndcg:.4f}")
