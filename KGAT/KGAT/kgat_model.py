import torch
import torch.nn as nn
import torch.nn.functional as F

class KGATModel(nn.Module):
    """
    KGATModel 클래스는 KGAT 모델의 네트워크 계층과 개선된 KG 통합 메커니즘을 정의합니다.
    """
    def __init__(self, num_items, num_entities, embedding_dim=64, dropout=0.5):
        super(KGATModel, self).__init__()
        self.num_items = num_items
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        
        # 아이템 임베딩
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # 엔티티 임베딩 (속성)
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        
        # 추가 네트워크 계층 (예시)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, item_indices, neighbor_indices):
        """
        item_indices: [batch_size]
        neighbor_indices: [batch_size, num_neighbors]
        """
        item_emb = self.item_embedding(item_indices)
        neighbor_emb = self.entity_embedding(neighbor_indices)
        # 이웃 임베딩의 평균 계산
        neighbor_emb = torch.mean(neighbor_emb, dim=1)
        # 아이템 임베딩과 결합
        x = item_emb + neighbor_emb
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

if __name__ == "__main__":
    # 간단한 실행 예제 (실제 사용 시에는 main 파일에서 실행)
    model = KGATModel(num_items=1000, num_entities=1500)
    sample_item = torch.LongTensor([1, 2, 3])
    sample_neighbors = torch.LongTensor([[10, 11, 12, 13, 14],
                                         [15, 16, 17, 18, 19],
                                         [20, 21, 22, 23, 24]])
    output = model(sample_item, sample_neighbors)
    print("모델 출력 shape:", output.shape)
