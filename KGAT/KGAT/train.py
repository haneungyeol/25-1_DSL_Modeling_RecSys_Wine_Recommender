import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from kgat_model import KGATModel
from bpr_dataset import BPRDataset
from kgat_config import LEARNING_RATE, EPOCHS, BATCH_SIZE, WEIGHT_DECAY

def bpr_loss(pos_scores, neg_scores, margin=0.3):
    """
    BPR Loss: margin ranking loss를 사용하여 긍정 샘플과 부정 샘플 간 점수 차이를 학습합니다.
    """
    return torch.mean(torch.clamp(margin - pos_scores + neg_scores, min=0))

def train_model(model, dataset, num_epochs=EPOCHS):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            # 배치로부터 사용자, 긍정 아이템, 부정 후보들을 가져옵니다.
            user, pos_item, neg_candidates = batch
            
            # 실제 모델에 맞게 전처리 후 forward pass를 수행합니다.
            # 아래 코드는 예시이므로, 실제 모델 아키텍처에 맞게 수정하세요.
            pos_scores = model(pos_item, neg_candidates)
            neg_scores = model(pos_item, neg_candidates)
            
            loss = bpr_loss(pos_scores, neg_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 최고 성능 모델 저장 로직 추가 가능
    return model

# __main__ 블록은 실제 학습 시 main 파일에서 호출하도록 생략합니다.
