# 학습 관련 함수

def train(model, train_loader, optimizer, epoch, device):
    """
    MCCF 모델 학습 함수
    """
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch_idx, (batch_u, batch_i, batch_ratings, batch_wine_features) in enumerate(train_loader):
        optimizer.zero_grad()

        # ✅ 장치 할당
        batch_u = batch_u.to(device)
        batch_i = batch_i.to(device)
        batch_ratings = batch_ratings.to(device)
        batch_wine_features = batch_wine_features.to(device)

        # ✅ 손실 계산 및 역전파
        loss = model.compute_loss(batch_u, batch_i, batch_ratings, batch_wine_features)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:
            print(f"🔹 [Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.5f}")

    avg_loss = total_loss / batch_count
    print(f"✅ [Epoch {epoch}] 완료 | 평균 손실: {avg_loss:.5f}")

