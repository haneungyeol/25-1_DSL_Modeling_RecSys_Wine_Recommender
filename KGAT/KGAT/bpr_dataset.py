import torch
from torch.utils.data import Dataset
import random

class BPRDataset(Dataset):
    """
    BPRDataset 클래스는 BPR 학습을 위한 데이터셋을 정의합니다.
    Hard Negative Sampling 개선을 포함합니다.
    """
    def __init__(self, user_item_interactions, num_items, n_candidates=3, positive_threshold=14):
        """
        user_item_interactions: dict, key: user, value: list of (item, rating)
        num_items: 전체 아이템 수
        n_candidates: 각 긍정 샘플 당 부정 후보 수
        positive_threshold: 긍정 상호작용 판단 기준
        """
        self.user_item_interactions = user_item_interactions
        self.num_items = num_items
        self.n_candidates = n_candidates
        self.positive_threshold = positive_threshold
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        samples = []
        for user, interactions in self.user_item_interactions.items():
            positive_items = [item for item, rating in interactions if rating >= self.positive_threshold]
            negative_items = [item for item, rating in interactions if rating < self.positive_threshold]
            for pos_item in positive_items:
                # Hard Negative Sampling: 후보군에서 무작위로 선택
                if negative_items:
                    neg_candidates = random.sample(negative_items, min(self.n_candidates, len(negative_items)))
                else:
                    neg_candidates = [random.randint(0, self.num_items - 1)]
                samples.append((user, pos_item, neg_candidates))
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user, pos_item, neg_candidates = self.samples[idx]
        return user, pos_item, neg_candidates

# __main__ 블록은 실제 데이터셋을 사용할 경우 main에서 실행하도록 생략합니다.
