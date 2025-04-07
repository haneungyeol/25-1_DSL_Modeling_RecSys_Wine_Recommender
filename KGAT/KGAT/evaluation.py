import numpy as np

def precision_at_k(actual, predicted, k):
    """
    실제(actual)와 예측(predicted) 리스트에서 상위 k개 항목을 기준으로 precision을 계산합니다.
    """
    actual_set = set(actual)
    predicted = predicted[:k]
    return len(actual_set & set(predicted)) / float(k)

def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted = predicted[:k]
    return len(actual_set & set(predicted)) / float(len(actual_set)) if actual_set else 0.0

def ndcg_at_k(actual, predicted, k):
    def dcg(relevances):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
    actual_set = set(actual)
    predicted = predicted[:k]
    relevances = [1 if item in actual_set else 0 for item in predicted]
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg(ideal_relevances)
    return dcg(relevances) / idcg if idcg > 0 else 0.0

if __name__ == "__main__":
    # 평가 함수 테스트 예제 (실제 평가 시 main 파일 또는 별도 평가 스크립트에서 사용)
    actual = [1, 2, 3, 4]
    predicted = [2, 3, 5, 1]
    k = 3
    print("Precision@K:", precision_at_k(actual, predicted, k))
    print("Recall@K:", recall_at_k(actual, predicted, k))
    print("NDCG@K:", ndcg_at_k(actual, predicted, k))
