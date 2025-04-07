import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 데이터 로드
file_path = "wine_info_processed_quintiles.csv"
df = pd.read_csv(file_path, dtype=str)  # 모든 데이터를 문자열로 불러오기 (타입 문제 방지)
df = df.drop(columns=['URL'], errors='ignore')
df = df.drop(columns=['Wine Name'], errors='ignore')


# 결측치 처리 (결측값을 "Unknown"으로 대체)
df.fillna("Unknown", inplace=True)

# 모든 문자열 데이터를 숫자로 변환
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # 각 컬럼별 인코더 저장

# 타겟 변수 선택 (가장 관련 있는 변수 중 하나 선택)
target_col = "Average Rating"  # 예제에서는 가격을 기준으로 변수 중요도 평가
X = df.drop(columns=[target_col])
y = df[target_col]

# 랜덤 포레스트 모델 생성 및 학습
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 변수 중요도 추출
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# 중요도 시각화
plt.figure(figsize=(12, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance using Random Forest")
plt.gca().invert_yaxis()
plt.show()

# 상위 10개 변수 출력
print("Top 10 Important Features:")
print(feature_importances.head(10))


import seaborn as sns
import matplotlib.pyplot as plt

# 속성 간 상관관계 계산
corr_matrix = df.corr()

# 상관계수 히트맵 출력
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import chardet


# 📌 1️⃣ 데이터 로드
with open("wine_info_processed_quintiles.csv", "rb") as f:
    result = chardet.detect(f.read(100000))  # 처음 100000바이트만 읽어서 감지
    detected_encoding = result["encoding"]
    print(f"✅ 감지된 파일 인코딩: {detected_encoding}")

# 📌 감지된 인코딩으로 파일 읽기
wine_info = pd.read_csv("wine_info_processed_quintiles.csv", encoding=detected_encoding, encoding_errors="replace")

# 불필요한 컬럼 제거 (URL 등)
drop_cols = ["Wine Name", "URL", "Source File", "Search Page Link", "Actual Page Link"]
wine_info = wine_info.drop(columns=[col for col in drop_cols if col in wine_info.columns], errors="ignore")

# 📌 2️⃣ 상관관계 분석 (중복 속성 제거)
print("📢 상관관계 분석 진행 중...")

wine_info_encoded = wine_info.copy()
encoder = LabelEncoder()

for col in wine_info_encoded.columns:
    wine_info_encoded[col] = encoder.fit_transform(wine_info_encoded[col].astype(str))

# 상관계수 행렬 계산
corr_matrix = wine_info_encoded.corr()
high_corr_pairs = set()
to_remove = set()
selected_pairs = {}  # 어떤 속성이 유지되고, 어떤 속성이 제거되는지 저장

for col in corr_matrix.columns:
    for idx in corr_matrix.index:
        if col != idx and abs(corr_matrix.loc[idx, col]) > 0.85:
            # 이미 제거된 속성이면 스킵
            if col in to_remove or idx in to_remove:
                continue
            
            # 무조건 하나만 제거 (col을 기본적으로 제거, idx를 유지)
            to_remove.add(idx)
            selected_pairs[idx] = col  # col이 제거되고, idx가 유지됨

# 📢 제거되는 변수 출력 (유지된 변수도 함께 출력)
print("🚨 제거된 속성 목록:")
for removed, kept in selected_pairs.items():
    print(f"❌ {removed} 제거 (🔗 {kept} 유지, 상관계수 {corr_matrix.loc[removed, kept]:.2f})")

# 중복 속성 제거
wine_info_filtered = wine_info.drop(columns=to_remove)
print(f"✅ 제거된 중복 속성 개수: {len(to_remove)}")


# 📢 제거되는 변수 출력
print("🚨 제거된 속성 목록:")
for col1, col2 in high_corr_pairs:
    if col1 in to_remove:
        print(f"❌ {col1} 제거 (🔗 {col2}와 상관계수 {corr_matrix.loc[col1, col2]:.2f})")

wine_info_filtered = wine_info.drop(columns=to_remove)
print(f"✅ 제거된 중복 속성 개수: {len(to_remove)}")


# 📌 3️⃣ 랜덤 포레스트 기반 속성 중요도 분석
print("📢 랜덤 포레스트 기반 속성 중요도 분석 중...")
target_column = "Average Rating"
wine_info_filtered[target_column] = pd.to_numeric(wine_info_filtered[target_column], errors='coerce')
wine_info_filtered = wine_info_filtered.dropna(subset=[target_column])
wine_info_filtered[target_column] = wine_info_filtered[target_column].astype(float)

feature_columns = [col for col in wine_info_filtered.columns if col != target_column]

# 숫자로 변환
for col in feature_columns:
    wine_info_filtered[col] = encoder.fit_transform(wine_info_filtered[col].astype(str))

# 결측값 처리
imputer = SimpleImputer(strategy="most_frequent")
wine_info_filtered[feature_columns] = imputer.fit_transform(wine_info_filtered[feature_columns])

# 랜덤 포레스트 학습
X = wine_info_filtered[feature_columns]
y = wine_info_filtered[target_column].astype(float)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 속성 중요도 저장
rf_importance = pd.DataFrame({'Feature': feature_columns, 'RF_Importance': model.feature_importances_})
rf_importance = rf_importance.sort_values(by='RF_Importance', ascending=False)

# 📌 4️⃣ SVD 기반 네트워크 중요도 분석
print("📢 네트워크 기반 속성 중요도 분석 중...")
feature_matrix = wine_info_filtered.drop(columns=[target_column]).values
cosine_sim_matrix = cosine_similarity(feature_matrix.T)

# 특이값 분해(SVD) 수행
svd = TruncatedSVD(n_components=1)
svd.fit(cosine_sim_matrix)
importance_scores = np.abs(svd.components_[0])  # 절댓값을 취해 중요도 스코어화

# 네트워크 중요도 저장
network_importance = pd.DataFrame({"Feature": feature_columns, "Network_Importance": importance_scores})
network_importance = network_importance.sort_values(by="Network_Importance", ascending=False)

# 📌 5️⃣ 두 점수를 합산하여 최종 중요도 계산
print("📢 랜덤 포레스트 + 네트워크 점수 조합 중...")
importance_df = rf_importance.merge(network_importance, on="Feature")

# 점수 정규화
scaler = MinMaxScaler()
importance_df[["RF_Importance", "Network_Importance"]] = scaler.fit_transform(
    importance_df[["RF_Importance", "Network_Importance"]]
)

# 최종 점수 = 0.5 * RF + 0.5 * 네트워크 (가중치를 조정 가능)
importance_df["Final_Importance"] = (
    0.2 * importance_df["RF_Importance"] + 0.8 * importance_df["Network_Importance"]
)

# 정렬 후 최상위 속성 선택
importance_df = importance_df.sort_values(by="Final_Importance", ascending=False)
final_selected_features = importance_df.nlargest(10, "Final_Importance")["Feature"].tolist()

# 📌 6️⃣ 최종 속성 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x=importance_df["Final_Importance"][:10], y=importance_df["Feature"][:10], palette="coolwarm")
plt.xlabel("Final Importance Score")
plt.ylabel("Feature")
plt.title("Top 10 Features by Combined Importance (RF + SVD)")
plt.show()

print(f"🎯 최종 선정된 relation 속성 목록: {final_selected_features}")

# 📌 7️⃣ 최종 속성 저장
final_features_df = pd.DataFrame(final_selected_features, columns=["Selected Features"])
final_features_df.to_csv("wine/final_selected_features.csv", index=False)
print("✅ 최종 선정된 relation 속성이 'wine/final_selected_features.csv'에 저장되었습니다!")


