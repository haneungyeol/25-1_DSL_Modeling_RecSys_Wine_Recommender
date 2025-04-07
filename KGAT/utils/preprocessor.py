import pandas as pd

# 파일 경로
wine_info_path = "wine_info_processed_quintiles.csv"
train_path = "train_data.csv"
val_path = "val_data.csv"
test_path = "test_data.csv"

# ===============================
# 1. item_map: 모든 와인명에 대한 ID
# ===============================
wine_info = pd.read_csv(wine_info_path)
wine_names = sorted(wine_info["Wine Name"].dropna().unique())
item_map = {name: idx for idx, name in enumerate(wine_names)}
pd.Series(item_map).to_csv("item_map_new.txt", sep="\t", header=False)

# ===============================
# 2. relation_map: 속성값에 대한 relation ID
# ===============================
attributes = ["Flavor Group 1", "Current Price_Quintile", "Region", "Winery"]

relation_set = set()
for attr in attributes:
    relation_set |= {f"{attr}: {val}" for val in wine_info[attr].dropna().unique()}

relation_list = sorted(relation_set)
relation_map = {rel: idx for idx, rel in enumerate(relation_list)}
pd.Series(relation_map).to_csv("relation_map_new.txt", sep="\t", header=False)

# ===============================
# 3. user_map: 유저 ID에 대한 정수 ID
# ===============================
# 모든 interaction 데이터 로딩
train = pd.read_csv(train_path)
val = pd.read_csv(val_path)
test = pd.read_csv(test_path)

all_user_ids = pd.concat([train["user"], val["user"], test["user"]]).dropna().unique()
user_list = sorted(all_user_ids)
user_map = {user_id: idx for idx, user_id in enumerate(user_list)}
pd.Series(user_map).to_csv("user_map_new.txt", sep="\t", header=False)

print("✅ item_map.txt, relation_map.txt, user_map.txt 생성 완료!")


import pandas as pd
import numpy as np

# 파일 경로
files = {
    "train_data.csv": "train_cf.txt",
    "val_data.csv": "val_cf.txt",
    "test_data.csv": "test_cf.txt"
}

# item_map 불러오기
item_map = pd.read_csv("item_map_new.txt", sep="\t", header=None, names=["Wine Name", "item_id"])
item_map_dict = dict(zip(item_map["Wine Name"], item_map["item_id"]))

# user_map 불러오기
user_map = pd.read_csv("user_map_new.txt", sep="\t", header=None, names=["user_id_raw", "user_id"])
user_map_dict = dict(zip(user_map["user_id_raw"], user_map["user_id"]))

# 모든 파일 처리
for input_csv, output_txt in files.items():
    df = pd.read_csv(input_csv)

    # melt: user_id_raw, wine_name, rating
    user_col = df.columns[0]
    wine_cols = df.columns[1:]
    melted = df.melt(id_vars=user_col, value_vars=wine_cols,
                     var_name="Wine Name", value_name="rating")

    # 결측치 제거
    melted = melted.dropna(subset=["rating"])

    # 평점 변환: log(rating + 1) * 10
    melted["rating"] = (np.log(melted["rating"] + 1) * 10).astype(int)

    # user_id 매핑 (문자열 → 정수)
    melted["user_id"] = melted[user_col].map(user_map_dict)

    # wine_name → item_id 매핑 후 정수 변환
    melted["item_id"] = melted["Wine Name"].map(item_map_dict)

    # NaN이 있는 행 삭제
    melted = melted.dropna(subset=["user_id", "item_id"])

    # item_id를 확실하게 int 형으로 변환
    melted["item_id"] = melted["item_id"].astype(int)

    # 필요한 열 정리
    cf_df = melted[["user_id", "item_id", "rating"]]

    # 저장
    cf_df.to_csv(output_txt, sep="\t", index=False, header=False)

print("✅ NaN 제거 및 정수 변환 완료 → train_cf_new.txt, val_cf_new.txt, test_cf_new.txt 생성됨!")


import pandas as pd
from itertools import product

# 파일 경로
wine_info_path = "wine_info_processed_quintiles.csv"
item_map_path = "item_map_new.txt"
relation_map_path = "relation_map_new.txt"
cf_files = ["train_cf.txt", "val_cf.txt", "test_cf.txt"]

# Step 1: item_map 불러오기 (Wine Name → item_id)
item_map = pd.read_csv(item_map_path, sep="\t", header=None, names=["Wine Name", "item_id"])
item_map_dict = dict(zip(item_map["Wine Name"], item_map["item_id"]))  # Name → ID 변환
reverse_item_map_dict = dict(zip(item_map["item_id"], item_map["Wine Name"]))  # ID → Name 변환

# Step 2: relation_map 불러오기 (Relation Name → relation_id)
relation_map = pd.read_csv(relation_map_path, sep="\t", header=None, names=["relation", "relation_id"])
relation_map_dict = dict(zip(relation_map["relation"], relation_map["relation_id"]))  # Name → ID 변환

print(f"🔎 relation_map.txt에서 불러온 relation 개수: {len(relation_map_dict)}")

# Step 3: CF 파일에서 사용된 item_id 수집
used_item_ids = set()
for file in cf_files:
    cf_data = pd.read_csv(file, sep="\t", header=None, usecols=[1], names=["item_id"])
    used_item_ids.update(cf_data["item_id"].unique())

print(f"🔎 CF에서 수집된 item_id 개수: {len(used_item_ids)}")

# Step 4: 사용된 item_id를 와인명으로 변환
used_wine_names = {reverse_item_map_dict[iid] for iid in used_item_ids if iid in reverse_item_map_dict}

print(f"🔎 CF에서 사용된 와인 개수: {len(used_wine_names)}")

# Step 5: wine_info에서 전체 데이터 로드
wine_info = pd.read_csv(wine_info_path)

# Step 6: ID 기반 트리플렛 생성 (Tail을 전체 데이터에서 찾도록 수정)
attributes = ["Flavor Group 1", "Current Price_Quintile", "Region"]
triplets = []

for attr in attributes:
    for value, group in wine_info.groupby(attr):  
        if isinstance(value, str) and value.lower() == "not available":  # 🔥 필터링 조건 추가
            print(f"🚨 {attr} 값이 'Not available'이므로 건너뜀")
            continue  # "Not available" 값은 트리플렛 생성 X

        wines = set(group["Wine Name"])  # 동일한 속성을 가진 모든 와인들
        heads = used_wine_names & wines  # CF에 등장한 와인들만 head
        tails = wines  # 전체 wine_info에서 tail을 포함하도록 수정

        print(f"🔎 속성: {attr}, 값: {value}, Head 개수: {len(heads)}, Tail 개수: {len(tails)}")

        for h, t in product(heads, tails):
            if h != t:  # **자기 자신으로의 연결 방지**
                relation = f"{attr}: {value}"
                if h in item_map_dict and t in item_map_dict and relation in relation_map_dict:
                    triplets.append((item_map_dict[h], relation_map_dict[relation], item_map_dict[t]))

print(f"🔎 생성된 트리플렛 개수: {len(triplets)}")

# Step 7: 저장
triplet_df = pd.DataFrame(triplets, columns=["head_id", "relation_id", "tail_id"])
print(triplet_df.head())
print(len(triplet_df))
triplet_df.to_csv("kg_triplets_id.txt", sep="\t", index=False, header=False)

print("✅ ID 기반 트리플렛 생성 완료 → kg_triplets_id.txt")


import pandas as pd
from itertools import product

# 파일 경로
wine_info_path = "wine_info_processed_quintiles.csv"
item_map_path = "item_map_new.txt"
cf_files = ["train_cf.txt", "val_cf.txt", "test_cf.txt"]

# Step 1: item_map 불러오기 (Wine Name → item_id)
item_map = pd.read_csv(item_map_path, sep="\t", header=None, names=["Wine Name", "item_id"])
item_map_dict = dict(zip(item_map["Wine Name"], item_map["item_id"]))  # Name → ID 변환
reverse_item_map_dict = dict(zip(item_map["item_id"], item_map["Wine Name"]))  # ID → Name 변환

# Step 2: CF 파일에서 사용된 item_id 수집
used_item_ids = set()
for file in cf_files:
    cf_data = pd.read_csv(file, sep="\t", header=None, usecols=[1], names=["item_id"])
    used_item_ids.update(cf_data["item_id"].unique())

print(f"🔎 CF에서 수집된 item_id 개수: {len(used_item_ids)}")

# Step 3: 사용된 item_id를 와인명으로 변환
used_wine_names = {reverse_item_map_dict[iid] for iid in used_item_ids if iid in reverse_item_map_dict}

print(f"🔎 CF에서 사용된 와인 개수: {len(used_wine_names)}")

# Step 4: wine_info에서 전체 데이터 로드
wine_info = pd.read_csv(wine_info_path)

# Step 5: ID 기반 트리플렛 생성 (단순화된 relation 사용)
attributes = ["Flavor Group 1", "Current Price_Quintile", "Region", "Winery"]
triplets = []

for attr in attributes:
    for value, group in wine_info.groupby(attr):  
        if isinstance(value, str) and value.lower() == "not available":  # "Not available" 제거
            continue  

        wines = set(group["Wine Name"])  # 동일한 속성을 가진 모든 와인들
        heads = used_wine_names & wines  # CF에 등장한 와인들만 head
        tails = wines  # 전체 wine_info에서 tail을 포함하도록 수정

        relation = f"same {attr}"  # 🔥 속성값 제거, relation을 단순화
        for h, t in product(heads, tails):
            if h != t and h in item_map_dict and t in item_map_dict:
                triplets.append((item_map_dict[h], relation, item_map_dict[t]))

print(f"🔎 생성된 트리플렛 개수: {len(triplets)}")

# Step 6: relation_map 생성 (새로운 단순화된 relation들 저장)
unique_relations = {rel for _, rel, _ in triplets}
relation_map_dict = {rel: idx for idx, rel in enumerate(sorted(unique_relations))}
relation_map_df = pd.DataFrame(relation_map_dict.items(), columns=["relation", "relation_id"])
relation_map_df.to_csv("relation_map_simplified.txt", sep="\t", index=False, header=False)

# Step 7: relation을 ID로 변환 후 저장
triplets_id = [(h, relation_map_dict[r], t) for h, r, t in triplets]
triplet_df = pd.DataFrame(triplets_id, columns=["head_id", "relation_id", "tail_id"])
triplet_df.to_csv("kg_triplets_simplified_id.txt", sep="\t", index=False, header=False)

print("✅ 단순화된 relation 기반 KG 데이터 생성 완료 → kg_triplets_simplified_id.txt")


import pandas as pd
import numpy as np

# 샘플 비율 설정 (예: 1% 수준)
sample_ratio = 0.01  # 혹은 sample_n = 1000 같이 개수 기준으로도 가능

# 파일 경로
cf_paths = {
    'train': 'train_cf.txt',
    'val': 'val_cf.txt',
    'test': 'test_cf.txt'
}
sampled_cf_paths = {
    name: f'{name}_cf_sample.txt' for name in cf_paths
}
kg_path = 'kg_triplets_id.txt'
kg_sampled_path = 'kg_triplets_sample.txt'

# 1️⃣ CF 샘플링
sampled_items = set()
for name, path in cf_paths.items():
    df = pd.read_csv(path, sep='\t', header=None, names=['user_id', 'item_id', 'rating'])
    
    # 유저 기준 샘플링
    unique_users = df['user_id'].unique()
    sampled_users = np.random.choice(unique_users, size=max(1, int(len(unique_users) * sample_ratio)), replace=False)
    sampled_df = df[df['user_id'].isin(sampled_users)].reset_index(drop=True)
    
    # 샘플링된 아이템 누적
    sampled_items.update(sampled_df['item_id'].unique())
    
    # 저장
    sampled_df.to_csv(sampled_cf_paths[name], sep='\t', header=False, index=False)
    print(f"✅ {name} 샘플링 완료 → {sampled_cf_paths[name]} ({len(sampled_df)} rows)")

# 2️⃣ KG 샘플링 (샘플링된 아이템만 포함)
sampled_items = set(map(int, sampled_items))  # 타입 일치 중요
kg_df = pd.read_csv(kg_path, sep='\t', header=None, names=['head_id', 'relation_id', 'tail_id'])

# head 또는 tail이 샘플링된 아이템이면 포함
kg_sampled = kg_df[(kg_df['head_id'].isin(sampled_items)) | (kg_df['tail_id'].isin(sampled_items))].reset_index(drop=True)

# 저장
kg_sampled.to_csv(kg_sampled_path, sep='\t', header=False, index=False)
print(f"✅ KG 샘플링 완료 → {kg_sampled_path} ({len(kg_sampled)} rows)")


