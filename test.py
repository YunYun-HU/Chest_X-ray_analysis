import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent  
csv_path_ins = PROJECT_DIR / "CXR8" / "Data_Entry_2017_v2020.csv"
img_dir = PROJECT_DIR / "CXR8" / "images"



# 開檔
df = pd.read_csv(csv_path_ins)

# 看欄位名稱
print("Columns:")
print(df.columns.tolist())

# 資料筆數
print("\nTotal rows:", len(df))

# 各欄位非空值數量
print("\nNon-null count:")
print(df.count())

# Image Index 有幾個（總數 / 唯一）
print("\nImage Index:")
print("Total:", df["Image Index"].count())
print("Unique:", df["Image Index"].nunique())

# Finding Labels 各種類出現次數（原始整串）
print("\nFinding Labels value_counts:")
print(df["Finding Labels"].value_counts())

# 病灶拆開後各疾病出現次數
print("\nEach disease count:")

all_labels = df["Finding Labels"].fillna("").str.split("|")
flat_labels = all_labels.explode()

print(flat_labels.value_counts())

# Patient Sex 分布
print("\nPatient Sex:")
print(df["Patient Sex"].value_counts())

# View Position 分布
print("\nView Position:")
print(df["View Position"].value_counts())

# Patient Age 基本統計
print("\nPatient Age:")
print(df["Patient Age"].describe())