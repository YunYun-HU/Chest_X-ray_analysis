from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torchvision import transforms


class DataCleaning:
    def __init__(self, csv_path, png_path, input_size, DISEASES=None):
        self.csv_path = csv_path
        self.png_map = png_path
        self.input_size = input_size

        self.DISEASES = DISEASES
        self.TARGETS = set(DISEASES)


    def findfrature(self, csv_path):
        all_labels = (
            csv_path["Finding Labels"]
            .dropna()
            .str.split("|")
            .explode()
            .str.strip()
        )

        # 找出所有不重複特徵
        unique_labels = sorted(all_labels.unique())

        print("共有幾種特徵：", len(unique_labels))
        print("所有特徵如下：")
        for label in unique_labels:
            print(label)

        label_counts = all_labels.value_counts()

        print("共有幾種特徵：", len(label_counts))
        print(label_counts)


    #切資料集
    def split_data(self, df):
        groups = df["Patient ID"].values

        # 第一次切：train 70%, temp 30%
        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=0.3,
            random_state=42
        )

        train_idx, temp_idx = next(gss1.split(df, groups=groups))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        temp_df = df.iloc[temp_idx].reset_index(drop=True)

        # 第二次切：temp -> val 20%, test 10%
        temp_groups = temp_df["Patient ID"].values

        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=1/3,
            random_state=42
        )

        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_groups))

        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)

        return train_df, val_df, test_df

    
    #計算樣本權重，給予多重疾病的樣本較低權重
    def get_sample_weight(self, label_str):
        labels = set(label_str.split("|"))
        others = labels - self.TARGETS - {"No Finding"}
        other_count = len(others)
        return 1.0 / (1.0 + other_count)


        # CSV 處理
    def clean_cxr8_csv(self):
        df = pd.read_csv(self.csv_path)

        # 保留有對應 png 的列
        df = df[df["Image Index"].isin(self.png_map)].copy()

        # 直接加入相對路徑
        df["image_path"] = df["Image Index"].map(self.png_map)

        # 多標籤轉換
        for disease in self.DISEASES:
            df[disease] = (
                df["Finding Labels"]
                .fillna("")
                .str.contains(disease, regex=False)
                .astype(np.float32)
            )
            
         #只保留目標疾病至少出現一種的資料
        df = df[df[self.DISEASES].sum(axis=1) > 0].copy()

        # 計算樣本權重
        df["sample_weight"] = df["Finding Labels"].apply(self.get_sample_weight)

        df = df.reset_index(drop=True)

        return df


# 數據轉換成pytorcch用格式
class DataForDenseNet(Dataset):
    def __init__(self, csv_path, data_root, diseases, input_size=224, mode = "train"):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.diseases = diseases
        self.mode = mode

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])

        
        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),

                # 資料增強：只給 train 用
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05)
                ),

                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. 讀圖片位置
        img_path = self.data_root / row["image_path"]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        # 2. 讀是否出現該特徵
        label = torch.tensor(
            row[self.diseases].values.astype("float32")
        )

        sample_weight = torch.tensor(
        row["sample_weight"],
        dtype=torch.float32
        )

        #label會對應到疾病的順序，ex: [0, 1, 0, 1] -> [Cardiomegaly, Effusion, Pneumothorax] = [沒有心臟肥大，有積液，沒有氣胸，有氣胸]
        return image, label, sample_weight
        