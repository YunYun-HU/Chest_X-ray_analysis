from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit


class CXR8Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DataCleaning:
    def __init__(self, csv_path, png_path, input_size, DISEASES=None):
        self.csv_path = csv_path
        self.png_map = png_path
        self.input_size = input_size

        self.DISEASES = DISEASES



    # CSV 處理
    def clean_cxr8_csv(self):
        df = pd.read_csv(self.csv_path)

        # 保留有對應 png 的列
        df = df[df["Image Index"].isin(self.png_map)].copy()

        # 加入完整圖片路徑
        df["image_path"] = df["Image Index"].map(self.png_map)

        # 多標籤轉換
        for disease in self.DISEASES:
            df[disease] = df["Finding Labels"].fillna("").str.contains(disease).astype(np.float32)

        df = df.reset_index(drop=True)

        return df



    # 讀取某個 df_part 的圖片
    def clean_png(self, df_part):
        images = []

        for path in df_part["image_path"]:
            img = Image.open(path).convert("L")
            img = img.resize((self.input_size, self.input_size))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)   # (1, H, W)
            images.append(img)

        images = np.array(images, dtype=np.float32)      # (N, 1, H, W)
        images = torch.tensor(images, dtype=torch.float32)
        return images
    



    # 把 df_part 轉成 tensor
    def build_xy_from_df(self, df_part):
        picture = self.clean_png(df_part)
        lab = torch.tensor(df_part[self.DISEASES].values, dtype=torch.float32)
        return picture, lab




    # 依Patient ID 切分資料集
    def split_data(self, df):
        groups = df["Patient ID"].values

        # 第一次切：train 70%, temp 30%
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, temp_idx = next(gss1.split(df, groups=groups))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        temp_df = df.iloc[temp_idx].reset_index(drop=True)

        # 第二次切：temp -> val 20%, test 10%
        temp_groups = temp_df["Patient ID"].values
        gss2 = GroupShuffleSplit(n_splits=1, test_size=1/3, random_state=42)
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_groups))

        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)

        # 轉成 tensor
        picture_train, lab_train = self.build_xy_from_df(train_df)
        picture_val, lab_val = self.build_xy_from_df(val_df)
        picture_test, lab_test = self.build_xy_from_df(test_df)

        return (
            picture_train, picture_val, picture_test,
            lab_train, lab_val, lab_test,
        )