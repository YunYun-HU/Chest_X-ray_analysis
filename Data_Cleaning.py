from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CXR8Dataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class DataCleaning:
    def __init__(self, csv_path, png_path, input_size=512):
        self.csv_path = csv_path
        self.png_map = png_path
        self.png_path = None
        self.input_size = input_size
        self.images = None

        #運用col
        self.DISEASES = [
            "Infiltration",
            "Effusion",
            "Atelectasis"
            "No Finding"
        ]

        #要刪除的col
        self.remove_labels = [
            "Fibrosis",
            "Pneumonia",
            "Hernia",
            "Edema",
            "Emphysema",
            "Cardiomegaly",
            "Pleural_Thickening",
            "Consolidation",
            "Pneumothorax",
            "Mass",
            "Nodule",
        ]

    # CSV 處理
    def clean_cxr8_csv(self):
        df = pd.read_csv(self.csv_path)


        # 刪除只要包含這些疾病的 row
        df = df[~df["Finding Labels"].apply(
            lambda x: any(label in x.split("|") for label in self.remove_labels)
        )]


        #保留有對應 png 的列
        df = df[df["Image Index"].isin(self.png_map)].copy()

        # 加入完整圖片路徑
        df["image_path"] = df["Image Index"].map(self.png_map)

        #多標籤轉
        for disease in self.DISEASES:
            df[disease] = df["Finding Labels"].fillna("").str.contains(disease).astype(np.float32)

        df = df.reset_index(drop=True)

        #對齊圖片順序
        self.png_path = df["image_path"].tolist()

        return df

    #圖片處理
    def clean_png(self):
        if self.images is not None:
            return self.images

        images = []

        for path in self.png_path:
            img = Image.open(path).convert("L")
            img = img.resize((self.input_size, self.input_size))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)   # (1, H, W)
            images.append(img)

        images = np.array(images, dtype=np.float32)      # (N, 1, H, W)
        self.images = torch.tensor(images, dtype=torch.float32)
        return self.images

    #切分資料集
    def split_data(self, df):
        picture = self.clean_png()   # tensor, shape = (N,1,H,W)
        lab = torch.tensor(df[self.DISEASES].values, dtype=torch.float32)

        # train_test_split 先轉 numpy
        picture_np = picture.numpy()
        lab_np = lab.numpy()

        picture_train, picture_temp, lab_train, lab_temp = train_test_split(
            picture_np,
            lab_np,
            test_size=0.3,
            random_state=42,
            shuffle=True
        )

        picture_val, picture_test, lab_val, lab_test = train_test_split(
            picture_temp,
            lab_temp,
            test_size=1/3,
            random_state=42,
            shuffle=True
        )

        #轉回 torch tensor
        picture_train = torch.tensor(picture_train, dtype=torch.float32)
        picture_val   = torch.tensor(picture_val, dtype=torch.float32)
        picture_test  = torch.tensor(picture_test, dtype=torch.float32)

        lab_train = torch.tensor(lab_train, dtype=torch.float32)
        lab_val   = torch.tensor(lab_val, dtype=torch.float32)
        lab_test  = torch.tensor(lab_test, dtype=torch.float32)

        return (
            picture_train, picture_val, picture_test,
            lab_train, lab_val, lab_test
        )

