from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataCleaning:
    def __init__(self, csv_path, png_path, input_size=512):
        self.csv_path = csv_path 
        self.png_map  = png_path
        self.png_path = None   # 清洗完才放對齊後的圖片 list
        self.input_size = input_size

        self.DISEASES = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Effusion",
            "Emphysema",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumothorax"
        ]
  

    #csv處理
    def clean_cxr8_csv(self):
        df = pd.read_csv(self.csv_path)

        #取得路徑
        png_map = self.png_map

        #保留png列
        df = df[df["Image Index"].isin(png_map)].copy()

        #加入完整圖片路徑
        df["image_path"] = df["Image Index"].map(png_map)

        #多標籤轉 0/1
        for disease in self.DISEASES:
            df[disease] = df["Finding Labels"].fillna("").str.contains(disease).astype(np.float32)

        df = df.reset_index(drop=True)

        #cnn要讀的路徑直接用這個對齊後的順序
        self.png_path = df["image_path"].tolist()

        return df



     #CNN的圖片處理
    def clean_png(self):
        #加速跳過
        if hasattr(self, "images") and self.images is not None:
            return self.images

        images = []

        for path in self.png_path:
            img = Image.open(path).convert("L")
            img = img.resize((self.input_size, self.input_size))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)   # (H, W, 1)
            images.append(img)

        return np.array(images)   # (N, H, W, 1)



    # 切分資料集
    def split_data(self, df):
        #X圖片
        X = self.clean_png()

        #y多標籤
        y = df[self.DISEASES].values.astype(np.float32)

        #切 train + temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        #切 val + test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1/3, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test