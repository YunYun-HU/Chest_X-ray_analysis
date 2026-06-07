from PIL import Image
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torchvision import transforms
import torchvision.transforms.functional as TF


def decode_rle(rle, height, width):
    """
    將 ChestX-Ray8.csv 裡面的 RLE 字串解碼成 2D mask。
    """
    mask = np.zeros(height * width, dtype=np.uint8)

    if pd.isna(rle) or str(rle).strip() == "":
        return mask.reshape((height, width))

    rle = [int(x) for x in str(rle).strip().split()]

    starts = np.array(rle[0::2]) - 1
    lengths = np.array(rle[1::2])
    ends = starts + lengths

    for start, end in zip(starts, ends):
        mask[start:end] = 1

    #校準方向c
    return mask.reshape((height, width), order="C")


def build_lung_mask(left_rle, right_rle, height, width):
    left_mask = decode_rle(left_rle, height, width)
    right_mask = decode_rle(right_rle, height, width)
    lung_mask = np.logical_or(left_mask, right_mask).astype(np.uint8)
    return lung_mask


class DataCleaning:
    def __init__(self, csv_path, png_path, input_size, DISEASES=None, lung_csv_path=None, min_dice_mean=0.7):
        self.csv_path = csv_path
        self.png_map = png_path
        self.input_size = input_size

        self.DISEASES = DISEASES
        self.TARGETS = set(DISEASES)

        # 肺野切割 CSV，檔名預設由 OpenFile 傳入：CXR8/ChestX-Ray8.csv
        self.lung_csv_path = lung_csv_path
        self.min_dice_mean = min_dice_mean


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


    # 讀取肺野切割 CSV，只留下訓練需要的欄位
    def read_lung_csv(self):
        if self.lung_csv_path is None:
            return None

        lung_csv_path = Path(self.lung_csv_path)
        if not lung_csv_path.exists():
            print(f"找不到肺野切割檔：{lung_csv_path}，將不使用 lung mask")
            return None

        lung_df = pd.read_csv(lung_csv_path)

        # 統一改成 Image Index
        if "Image Index" not in lung_df.columns and "Image ID" in lung_df.columns:
            lung_df = lung_df.rename(columns={"Image ID": "Image Index"})

        keep_cols = [
            "Image Index",
            "Dice RCA (Mean)",
            "Left Lung",
            "Right Lung",
            "Height",
            "Width"
        ]

        missing_cols = [col for col in keep_cols if col not in lung_df.columns]
        if missing_cols:
            raise ValueError(f"肺野切割 CSV 缺少欄位：{missing_cols}")

        lung_df = lung_df[keep_cols].copy()

        # 過濾品質太低或 mask 不完整的資料
        lung_df = lung_df[lung_df["Dice RCA (Mean)"] >= self.min_dice_mean].copy()
        lung_df = lung_df.dropna(subset=["Left Lung", "Right Lung", "Height", "Width"])

        lung_df["Height"] = lung_df["Height"].astype(int)
        lung_df["Width"] = lung_df["Width"].astype(int)

        return lung_df


        # CSV 處理
    def clean_cxr8_csv(self):
        df = pd.read_csv(self.csv_path)

        # 保留有對應 png 的列
        df = df[df["Image Index"].isin(self.png_map)].copy()

        # 直接加入相對路徑
        df["image_path"] = df["Image Index"].map(self.png_map)

        # 合併肺野切割資料
        lung_df = self.read_lung_csv()
        if lung_df is not None:
            before_count = len(df)
            df = df.merge(lung_df, on="Image Index", how="inner")
            print(f"使用 lung mask：{before_count} -> {len(df)}")

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
    def __init__(self, csv_path, data_root, diseases, input_size=224, mode = "train", use_lung_mask=True, mask_outside_value=0.15, mask_as_channel=False):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.diseases = diseases
        self.mode = mode
        self.input_size = input_size
        self.use_lung_mask = use_lung_mask
        self.mask_outside_value = mask_outside_value
        self.mask_as_channel = mask_as_channel
        
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


    def has_lung_mask(self, row):
        need_cols = ["Left Lung", "Right Lung", "Height", "Width"]
        return all(col in row.index for col in need_cols)


    def load_lung_mask(self, row):
        lung_mask = build_lung_mask(
            left_rle=row["Left Lung"],
            right_rle=row["Right Lung"],
            height=int(row["Height"]),
            width=int(row["Width"])
        )
        return Image.fromarray((lung_mask * 255).astype(np.uint8))


    def transform_with_mask(self, image, mask):
        image = TF.resize(image, [self.input_size, self.input_size])
        mask = TF.resize(mask, [self.input_size, self.input_size], interpolation=transforms.InterpolationMode.NEAREST)

        # train 時 image 和 mask 必須做同一組隨機增強，否則位置會對不起來
        if self.mode == "train":
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.uniform(-7, 7)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

            max_dx = int(round(self.input_size * 0.03))
            max_dy = int(round(self.input_size * 0.03))
            translate = [
                random.randint(-max_dx, max_dx),
                random.randint(-max_dy, max_dy)
            ]
            scale = random.uniform(0.95, 1.05)
            shear = [0.0, 0.0]

            image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=shear)
            mask = TF.affine(
                mask,
                angle=0,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=transforms.InterpolationMode.NEAREST
            )

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. 讀圖片位置
        img_path = self.data_root / row["image_path"]
        image = Image.open(img_path).convert("L")

        # 2. 如果 csv 裡面有肺野切割欄位，就使用 Left Lung + Right Lung mask
        if self.use_lung_mask and self.has_lung_mask(row):
            mask = self.load_lung_mask(row)
            image, mask = self.transform_with_mask(image, mask)

            if self.mask_as_channel:
                # 原圖 + 肺野 mask 作為第二通道，shape: [2, H, W]
                image = torch.cat([image, mask], dim=0)
            else:
                # 原本 soft mask 做法
                # 肺內保留 100%，肺外保留 mask_outside_value，例如 0.15
                soft_mask = mask + self.mask_outside_value * (1.0 - mask)
                image = image * soft_mask
        else:
            image = self.transform(image)

            if self.mask_as_channel:
                # 沒有 mask 時補一張全 0 mask，避免 channel 數不一致
                empty_mask = torch.zeros_like(image)
                image = torch.cat([image, empty_mask], dim=0)

        # 3. 讀是否出現該特徵
        label = torch.tensor(
            row[self.diseases].values.astype("float32")
        )

        sample_weight = torch.tensor(
        row["sample_weight"],
        dtype=torch.float32
        )

        #label會對應到疾病的順序，ex: [0, 1, 0, 1] -> [Cardiomegaly, Effusion, Pneumothorax] = [沒有心臟肥大，有積液，沒有氣胸，有氣胸]
        return image, label, sample_weight
        
