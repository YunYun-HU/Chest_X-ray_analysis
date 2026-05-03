from pathlib import Path
import torch


class OpenFile:
    def __init__(self, number):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.PROJECT_DIR = self.BASE_DIR.parent

        self.CXR8_DIR = self.PROJECT_DIR / "CXR8"

        self.csv_path_ins = self.CXR8_DIR / "Data_Entry_2017_v2020.csv"
        self.img_dir = self.CXR8_DIR / "images"
        self.number = number

    # 讀取原始 CSV
    def read_csv(self):
        return self.csv_path_ins

    # 讀取 png，回傳相對於 CXR8 的路徑
    def read_png(self):
        png_map = {
            p.name: p.relative_to(self.CXR8_DIR).as_posix()
            for p in self.img_dir.rglob("*.png")
        }
        return png_map

    #---------------------------------------讀切割檔------------------------------------------------------------

    def read_train_df(self):
        return self.CXR8_DIR / f"set{self.number}" / "train.csv"

    def read_val_df(self):
        return self.CXR8_DIR / f"set{self.number}" / "val.csv"

    def read_test_df(self):
        return self.CXR8_DIR / f"set{self.number}" / "test.csv"

    #---------------------------------------做切割檔------------------------------------------------------------

    # 將清理後的 df 寫入新的 csv
    def new_file(self, df, num_DISEASES):
        csv_path_new = self.CXR8_DIR / f"set{self.number}" / f"source_{num_DISEASES}feature.csv"
        csv_path_new.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path_new, index=False, encoding="utf-8-sig")
        return df

    # 切分後的 train_df
    def split_train_df(self, train_df):
        save_path = self.CXR8_DIR / f"set{self.number}" / "train.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        return train_df

    # 切分後的 val_df
    def split_val_df(self, val_df):
        save_path = self.CXR8_DIR / f"set{self.number}" / "val.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        val_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        return val_df

    # 切分後的 test_df
    def split_test_df(self, test_df):
        save_path = self.CXR8_DIR / f"set{self.number}" / "test.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        test_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        return test_df