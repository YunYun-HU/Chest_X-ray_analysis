from pathlib import Path
import pandas as pd

class Readfile:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.PROJECT_DIR = self.BASE_DIR.parent  
        self.csv_path_ins = self.PROJECT_DIR / "CXR8" / "Data_Entry_2017_v2020.csv"
        self.img_dir = self.PROJECT_DIR / "CXR8" / "images"


    #讀取csv
    def read_csv(self):
        return self.csv_path_ins
    

    #讀取png
    def read_png(self):
        png_map = {p.name: str(p) for p in self.img_dir.rglob("*.png")}
        return png_map
    
