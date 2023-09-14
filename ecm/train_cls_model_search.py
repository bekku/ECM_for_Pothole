import torch
import torch.nn.functional as f
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
import random
import copy
import pandas as pd
from torch.autograd import Variable
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

from torch.utils.data import Dataset
from PIL import Image
import cv2

class COCO_PRFDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        image_paths = self.df.iloc[:, 0]
        self.len = len(image_paths)
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),    # 必要な画像サイズにリサイズ
            transforms.ToTensor(),             # テンソルに変換
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 平均と標準偏差で正規化
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # 1列目が画像ファイルのパス
        image_path = self.df.iloc[index, 0]

        # 画像の読込と前処理
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')  # グレースケール画像をRGBに変換
        image = self.transform(image)

        # モデルの性能の取得
        model_loss_list = self.df.iloc[index, 1:].values.astype(float)
        return image, torch.tensor(model_loss_list)


set_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_df = pd.read_csv('/content/drive/MyDrive/修論/pothole/train_dir/PR_select/pothole_yolo_output_prf1_by_conf.csv', index_col=0)