import torch
import torch.nn.functional as f
import torch.nn as nn
import torch.nn.functional as F
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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import cv2
from sklearn.metrics import accuracy_score

# seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# class dataset
class Potholes_PRFDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        image_paths = self.df.iloc[:, 0]
        self.len = len(image_paths)
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),    # 必要な画像サイズにリサイズ
            transforms.ToTensor(),             # テンソルに変換
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

def score_to_prob(input_tensor):
    # 1から引いた新しい値を計算
    new_value = 1 - input_tensor
    # 新しい値と元の値を結合して2つのテンソルを作成
    combined_tensor = torch.cat((input_tensor, new_value), dim=1)
    return combined_tensor

# router
def sigmoid(logits, hard=False, threshold=0.5):
    y_soft = logits.sigmoid()
    if hard:
        indices = (y_soft < threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
class AdaptiveRouter(nn.Module):
    def __init__(self, features_channels, out_channels, reduction=4):
        super(AdaptiveRouter, self).__init__()
        self.inp = sum(features_channels)
        self.oup = out_channels
        self.reduction = reduction
        self.layer1 = nn.Conv2d(self.inp, self.inp//self.reduction, kernel_size=1, bias=True)
        self.layer2 = nn.Conv2d(self.inp//self.reduction, self.oup, kernel_size=1, bias=True)

    def forward(self, xs, thres=0.5):
        xs = [x.mean(dim=(2, 3), keepdim=True) for x in xs]
        xs = torch.cat(xs, dim=1)
        xs = self.layer1(xs)
        xs = F.relu(xs, inplace=True)
        xs = self.layer2(xs).flatten(1)
        xs = xs.sigmoid()
        return xs

def Train_Router(path, net, router, router_ins, epoch, loader, dotest=False):
    # 入力された乗数から、本当のモデルのノード数とする。
    set_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net.to(device)
    router.to(device)

    net.eval()

    kldiv = nn.KLDivLoss(reduction="batchmean")
    softmax = nn.Softmax(dim=1)
    # criterion = nn.CrossEntropyLoss()

    alpha = 1.0
    gamma = 3.0

    # criterion = FocalLoss(alpha=alpha, gamma=gamma)
    criterion = kldiv

    global history
    history = {
        'train_loss': [],
        'test_accuracy':[],
    }

    # 最適化関数とスケジューラーの定義
    lr_set = 1e-5
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr_set, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for e in range(epoch):
        loss = torch.tensor([0.0]).to(device)

        router.train()
        o_list = []
        gt_list = []
        for i, (images, model_prf) in enumerate(loader['train']):
            optimizer.zero_grad()
            images = images.to(device)
            images /= 255.0
            model_prf = model_prf.to(device)
            # ラベル作成
            f1_label = model_prf[:, [2, 5]]

            # router
            _, score = net.forward_score(images, router, router_ins)
            output = score_to_prob(score)

            # loss
            loss = criterion(softmax(output*100).log(), softmax(f1_label*100))
            # print(softmax(output*100), softmax(f1_label*100))
            loss.backward()
            optimizer.step()

            # 可視化用にリスト化

            o_list += [i.item() for i in torch.argmax(output, 1)]
            gt_list += [i.item() for i in torch.argmax(f1_label, 1)]

        # 学習率の更新
        scheduler.step()

        # lossの出力とモデル保存
        history['train_loss'].append(loss.item())
        print('Training log: {}, epoch：Loss: {}'.format(e + 1,loss.item()))

        torch.save(router.state_dict(), path)

        if dotest == True:
          router.eval()
          output_list = []
          test_loss_onehot_list = []
          for i, (images, model_prf) in enumerate(loader['test']):
              images = images.to(device)
              images /= 255.0
              model_prf = model_prf.to(device)
              # ラベル作成
              f1_label = model_prf[:, [2,5]]

              # router_output
              _, score = net.forward_score(images, router, router_ins)
              output = score_to_prob(score)

              output_list += [i.item() for i in torch.argmax(output, 1)]
              test_loss_onehot_list += [i.item() for i in torch.argmax(f1_label, 1)]

          print("test accuracy : ", accuracy_score(test_loss_onehot_list, output_list))
          print("y =",  test_loss_onehot_list)
          print("p =",  output_list)
          history["test_accuracy"].append(accuracy_score(test_loss_onehot_list, output_list))

    # 結果をプロット
    plt.figure()
    plt.plot(range(1, epoch+1), history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure()
    plt.plot(range(1, epoch+1), history['test_accuracy'])
    plt.title('Test Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    return router


if __name__ == '__main__':
    CSV_PATH = '/content/drive/MyDrive/修論/pothole/ECM_for_potholes/ecm_output_prf1.csv'
    BATCH_SIZE = 20
    YOLOV7_WEIGHT = "./ecm/model_path/potholedet_yolov7.pt"
    SAVE_ROUTER_PATH = "/content/router.pth"

    set_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load df
    data_df = pd.read_csv(CSV_PATH, index_col=0)

    # dataset
    image_dataset = Potholes_PRFDataset(data_df)
    # train_test_split：今回は分裂しているので使わない
    train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [int(len(data_df)*0.9), len(data_df)-int(len(data_df)*0.9)] )
    # train_dataset = image_dataset
    test_dataloader = ""

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        drop_last=False
    )
    # 辞書にまとめる
    dataloaders_dict = {'train': train_dataloader, 'test':test_dataloader}

    from models.experimental import attempt_load
    model = attempt_load(YOLOV7_WEIGHT, map_location=device)

    router_ins = [2, 11, 24, 37, 50]
    router_channels = [64, 256, 512, 1024, 1024]
    router = AdaptiveRouter(router_channels, 1).to(device)

    Train_Router(SAVE_ROUTER_PATH, model, router, router_ins, 50, dataloaders_dict, dotest=True)
