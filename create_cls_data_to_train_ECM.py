# pythonでよく使われるライブラリ
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ニューラルネットワーク構築でよく使うライブラリ、他にもtfやkerasなど
import torch
import torch.nn as nn
import torchvision

# 画像分野でよく使われるライブラリ、他にopencvなどいろいろ
from PIL import Image
import glob

def yolo_ano_convert(x, y, w, h, size_x, size_y):
  x = float(x)
  y = float(y)
  w = float(w)
  h = float(h)
  size_x = float(size_x)
  size_y = float(size_y)
  xmin = x-(w/2)
  xmax = x+(w/2)
  ymin = y-(h/2)
  ymax = y+(h/2)

  xmin *= size_x
  xmax *= size_x
  ymin *= size_y
  ymax *= size_y
  return xmin, ymin, xmax, ymax


# save_dict = {"0":"/content/drive/MyDrive/2022-s/minamiku卒論/resnet_images_updata/hole/", 
#             "1":"/content/drive/MyDrive/2022-s/minamiku卒論/resnet_images_updata/manhole/", 
#             "2":"/content/drive/MyDrive/2022-s/minamiku卒論/resnet_images_updata/shadow/"}
# img_path = '/content/drive/MyDrive/2022-s/minamiku卒論/★train_dataset/★cg96gan96real885/images/'
# ano_path = '/content/drive/MyDrive/2022-s/minamiku卒論/★train_dataset/★cg96gan96real885/labels/'

def main():
    options = {'-img_path': True, '-ano_path': True, , '-save_path': True}
    args = {'img_path': None, 'ano_path': None, 'save_path': None}  # デフォルト値

    for key in options.keys():
        if key in sys.argv:
            index = sys.argv.index(key)
            if options[key]:
                value = sys.argv[index+1]
                if value.startswith('-'):
                    print("Option "+key+" must have a value.")
                    sys.exit()
                args[key[1:]] = value
                del sys.argv[index:index+2]
            else:
                args[key[1:]] = True
                del sys.argv[index]

    # ディレクトリのパスを指定
    save_dict = {"0":os.path.join(args["save_path"], "hole/"), 
                "1":os.path.join(args["save_path"], "manhole/"), 
                "2":os.path.join(args["save_path"], "shadow/"),}
    img_path = args["img_path"]
    ano_path = args["ano_path"]

    img_path_len = len(img_path)
    save_count = 0

    for i_img_path in glob.glob(img_path+"*.jpg"):
        img_pil = Image.open(img_path + i_img_path[img_path_len:])
        img_pil = img_pil.resize((640, 640))
        # 画像と一致するアノテーションファイルを開いて、リスト化
        with open(ano_path + i_img_path[img_path_len:-4]+".txt", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        
        for i_ano in lines:
            i_ano = list(i_ano.split())
            xmin, ymin, xmax, ymax = yolo_ano_convert(i_ano[1], i_ano[2], i_ano[3], i_ano[4], img_pil.size[0], img_pil.size[1])
            save_img = img_pil.crop((xmin, ymin, xmax, ymax))
            save_img = save_img.resize((224, 224))
            
            print(save_dict[i_ano[0]]+str(save_count)+".jpg")
            save_img.save(save_dict[i_ano[0]]+str(save_count)+".jpg", quality=95)
            save_count += 1

if __name__ == "__main__":
    main()
