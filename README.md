# ECM for pothole

## 1. train

### train p5 models
```
python train.py --device 0 --epoch 100 --batch-size 8 --data data/potholes.yaml --img 640 640 --seed 1 \
                --cfg cfg/training/yolov7-tiny.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

## 2. evaluation
<ins>ここに記載される評価コードは, クラスの0番目が最大となるPやRを出力するため, potholeのクラスは0番目にする. 詳細はap_per_classにて.</ins>
```
data/potholes.yaml

train: ./ecm/data/train_dataset/
val: ./ecm/data/test_dataset/
# Classes
nc: 3  # number of classes
names: ["pothole", "manhole", "shadow"]
```

### 2.1 test.py : yoloの性能評価
```
python test.py --data ./data/potholes.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 \
               --weights ./ecm/model_path/potholedet_yolov7.pt --name yolov7_640_val
```


### 2.2 test_ECM.py : yolo + ECMの性能評価
```
python test_ECM.py --data ./data/potholes.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 \
                   --weights ./ecm/model_path/potholedet_yolov7.pt --name yolov7_640_val
```
```
python test_ECM.py --data ./data/potholes.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0 \
                    --weights ./ecm/model_path/potholedet_yolov7.pt --name yolov7_640_val --ecm_th 0.9
```
ecm_th：ecm(cls)側, potholeと分類した確信度に閾値パラメータである. この値未満の場合は, potholeとみなさない。


### 2.3 test_model_search.py : ECMを実行するかどうかをランダムで決めた場合の性能評価
```
python test_model_search.py --data ./data/potholes.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 \
                            --device 0 --seed 1--weights ./ecm/model_path/potholedet_yolov7.pt \
                            --name yolov7_640_val --model_search "random" --random_p 0.1
```
必ず, model_searchで"random"と設定する.
random_p：指定された確率でECMを選択する.


### 2.4 test_dynamic_det.py
```
python test_dynamic_det.py --data ./data/potholes.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 --device 0  \
                           --weights ./ecm/model_path/potholedet_yolov7.pt --name yolov7_640_val --router_th 0.51 \
                           --router_path /content/router_10ep.pth
```
router_th：routerが出力するscoreの閾値である. この閾値よりも大きければ, ECMを実行する.

```
test_ECM, test_dynamic_detには, ecm_path が存在する.
test_model_searchには, router_model_pathが存在し, efficientnet-b0のモデルのpathを入力することができる.
```
### 2.5 test_wbf.py : yoloにおけるwbfでの性能評価
```
python test.py --data ./data/potholes.yaml --img 640 --batch 1 --conf 0.001 --iou 0.65 \
               --weights ./ecm/model_path/potholedet_yolov7.pt --name yolov7_640_val \
               --weights2 ./ecm/model_path/potholedet_yolov7_wbf.pt
```
