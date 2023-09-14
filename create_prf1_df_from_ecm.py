import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, all_ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
import timm
import copy
import torchvision
import pandas as pd

def f1_calc(a,b):
    if a+b==0:
      return 0
    else:
      return 2*a*b/(a+b)

# Model validation metrics

from utils import general
from utils.metrics import *

# PrecisonとRecall, F1の計算を行う関数
def p_r_f1_calc(ecm, model, img, targets, paths, shapes, pr_th=None, conf_thres=0.001, iou_thres=0.65):
    # パラメータ
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []

    img = img.to(device, non_blocking=True)
# ====================================
    im_to_ecm = copy.deepcopy(img)
# ====================================
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    targets = targets.to(device)
    nb, _, height, width = img.shape  # batch size, channels, height, width

    with torch.no_grad():
        # Run model
        out, train_out = model(img, augment=False)  # inference and training outputs

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if False else []  # for autolabelling
        out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

# ============================================================================================================================
#   ECM
    if ecm:
        preds = copy.deepcopy(out)
        if True:
            convert_tensor = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224, 224), antialias=None),
                torchvision.transforms.Normalize(
                    mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ])
            for batch_num in range(len(preds)):
                pil_image = torchvision.transforms.functional.to_pil_image(im_to_ecm[batch_num])
                np_preds = np.array(preds[batch_num].to("cpu"))
                # np_predsから削除した物をreturn_predsにいれる。最後にnp化する。
                return_preds = []
                for object_bbox in np_preds:
                    # yoloが穴ではないと判別した時、そのまま追加。
                    if int(object_bbox[-1])!=0:
                        return_preds.append(object_bbox)
                    
                    # yoloが穴だと認識した時、resnetが穴というなら穴。
                    else:
                        crop_image = pil_image.crop((object_bbox[0], object_bbox[1], object_bbox[2], object_bbox[3]))
                        convert_crop_image = convert_tensor(crop_image)
                        # resnetに入力
                        resnet_results = resnet(convert_crop_image.unsqueeze(0).to(device))
                        resnet_pred_label = torch.argmax(resnet_results[0])
                        # resnetが穴だと認識したら、検出結果を含める。
                        # print("resnetの結果, yoloの結果 : ", resnet_pred_label.item(), object_bbox[-1])
                        if resnet_pred_label.item() == 0:
                            return_preds.append(object_bbox)
                        else:
                            object_bbox[-1] = resnet_pred_label.item()
                            return_preds.append(object_bbox)
                np_return_preds = np.array(return_preds)
                preds[batch_num] = torch.from_numpy(np_return_preds.astype(np.float32)).clone().to(device)

        out = preds
# ============================================================================================================================


    # Statistics per image
    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        path = Path(paths[si])
        if len(pred) == 0:
            # if nl:
            #     stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), torch.ones_like(pred[:, 4]).cpu(), pred[:, 5].cpu(), tcls))
        # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # all_ap_per_class
    # i番目のstatsのstats_i
    p_list = []
    r_list = []
    f1_list = []
    # max_i_list = []
    for i_num in range(len(stats)):
        stats_i = stats[i_num:i_num+1]
        stats_i = [np.concatenate(x, 0) for x in zip(*stats_i)]  # to numpy
        # Compute statistics
        if len(stats_i) and stats_i[0].any():
            p, r, ap, f1, max_i, ap_class = all_ap_per_class(*stats_i, plot=plots, v5_metric=False, save_dir=Path(''), names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mean_p, mean_r, mean_f1, map50, map = p.mean(), r.mean(), f1.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats_i[3].astype(np.int64), minlength=nc)  # number of targets per class
            mp, mr, mf1, map50, map = p[:, max_i].mean(), r[:, max_i].mean(), f1[:, max_i].mean(), map50, map

            if pr_th is None:
                p_list.append(mp)
                r_list.append(mr)
                f1_list.append(f1_calc(mp,mr))
            else:
                p_list.append(p[:, pr_th].mean())
                r_list.append(r[:, pr_th].mean())
                f1_list.append(f1_calc(p[:, pr_th].mean(),r[:, pr_th].mean()))
                # print(p[:, pr_th].mean(), r[:, pr_th].mean())

        else:
            nt = torch.zeros(1)
            p_list.append(0.0)
            r_list.append(0.0)
            f1_list.append(0.0)
    return torch.tensor(p_list), torch.tensor(r_list), torch.tensor(f1_list)


# 実行部分
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         pr_conf_yolo=0,
         pr_conf_yolo_ecm=0):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if trace:
            model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

# =================================================    
# # VIT
    resnet = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=3)
    resnet = resnet.to(device)
    try:
        path = "./ecm/model_path/ViT_GPU20ep.pth"
        params = torch.load(path)
        resnet.load_state_dict(params)
    except:
        path = "./ecm/model_path/ViT_CPU20ep.pth"
        params = torch.load(path)
        resnet.load_state_dict(params)
    resnet.eval()
    resnet.to(device)
# =================================================

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('potholes.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]


    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    coco91class = coco80_to_coco91_class()
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    pd_to_dict = dict()
    col_list = ['paths',
                'model_p', 'model_r', 'model_f1',
                'ecm_p', 'ecm_r', 'ecm_f1',]
    for col in col_list:
        pd_to_dict[col] = []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        # print((batch_i+1)*32)
        model_p, model_r, model_f1 = p_r_f1_calc(False, model, img, targets, paths, shapes, pr_th=None, conf_thres=0.05, iou_thres=0.65)
        ecm_p, ecm_r, ecm_f1 = p_r_f1_calc(True, model, img, targets, paths, shapes, pr_th=None, conf_thres=0.25, iou_thres=0.65)

        for bi in range(len(img)):
            pd_to_dict['paths'].append(paths[bi])
            pd_to_dict['model_p'].append(model_p[bi].item())
            pd_to_dict['model_r'].append(model_r[bi].item())
            pd_to_dict['model_f1'].append(model_f1[bi].item())

            pd_to_dict['ecm_p'].append(ecm_p[bi].item())
            pd_to_dict['ecm_r'].append(ecm_r[bi].item())
            pd_to_dict['ecm_f1'].append(ecm_f1[bi].item())

    df = pd.DataFrame(pd_to_dict)
    df.to_csv('./ecm/data/ecm_output_prf1.csv.csv')
    return 

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='potholes.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--pr_conf_yolo')
    parser.add_argument('--pr_conf_yolo_ecm')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('potholes.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric
             pr_conf_yolo=opt.pr_conf_yolo
             pr_conf_yolo_ecm=opt.pr_conf_yolo_ecm
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
