# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 08:47:35 2022

@author: kilogrand
"""
import sys
import argparse
import os
from numpy import random
import time
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.backends.cudnn as cudnn
# 添加文件夹到环境变量，解决导入py文件问题
sys.path.extend(["./yolov5", "./yolov5/models", "./yolov5/utils"])

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from yolov5.utils.torch_utils import time_synchronized
from LPRNet.model.LPRNet import LPRNet
from LPRNet.data.load_data import CHARS
import warnings
warnings.filterwarnings("ignore")

def change_cv2_draw(image,strs,local,sizes,colour):
    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("./simsun.ttc", 32, encoding="unic")
    draw.text(local, strs, 'white', font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        image = change_cv2_draw(img,label,(int(c1[0]), int(c1[1]) - 30),5,[225,225,225])
    return image

def transform( img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img

def apply_classifier(x, model, img, im0, CHARS):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    plat_num = 0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (94, 24))  # BGR
                im = transform(im)
                ims.append(im)
            preds = model(torch.Tensor(ims).to(d.device))  # classifier prediction
            prebs = preds.cpu().detach().numpy()
            preb_labels = list()
            for w in range(prebs.shape[0]):
                preb = prebs[w, :, :]
                preb_label = list()
                for j in range(preb.shape[1]):
                    preb_label.append(np.argmax(preb[:, j], axis=0))
                no_repeat_blank_label = list()
                pre_c = preb_label[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label:  # dropout repeate label and blank label
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                preb_labels.append(no_repeat_blank_label)
            plat_num = np.array(preb_labels)
    return x, plat_num


def detect(opt):
    out, source, view_img, save_txt = \
        opt.output, opt.source, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    imgsz = 640
    yolov5_weights = './yolov5/weights/last.pt'
    LPRNet_weights = './LPRNet/weights/Final_LPRNet_model.pth'
    device = torch.device('cpu')  # cpu版本，只使用cpu
    half = False
    save_img = True

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Load yolov5s model
    model = torch.load(yolov5_weights, map_location=device)["model"].float().eval()  # load FP32 model
    # model = attempt_load(yolov5_weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # load Second-stage classifier LPRNet
    modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
    modelc.load_state_dict(torch.load(LPRNet_weights, map_location=device))
    modelc.to(device).eval()
    print("Load pretrained model successful!")

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        print(pred.shape)
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        # Apply Classifier
        pred, plat_num = apply_classifier(pred, modelc, img, im0s, CHARS)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 统计每个类别的个数
                    s += 'class %s %g个, ' % (names[int(c)-1], n)  # add to string

                # Write results
                for de, lic_plat in zip(det, plat_num):
                    *xyxy, conf, cls = de
                    lb = ""
                    for _, i in enumerate(lic_plat):
                        lb += CHARS[int(i)]
                    label = '%s %.2f' % (lb, conf)
                    print(label)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(label+'\n')  # label format

                    if save_img or view_img:  # Add bbox to image
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)-1], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % out)
        if sys.platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./inference/images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./inference/output', help='output folder')  # output folder
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt)
