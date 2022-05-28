# LPRNet_Pytorch
Pytorch Implementation For LPRNet, A High Performance And Lightweight License Plate Recognition Framework.  
完全适用于中国车牌识别（Chinese License Plate Recognition）及国外车牌识别！  
目前仅支持同时识别蓝牌和绿牌即新能源车牌等中国车牌，但可通过扩展训练数据或微调支持其他类型车牌及提高识别准确率！

# dependencies

- pytorch >= 1.0.0
- opencv-python 3.x
- python 3.x
- imutils
- Pillow
- numpy

# pretrained model

* [pretrained_model](./weights/)

# training and testing

1. prepare your datasets, image size must be 94x24.
2. base on your datsets path modify the scripts its hyperparameters --train_img_dirs or --test_img_dirs.
3. adjust other hyperparameters if need.
4. run 'python train_LPRNet.py' or 'python test_LPRNet.py'.
5. if want to show testing result, add '--show true' or '--show 1' to run command.

# performance

- personal test datasets.
- include blue/green license plate.
- images are very widely.
- total test images number is 27320.

|  size  | personal test imgs(%) | inference@gtx 1060(ms) |
| ------ | --------------------- | ---------------------- |
|  1.7M  |         96.0+         |          0.5-          |

# References

1. [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
2. [PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)

# 车牌图片生成
训练集是使用中国车牌生成器生成的车牌，git仓库：<https://gitee.com/leijd/chinese_license_plate_generator>，`generate_multi_plate.py`代码里面两个错误：
- 第91行：解决无法读取以中文命名的文件
```python
# font_img = cv2.imread(font_filename, cv2.IMREAD_GRAYSCALE)
            # OpenCV cv2.imread()函数无法读取以中文命名的图像文件，解决方案：
            font_img = cv2.imdecode(np.fromfile(font_filename, dtype=np.uint8), 0)
```

- 第298行：解决无法保存问题
```python
# cv2.imwrite(os.path.join(args.save_adr, '{}_{}_{}.jpg'.format(gt_plate_number, bg_color, is_double)), img)
        # OpenCV cv2.imwrite()函数写入以中文命名的图像文件，解决方案：
        cv2.imencode('.jpg', img)[1].tofile(os.path.join(args.save_adr, '{}_{}_{}.jpg'.format(gt_plate_number, bg_color, is_double)))
```


`generate_special_plate.py`代码里面一个错误：
- 最后一行
```python
# cv2.imwrite('{}.jpg'.format(args.plate_number), img)
cv2.imencode('.jpg', img)[1].tofile(os.path.join('./', '{}_{}_{}.jpg'.format(args.plate_number, args.bg_color, args.double)))
```

`chinese_license_plate_generator是已经修复好bug的版本`

随机生成10000张车牌图片，保存到`../LPRNet/data/train`：

```bash
python generate_multi_plate.py --number 10000 --save-adr ../LPRNet/train_data
```

生成特定的车牌
```bash
python generate_special_plate.py --plate-number 湘999997 --bg-color yellow
```
