# YOLOv5
## 模型介绍
YOLOv5(You Only Look Once version 5)是一种目标检测模型,用于在视频或图像中识别和定位物体
## 模型结构
1.Backbone网络：YOLOv5使用CSPDarknet作为主干网络（backbone）。CSPDarknet是一种轻量级的神经网络结构，它基于Darknet53网络进行了改进，可以提供更好的特征提取能力。

2.特征金字塔网络（FPN）：在CSPDarknet之后，YOLOv5引入了一种特征金字塔网络来处理不同尺度的特征。FPN能够融合来自不同层级的特征，并生成多尺度的特征图，这有助于检测不同大小的目标。

3.检测头（Detection Head）：YOLOv5的检测头由几个卷积层和全连接层组成。检测头负责将特征图转换为目标框的坐标和类别概率预测。YOLOv5采用了一种预测方法，称为YOLOv3风格的预测。它将输入特征图划分为不同大小的网格，每个网格负责预测一组目标框。
## 数据集
图片来源是WIDER数据集，从中挑选出了32,203图片并进行了人脸标注，总共标注了393,703个人脸数据。并且对于每张人脸都附带有更加详细的信息，包扩blur（模糊程度）, expression（表情）, illumination（光照）, occlusion（遮挡）, pose（姿态）

## 训练及推理
### 环境配置
提供[光源](https://www.sourcefind.cn/#/service-details)拉取的训练以及推理的docker镜像：
* 训练镜像：
* 推理镜像：

python依赖安装：

    pip install -r requirement.txt
### 训练与Fine-tunning
训练命令：

    python train.py \
        --python3 train2yolo.py /path/to/original/widerface/train [/path/to/save/widerface/train]
        --python3 val2yolo.py  /path/to/original/widerface [/path/to/save/widerface/val]
        ...

Fine-tunning命令：

    python train.py 
       --img 640 
       --batch 16 
       --epochs 100 
       --data dataset.yaml 
       --cfg models/yolov5 
       --weights weights/yolov5m.pt 
       --name yolov5_face 
       --cache
### 预训练模型
model文件夹提供的预训练模型介绍，例如：

    Project
    ├── models
    │   ├── yolov5m.pth #pytorch预训练模型 
    │   ├── yolov5m-face.onnx #对应的onnx模型

### 测试
测试命令：

    python detect_face.py 
       --models 
       --source /xxx/dataset 
       --save-img

### 推理
推理引擎版本：
* ONNXRuntime(DCU版本) >= 1.14.0
* Migraphx(DCU版本) >= 2.5.0
#### ORT
基于ORT的推理命令：

    python ORT_infer.py \
        --args0 xxx \
        --args1 xxx \
        ...
#### Migraphx
基于Migraphx的推理命令：

    python Migraphx_infer.py \
        --args0 xxx \
        --args1 xxx \
        ...

## 性能和准确率数据
测试数据：https://share.weiyun.com/5vSUomP

根据模型情况填写表格：

| 任务                           | 准确率    |
| ----------------------------- | --------- |
| GLUE任务集平均准确率            | 80.4%     |
| IMDb电影评论情感分类任务准确率   | 97.1%     |
| SST-2电影评论情感分类任务准确率 | 93.5%     |
| MNLI任务准确率                  | 84.3%     |
| SNLI任务准确率                  | 90.0%     |
| QQP问题配对任务准确率           | 91.3%     |
| SQuAD 1.1阅读理解任务准确率     | 90.8%     |
| SQuAD 2.0阅读理解任务准确率     | 83.1%     |


## 源码仓库及问题反馈
* https://github.com/cc0212/yolov5-face-dxq
## 参考
* https://blog.csdn.net/m0_63879480/article/details/131029197?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-131029197-blog-120665458.235%5Ev38%5Epc_relevant_sort_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-131029197-blog-120665458.235%5Ev38%5Epc_relevant_sort_base3&utm_relevant_index=6

