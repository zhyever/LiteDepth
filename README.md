# LiteDepth

> **LiteDepth: Digging into Fast and Accurate Depth Estimation on Mobile Devices**
>
> Zhenyu Li, Zehui Chen, Jialei Xu, Xianming Liu, Junjun Jiang
>
> [ECCVW 2022 (arXiv pdf)](https://arxiv.org/abs/2209.00961)

## Notice
- Redundancy version of LiteDepth. Main codes are in projects/.
- I'd like to complete the docs ASAP.

## Install
This project is based on the following packages:
- python 3.7
- cuda 11.1
- cudnn 8.0.5_0
- pytorch 1.8.0

Before running, you should also install some packages facilitating training (Refer to the repos for installation details):
- mmclassification
- monocular-depth-estimation-toolbox
- pytorch-image-models
- robust_loss_pytorch

As for converting Pytorch to tfLite, you need to install:
- onnx 1.11.0
- onnx-simplifier 0.3.10
- onnx-tf 1.9.0
- tensorflow 2.5.0

That can be sort of tricky to handle the environment. I will double-check the environment to ensure its correctness.


## Configs
- Basemodel config: projects/configs/configs_baseline/basemodel_crop_gradloss_vnl_robust.py
- Teacher config: projects/configs/configs_distll/swinl_w7_teacher.py
- Distill config: projects/configs/configs_distll/swinl_w7_similarity.py