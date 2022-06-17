# image_classification
apply diffierent pretrained models to image_classification


## 安装要求

1. 安装pytorch torchvision 可以在官网根据自己的环境选择安装指令 https://pytorch.org/get-started/locally/
2. 安装pytroch lightning， pytorch的精简版，类似于tensorflow里的keras
4. 安装wandb，监视模型，观察loss曲线等。wandb使用前需要在wandb官网注册账号，在本地环境初始化。

```
pip install pytorch-lightning
pip install wandb
```


## 运行
支持的模型如下
```
supported_model = [  'alexnet', 'AlexNet', 'resnet18',
                     'resnet34', 'resnet50', 'resnet101', 'resnet152',
                     'resnext50_32x4d', 'resnext101_32x8d',
                     'wide_resnet50_2', 'wide_resnet101_2',
                     'vgg11', 'vgg11_bn','vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                     'vgg19_bn', 'vgg19', 'Inception3', 'inception_v3',
                     'DenseNet', 'densenet121', 'densenet169',
                     'densenet201', 'densenet161', 'googlenet', 'GoogLeNet',
                     'MobileNetV2', 'mobilenet_v2','mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0',
                     'mnasnet1_3','shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
                     'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
```
1. 准备数据 数据应文件目录应如下
```
image_classification
├── train
│   ├── CIJam
│   ├── DisMFTJam
│   └── ...
└── test
    ├── CIJam
    ├── DisMFTJam
    └── ...
```
2. 打开main.py修改数据路径参数data_dir,对应的模型名字model_name等。
3. 运行main.py 等待训练完成, 模型默认保存在'./None/model_name/checkpoints'路径下
4. 打开wandb，登陆账号，观察训练验证loss曲线等
5. 打开evaluate.py，修改本地保存ckpt模型的路径参数，运行代码得到混淆矩阵精确率召回率等数据指标

