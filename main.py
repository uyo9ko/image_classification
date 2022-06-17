
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from model import MyModel
from dataset import MyDataModule
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
seed_everything(7)

# NUM_WORKERS = int(os.cpu_count() / 2)

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


#修改这两个值
model_name = 'resnet18'
data_dir = "/root/zhshen/image_calssfication/"


wandb_logger = WandbLogger(project='image_classification', version =  model_name, log_model=True )
model = MyModel(lr=0.001,classes=8, model_name=model_name)
data = MyDataModule(data_dir=data_dir, batch_size=32)
trainer = Trainer(gpus=1, max_epochs=5, logger=wandb_logger, log_every_n_steps=5, val_check_interval=0.1)
trainer.fit(model,data) 

train_loss =  np.array(model.train_loss)
train_acc = np.array(model.train_acc)
val_loss = np.array(model.val_loss)
val_acc = np.array(model.val_acc)

np.save('train_loss.npy', train_loss)
np.save('train_acc.npy', train_acc)
np.save('val_loss.npy', val_loss)
np.save('val_acc.npy', val_acc)



