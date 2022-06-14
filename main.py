
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


wandb_logger = WandbLogger(project='image_classification', log_model=True )
model = MyModel(lr=0.001,classes=8, model_name='googlenet')
data = MyDataModule(data_dir="/root/zhshen/image_calssfication/", batch_size=32)
# most basic trainer, uses good defaults
#trainer = Trainer(num_tpu_cores=8, precision=16,max_epochs=2)
trainer = Trainer(gpus=1, max_epochs=30, logger=wandb_logger, log_every_n_steps=5)
#trainer = Trainer()
#trainer = Trainer(num_tpu_cores=8, max_epochs=2, precision=16)
trainer.fit(model,data) 

