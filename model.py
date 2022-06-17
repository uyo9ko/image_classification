import os
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import pytorch_lightning as pl



def get_models_last(model):
    last_name = list(model._modules.keys())[-1]
    last_module = model._modules[last_name]
    return last_module, last_name

class CustomClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
           self.model = torchvision.models.__dict__[arch](pretrained = pretrained)
        else:
           self.model = torchvision.models.__dict__[arch]()

        #这两句话，required_graa是控制是否冻结的，如果注释掉就不冻结
        for param in self.model.parameters():
            param.requires_grad = False

            
        last_module, last_name = get_models_last(self.model)
        if isinstance(last_module, nn.Linear):
            n_features = last_module.in_features
            self.model._modules[last_name] = nn.Sequential(nn.Linear(n_features, 256),
                                                nn.ReLU(),
                                                nn.Dropout(0.4),
                                                nn.Linear(256, 64),
                                                nn.ReLU(),
                                                nn.Dropout(0.3),
                                                nn.Linear(64, num_classes))
            self.model._modules[last_name].requires_grad = True
        elif isinstance(last_module, nn.Sequential):
            seq_last_module, seq_last_name = get_models_last(last_module)
            n_features = seq_last_module.in_features
            last_module._modules[seq_last_name] = nn.Sequential(nn.Linear(n_features, 256),
                                                nn.ReLU(),
                                                nn.Dropout(0.4),
                                                nn.Linear(256, 64),
                                                nn.ReLU(),
                                                nn.Dropout(0.3),
                                                nn.Linear(64, num_classes))
            last_module._modules[seq_last_name].requires_grad = True

        # just for test
        # self.last = list(self.model.named_modules())[-1][1]


    def forward(self, input_neurons):
        # TODO: add dropout layers, or the likes.
        output_predictions = self.model(input_neurons)
        return output_predictions



class MyModel(pl.LightningModule):

    def __init__(self,lr=0.001,classes=8, model_name='resnet18'):
        super(MyModel, self).__init__()
        self.lr = lr

        self.model_name = model_name
        self.num_classes = classes
        self.feature_extract = True
        self.use_pretrained = True

        self.models = CustomClassifier(self.model_name, self.num_classes, self.use_pretrained)
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.tmp_train_loss = []
        self.tmp_train_acc = []


    def forward(self, x):
        output = self.models(x)
        return output

    def training_step(self, batch, batch_idx):
        images, target = batch
        preds = self.forward(images)
        loss = F.cross_entropy(preds, target)
        acc = accuracy(preds, target)
        # tensorboard_logs = {'train_loss': loss}
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.tmp_train_loss.append(loss)
        self.tmp_train_acc.append(acc)
        # self.train_loss.append(loss.cpu().detach().numpy()) 
        # self.train_acc.append(acc.cpu().detach().numpy())
        return {'loss': loss, 'acc': acc}


    
    def validation_step(self, batch, batch_idx):
        images, target = batch
        preds = self.forward(images)
        loss = F.cross_entropy(preds, target)
        acc = accuracy(preds, target)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.val_loss.append(avg_loss.cpu().detach().numpy())
        self.val_acc.append(avg_acc.cpu().detach().numpy())
        if len(self.tmp_train_loss) > 0:
            avg_train_loss = torch.stack(self.tmp_train_loss).mean()
            avg_train_acc = torch.stack(self.tmp_train_acc).mean()
            self.train_loss.append(avg_train_loss.cpu().detach().numpy())
            self.train_acc.append(avg_train_acc.cpu().detach().numpy())
            self.tmp_train_acc = []
            self.tmp_train_loss = []



    def predict_step(self, batch, batch_idx):
        images, target = batch
        preds = self.forward(images)
        return preds
        
    # def test_step(self, batch, batch_idx):
    #     images, target = batch
    #     preds = self.forward(images)
    #     return {'test_loss': F.cross_entropy(preds, target)}

    # def test_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'test_loss': avg_loss}
    #     return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.models.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]



