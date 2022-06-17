#load model
from pytorch_lightning import Trainer, seed_everything
from model import MyModel
from dataset import MyDataModule
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore")


#修改这三个参数
model_name = 'resnet18'
saved_model_path = '/root/zhshen/image_calssfication/None/resnet18/checkpoints/epoch=9-step=799-v1.ckpt'
data_dir = "/root/zhshen/image_calssfication"



model = MyModel.load_from_checkpoint(saved_model_path,lr=0.001,classes=8, model_name=model_name)
data = MyDataModule(data_dir=data_dir, batch_size=32)



train_loss = np.load('./train_loss.npy')
train_acc = np.load('./train_acc.npy')
val_loss = np.load('./val_loss.npy')
val_acc = np.load('./val_acc.npy')
print(train_loss.shape)


def plot_loss(train_loss, val_loss):
    plt.figure()
    X = np.arange(len(train_loss))
    plt.plot(X, train_loss, 'k-+', label="train")
    plt.plot(X, val_loss[1:], 'k-.', label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("train loss and val loss")
    plt.legend()
    plt.savefig('loss.png')

def plot_acc(train_acc, val_acc):
    plt.figure()
    X = np.arange(len(train_acc))
    plt.plot(X, train_acc, 'k-+', label="train")
    plt.plot(X, val_acc[1:], 'k-.', label="val")
    plt.xlabel("step")
    plt.ylabel("acc")
    plt.title("train acc and val acc")
    plt.legend()
    plt.savefig('acc.png')


plot_loss(train_loss, val_loss)
print('训练集与验证集损失函数已保存为loss.png')
plot_acc(train_acc, val_acc)
print('训练集与验证集准确率函数已保存为acc.png')

t_labels = []
t_preds = []
data.prepare_data()
for batch in data.test_dataloader():
    images, labels = batch
    preds = model(images)
    t_labels.append(labels)
    t_preds.append(preds)


import torch
t_labels = torch.cat(t_labels, dim=0)
t_preds = torch.cat(t_preds, dim=0)
t_labels = t_labels.detach().cpu().numpy()
t_preds = t_preds.detach().cpu().numpy()


t_preds = np.argmax(t_preds, axis=1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(t_labels, t_preds)
print("accuracy: {}".format(acc))
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(t_labels, t_preds)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('confusion_matrix.png')


cnf_matrix = confusion_matrix(t_labels, t_preds)

# Plot normalized confusion matrix
fig = plt.figure()
fig.set_size_inches(14, 12, forward=True)
#fig.align_labels()
print(data.test_dataset.class_to_idx)
label_names = data.test_dataset.classes
# fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
plot_confusion_matrix(cnf_matrix, classes=np.asarray(label_names), normalize=True,
                      title='Normalized confusion matrix')
print('confusion_matrix done')




