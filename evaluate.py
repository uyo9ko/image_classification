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


model = MyModel.load_from_checkpoint('./None/version_None/checkpoints/epoch=29-step=2399-v1.ckpt',lr=0.001,classes=8, model_name='googlenet')
data = MyDataModule(data_dir="/root/zhshen/image_calssfication", batch_size=32)
label_names = os.listdir('./test/')
train_loss = np.load('train_loss.npy')
train_acc = np.load('train_acc.npy')
val_loss = np.load('val_loss.npy')
val_acc = np.load('val_acc.npy')

import numpy as np
def plot_curve(data, name):
    plt.figure(figsize=(10,8))
    plt.plot(data, label=name)
    plt.legend()
    plt.title(name+' curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(name+'.png')

plot_curve(train_loss, 'train_loss')
print('train_loss done')
plot_curve(train_acc, 'train_acc')
print('train_acc done')
plot_curve(val_acc, 'val_acc')
print('val_acc done')
plot_curve(val_loss, 'val_loss')
print('val_loss done')



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

# fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
plot_confusion_matrix(cnf_matrix, classes=np.asarray(label_names), normalize=True,
                      title='Normalized confusion matrix')
print('confusion_matrix done')




