from torch.utils.data import DataLoader, random_split
import os
from torchvision import datasets, transforms, models
import torchvision
import pytorch_lightning as pl




class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 32, image_size: int = 224, numworkers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.train_data_dir = os.path.join(self.data_dir, "train")
        self.test_data_dir = os.path.join(self.data_dir, "test")
        self.batch_size = batch_size
        self.numworkers = numworkers
        self.train_transforms = torchvision.transforms.Compose(
            [
                # transforms.RandomResizedCrop(256,scale=(0.8, 1.0),ratio=(0.75, 1.33)),
                # transforms.RandomRotation(degrees=15),
                # transforms.CenterCrop(224),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]
        )

        self.test_transforms = torchvision.transforms.Compose(
            [
                # transforms.CenterCrop(224), 
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]
        )
        

    def prepare_data(self, validation_split=0.2):
        # Used to prepare the data for the run
        train_dataset = datasets.ImageFolder(root=self.train_data_dir, transform=self.train_transforms)
        self.test_dataset = datasets.ImageFolder(root=self.test_data_dir, transform=self.test_transforms)
        val_len = int(len(train_dataset) * validation_split)
        train_len = int(len(train_dataset) - val_len)
        self.train_dataset, self.val_dataset = random_split(train_dataset, [train_len, val_len])
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.numworkers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.numworkers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.numworkers,
            pin_memory=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.numworkers,
            pin_memory=False,
        )