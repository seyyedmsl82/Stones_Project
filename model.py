import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

batch_size = 64

device = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu")


# Custom Dataset
class StoneDataset(Dataset):
    def __init__(self, data_address, gp="train", transform=None, target_transform=None, resize=None):
        self.data = pd.read_csv(data_address)
        self.gp = gp
        self.resize = resize

        self.img_path = self.data.iloc[:, 0]
        self.label = self.data.iloc[:, -1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(f"data/{self.gp}/", self.img_path[index])
        label = self.label[index]
        # image = plt.imread(image_path)

        image = Image.open(image_path).convert('RGB')

        if self.resize is not None:
            image = self.resize(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


# Datasets
train_dataset = StoneDataset(r"data/train.csv", gp="train", transform=ToTensor(), resize=transforms.Resize((256, 256)))
test_dataset = StoneDataset(r"data/test.csv", gp="test", transform=ToTensor(), resize=transforms.Resize((256, 256)))

# DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
