import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

batch_size = 64
num_epochs = 100

device = ("cuda"
          if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu")


# Custom Dataset
class StoneDataset(Dataset):
    def __init__(self, data_address, gp="train", transform=None, target_transform=None):
        self.data = pd.read_csv(data_address)
        self.gp = gp

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

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Datasets
# train_dataset = StoneDataset(r"data/train.csv", gp="train", transform=ToTensor(), resize=transforms.Resize((256, 256)))
# test_dataset = StoneDataset(r"data/test.csv", gp="test", transform=ToTensor(), resize=transforms.Resize((256, 256)))

train_dataset = StoneDataset(r"data/train.csv", gp="train", transform=transform)
test_dataset = StoneDataset(r"data/test.csv", gp="test", transform=transform)

# DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Additional layers
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.4)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.batch_norm1(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout3(x)
        x = self.batch_norm2(x)
        x = self.fc3(x)
        return x
