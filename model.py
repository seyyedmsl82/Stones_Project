import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


class StoneDataset(Dataset):
    def __init__(self, data_address, gp="train"):
        self.data = pd.read_csv(data_address)
        self.gp = gp

        self.img_path = self.data.iloc[:, 0]
        self.label = self.data.iloc[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(f"data/{self.gp}/", self.img_path[index])
        label = self.label[index]
        return image_path, label


dataset = StoneDataset(r"data/test.csv", gp='test')
# print(dataset[0])
img = plt.imread(dataset[0][0])
plt.imshow(img)
plt.show()
