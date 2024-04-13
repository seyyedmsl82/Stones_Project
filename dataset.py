# Import necessary libraries
import os
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing


# Custom Dataset class for handling data
class StoneDataset(Dataset):
    def __init__(self, data_address, gp, transform=None, target_transform=None):
        self.data = pd.read_csv(data_address)
        self.gp = gp

        self.img_path = self.data.iloc[:, 0]
        self.label = self.data.iloc[:, -1]
        self.transform = transform
        self.target_transform = target_transform

        # Initialize LabelEncoder
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(f"data/{self.gp}/", self.img_path[index])
        label = self.label[index]
        # image = plt.imread(image_path)
        # Encode the label
        label = self.label_encoder.transform([label])[0]
        label = torch.as_tensor(np.uint8(label))

        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
