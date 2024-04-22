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
    """
    Description:
        This class implements a custom PyTorch dataset, StoneDataset, for loading and preprocessing
        stone image data from a CSV file. It reads the CSV file containing image paths and corresponding
        labels, and loads the images on-demand during training or evaluation. It also performs label
        encoding using LabelEncoder from scikit-learn library.


    Arguments:
        :param data_address: Path to the CSV file containing image paths and labels.
        :param gp: Group parameter to specify the group or category of the dataset.
        :param transform: Optional image transformations to be applied to the input images (e.g., resizing).
        :param target_transform: Optional transformations to be applied to the target labels (e.g., one-hot encoding).


    Methods:
        __init__(self, data_address, gp, transform=None, target_transform=None):
            Constructor method that initializes the StoneDataset class. It reads the CSV file, stores the
            image paths and labels, and initializes the transformation functions and label encoder.
        __len__(self):
            Returns the total number of samples in the dataset.
        __getitem__(self, index):
            Retrieves and preprocesses a single sample from the dataset. It loads the image from the
            specified path, applies transformations, and returns the image and corresponding label.


    Example Usage:
        ```python

            # Instantiate StoneDataset
            stone_dataset = StoneDataset(data_address='data.csv', gp='group1', transform=transforms.ToTensor())

            # Create a DataLoader for batching and shuffling the dataset
            stone_dataloader = DataLoader(stone_dataset, batch_size=32, shuffle=True)

            # Iterate over batches of data
            for images, labels in stone_dataloader:
                # Perform training or evaluation steps
                pass

        ```
    """

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
