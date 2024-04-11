import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image as Image
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader, Dataset

batch_size = 50
num_epochs = 10

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


transform = transforms.Compose([
    Resize((512, 512)),
    ToTensor()
])

# Datasets
train_dataset = StoneDataset(r"data/train.csv", gp="train", transform=transform)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.33, random_state=42)
test_dataset = StoneDataset(r"data/test.csv", gp="test", transform=transform)

# DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
        self.fc1 = nn.Linear(1000, 256)
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


model = Net()
model = model.to(device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop with tqdm progress bar
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model.train()
    train_dataloader_with_progress = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")

    for i, (images, labels) in enumerate(train_dataloader_with_progress):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        corrects = (torch.max(logits, 1)[1] == labels).sum().item()
        accuracy = 100.0 * corrects / len(labels)
        train_acc += accuracy
        train_dataloader_with_progress.set_postfix(loss=train_running_loss / (i + 1), accuracy=train_acc / (i + 1))

    # Compute average loss and accuracy for the epoch
    epoch_loss = train_running_loss / len(train_dataloader)
    epoch_acc = train_acc / len(train_dataloader)

    # Validation step
    val_running_loss = 0.0
    val_acc = 0.0

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            val_loss = criterion(logits, labels)
            val_running_loss += val_loss.item()

            corrects = (torch.max(logits, 1)[1] == labels).sum().item()
            accuracy = 100.0 * corrects / len(labels)
            val_acc += accuracy

    val_epoch_loss = val_running_loss / len(val_dataloader)
    val_epoch_acc = val_acc / len(val_dataloader)

    print(f'Epoch: {epoch + 1}/{num_epochs}, '
          f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}, '
          f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}')
