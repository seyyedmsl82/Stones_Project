# Import necessary libraries
import torch
from torch import nn
import torchvision.models as models


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(pretrained=True)

        # Freeze parameters to prevent backpropagation
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Additional convolutional layer
        self.additional_conv = nn.Conv2d(in_channels=1000, out_channels=512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

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

        x = x.unsqueeze(2).unsqueeze(3)  # Add dummy height and width dimensions
        x = self.additional_conv(x)
        x = self.relu(x)
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
