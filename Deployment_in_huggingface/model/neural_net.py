# Import necessary libraries
import torch
from torch import nn
import torchvision.models as models


# Define the neural network architecture
class Net(nn.Module):
    """
    Description:
        This class defines a neural network model called Net using the PyTorch framework. The model
        architecture consists of a pre-trained ResNet-50 backbone followed by additional layers for
        feature extraction and classification. The ResNet-50 backbone is initialized with pre-trained
        ImageNet weights, and its parameters are frozen to prevent backpropagation during training.
        The additional layers include a convolutional layer, followed by fully connected layers with
        ReLU activation, dropout, and batch normalization.


    Model Architecture:
        1. Base Model (Pre-trained ResNet-50):
            * The base_model attribute represents the ResNet-50 backbone pre-trained on ImageNet.
            * Parameters of the base model are frozen (requires_grad = False) to prevent backpropagation.

        2. Additional Convolutional Layer:
            * The additional_conv attribute represents a convolutional layer with 1000 input channels
              (from ResNet output) and 512 output channels.
            * ReLU activation function (relu) is applied after the convolutional layer.

        3. Additional Layers:
            * dropout1: Dropout layer with a dropout probability of 0.3.
            * fc1: Fully connected layer with 512 input features and 256 output features.
            * dropout2: Dropout layer with a dropout probability of 0.4.
            * batch_norm1: Batch normalization layer with 256 features.
            * fc2: Fully connected layer with 256 input features and 128 output features.
            * dropout3: Dropout layer with a dropout probability of 0.4.
            * batch_norm2: Batch normalization layer with 128 features.
            * fc3: Fully connected layer with 128 input features and 5 output features (for classification).


    Forward Pass:
        The forward method defines the forward pass of the neural network. It takes an input tensor x
        and passes it through the layers defined in the model architecture:

            * The input tensor is passed through the base model (ResNet-50) to extract features.
            * Additional convolutional layer and activation functions (ReLU) are applied to further
              process the features.
            * Dropout layers are applied to prevent overfitting during training.
            * The feature tensor is flattened and passed through fully connected layers with ReLU
              activation and batch normalization.
            * The output tensor represents the predicted class probabilities for the input samples.


    Example Usage:
        ```python

            # Instantiate the model
            model_ = Net()

            # Forward pass with input tensor x
            output = model_(x)

        ```
    """
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)

        # Freeze parameters to prevent backpropagation
        for param in self.base_model.parameters():
            param.requires_grad = False

        # # Additional convolutional layer
        # self.additional_conv = nn.Conv2d(in_channels=1000, out_channels=600, kernel_size=3, padding=1)
        # self.relu = nn.ReLU(inplace=True)

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

        # x = x.unsqueeze(2).unsqueeze(3)  # Add dummy height and width dimensions
        # x = self.additional_conv(x)
        # x = self.relu(x)
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

    def unfreeze_layer(self, layer_name):
        """
        Unfreeze a specific layer in the base model.

        Args:
            layer_name (str): Name of the layer to unfreeze.
        """
        for name, param in self.base_model.named_parameters():
            if name == layer_name:
                param.requires_grad = True
                break
