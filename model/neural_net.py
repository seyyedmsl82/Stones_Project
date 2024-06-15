"""
This script defines a neural network model for image classification using the PyTorch framework.
The main functionalities and components include:

1. **Import Libraries**:
    Import necessary libraries such as PyTorch, torchvision, and torch.nn modules.

2. **Net Class**:
   - **Description**:
        The `Net` class defines a neural network model using a pre-trained ResNet-18 backbone followed by additional
        fully connected layers for feature extraction and classification.

   - **Model Architecture**:
        1. **Base Model (Pre-trained ResNet-18)**:
            - The `base_model` attribute represents the ResNet-18 backbone pre-trained on ImageNet.
            - Parameters of the base model are frozen (`requires_grad = False`) to prevent backpropagation.

        2. **Additional Layers**:
            - Dropout layers with probabilities of 0.3 and 0.4.
            - Fully connected layers with 1000, 256, 128 input/output features.
            - Batch normalization layers for 256 and 128 features.
            - The final fully connected layer outputs 5 classes for classification.

3. **Forward Pass**:
    The `forward` method defines the forward pass through the network, passing input tensors through the base
    model and additional layers to produce class scores.

4. **Layer Unfreezing**:
    The `unfreeze_layer` method allows specific layers in the base model to be unfrozen for fine-tuning during
    training.


Author: SeyyedReza Moslemi
Date: Jun 15, 2024
"""


# Import necessary libraries
import torch
from torch import nn
import torchvision.models as models


# Define the neural network architecture
class Net(nn.Module):
    """
    Description:
        This class defines a neural network model called Net using the PyTorch framework. The model
        architecture consists of a pre-trained ResNet-18 backbone followed by additional layers for
        feature extraction and classification. The ResNet-18 backbone is initialized with pre-trained
        ImageNet weights, and its parameters are frozen to prevent backpropagation during training.
        The additional layers include fully connected layers with ReLU activation, dropout, and batch normalization.

    Model Architecture:
        1. Base Model (Pre-trained ResNet-18):
            * The base_model attribute represents the ResNet-18 backbone pre-trained on ImageNet.
            * Parameters of the base model are frozen (requires_grad = False) to prevent backpropagation.

        2. Additional Layers:
            * dropout1: Dropout layer with a dropout probability of 0.3.
            * fc1: Fully connected layer with 1000 input features (ResNet output) and 256 output features.
            * dropout2: Dropout layer with a dropout probability of 0.4.
            * batch_norm1: Batch normalization layer with 256 features.
            * fc2: Fully connected layer with 256 input features and 128 output features.
            * dropout3: Dropout layer with a dropout probability of 0.4.
            * batch_norm2: Batch normalization layer with 128 features.
            * fc3: Fully connected layer with 128 input features and 5 output features (for classification).

    Forward Pass:
        The forward method defines the forward pass of the neural network. It takes an input tensor x
        and passes it through the layers defined in the model architecture:
            * The input tensor is passed through the base model (ResNet-18) to extract features.
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
        # Load pre-trained ResNet-18 model
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
        """
        Description:
            Perform a forward pass through the network.

        Arguments:
            x (torch.Tensor): Input tensor containing the images.

        Returns:
            torch.Tensor: Output tensor containing the class scores for each image.
        """
        # Forward pass through base model
        x = self.base_model(x)
        x = self.dropout1(x)

        # Flatten the tensor
        x = torch.flatten(x, 1)

        # Forward pass through additional layers
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
        Description:
            Unfreeze a specific layer in the base model.

        Arguments:
            layer_name (str): Name of the layer to unfreeze.

        Returns:
            None
        """
        for name, param in self.base_model.named_parameters():
            if name == layer_name:
                param.requires_grad = True
                break
