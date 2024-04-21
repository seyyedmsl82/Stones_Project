# Import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch import nn


def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, criterion, epochs, device, scheduler):
    """
    Description:
    This function trains a given PyTorch model using the provided training and validation dataloaders,
    for a specified number of epochs. It also evaluates the model on a test dataloader after training.
    It logs the training and validation loss and accuracy for each epoch in a text file named "mylog.txt".

    Arguments:
    :param model: The PyTorch model to be trained.
    :param train_dataloader: Dataloader for the training dataset.
    :param val_dataloader: Dataloader for the validation dataset.
    :param test_dataloader: Dataloader for the test dataset.
    :param optimizer: Optimizer used for training the model.
    :param criterion: Loss function used for calculating the loss.
    :param epochs: Number of epochs for training.
    :param device: Device to be used for training ('cpu' or 'cuda').
    :param scheduler: Learning rate scheduler.

    Outputs:
    :return: Returns the trained PyTorch model.
    """

    # Training loop with tqdm progress bar
    for epoch in range(epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model.train()  # Set the model to training mode
        train_dataloader_with_progress = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}")

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
            train_dataloader_with_progress.set_postfix(loss=train_running_loss / (i + 1),
                                                       accuracy=train_acc / (i + 1))

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

        training_log = f'''Epoch: {epoch + 1}/{epochs},
                    Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f},
                    Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f} \n'''
        with open("mylog.txt", "a") as f:
            f.write(training_log)

        # Update the learning rate
        scheduler.step(val_epoch_loss)

    for i, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        corrects = (torch.max(logits, 1)[1] == labels).sum().item()
        test_accuracy = 100.0 * corrects / len(labels)

        with open("mylog.txt", "a") as f:
            f.write(f'Test Accuracy: {test_accuracy:.2f}')

        print(test_accuracy)

    return model


def feature_maps(model, image_path, transform=None, device="cpu"):
    """
    Description:
    This function visualizes the feature maps of a given image using a pre-trained model.
    It takes an image path, applies transformations if provided, and extracts feature maps
    from a specified layer of the model. It then visualizes the feature maps using matplotlib
    and saves the visualization as "feature_maps.png".

    Arguments:
    :param model: Pre-trained PyTorch model.
    :param image_path: Path to the input image.
    :param transform: Optional image transformations.
    :param device: Device to be used ('cpu' or 'cuda').

    Outputs:
    Displays and saves the visualization of feature maps.
    """

    image = Image.open(image_path)
    image = transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)
    # output = model(image)

    # Choose the layer whose feature maps you want to visualize
    target_layer = model.base_model.conv1

    # Get the activations (feature maps) from the target layer
    activations = target_layer(image).cpu()

    # Convert activations to numpy array
    activations = activations.detach().numpy()

    # Visualize the feature maps
    num_feature_maps = activations.shape[1]  # Number of feature maps
    plt.figure(figsize=(24, 24))
    for i in range(num_feature_maps):
        plt.subplot(9, 8, i+1)  # Adjust the subplot layout based on the number of feature maps
        plt.imshow(activations[0, i, :, :], cmap='gray')
        plt.axis('off')

    plt.savefig("feature_maps.png")
    plt.show()


def grad_cam(model, image_path, image_size, transform=None, device="cpu"):
    """
    Description:
    This function computes the Gradient-weighted Class Activation Mapping (Grad-CAM) for a
    given input image using a pre-trained model. It computes the class activation map (CAM)
    and visualizes it overlaid on the input image. It returns the CAM and the input image as
    NumPy arrays.

    Arguments:
    :param model: Pre-trained PyTorch model.
    :param image_path: Path to the input image.
    :param image_size: Size of the input image.
    :param transform: Optional image transformations
    :param device: Device to be used ('cpu', 'cuda')

    Outputs:
    cam: Gradient-weighted Class Activation Map (CAM) as a NumPy array.
    img: Input image as a NumPy array.
    """

    # set the model to evaluation mode
    model.eval()
    img_ = Image.open(image_path)
    img = transform(img_).unsqueeze(0)
    img = img.to(device)

    logits = model(img)
    pred = torch.argmax(logits, dim=1)

    # Compute the gradients of the predicted class output with respect to the activations of the target layer
    model.zero_grad()
    logits[:, pred].backward()

    # Get the activations of the additional_conv layer
    gradients = model.additional_conv.weight.grad  # Gradients of additional_conv with respect to the output

    # Compute the gradient-weighted class activation map (CAM)
    cam = torch.mean(gradients, dim=(2, 3))  # Global average pooling along spatial dimensions
    cam = nn.functional.relu(cam)
    cam = cam.detach().cpu().numpy()[0]  # Convert to numpy array

    # # Normalize the CAM
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)

    # Resize the CAM to match the input image size
    cam = np.uint8(255 * cam)
    cam = np.uint8(Image.fromarray(cam).resize(image_size, Image.Resampling.LANCZOS))

    # Convert the input image to a numpy array
    input_image_np = img.squeeze().cpu().numpy().transpose(1, 2, 0)

    return cam, input_image_np
