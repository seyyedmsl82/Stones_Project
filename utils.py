import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from pytorch_grad_cam import GradCAM


def train(model, train_dataloader, val_dataloader, test_dataloader, optimizer, criterion, epochs, device, scheduler):
    """
    Description:
        Trains a given PyTorch model using the provided training, validation, and test dataloaders.

    Arguments:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation dataset.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for the test dataset.
        optimizer (torch.optim.Optimizer): Optimizer used for training the model.
        criterion (torch.nn.Module): Loss function used for calculating the loss.
        epochs (int): Number of epochs for training.
        device (str): Device to be used for training ('cpu' or 'cuda').
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.

    Returns:
        torch.nn.Module: The trained PyTorch model.
    """

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop with tqdm progress bar
    for epoch in range(epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model.train()  # Set the model to training mode
        train_dataloader_with_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

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
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

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
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        training_log = f'''Epoch: {epoch + 1}/{epochs},
                    Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f},
                    Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f} \n'''
        with open("training_log.txt", "a") as f:
            f.write(training_log)

        # Update the learning rate
        scheduler.step(val_epoch_loss)

    # Plotting accuracy and loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Testing the model
    model.eval()
    test_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * test_correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    with open("training_log.txt", "a") as f:
        f.write(f'Test Accuracy: {test_accuracy:.2f}')

    return model


def feature_maps(model, image_path, transform=None, device="cpu"):
    """
    Description:
        Visualizes the feature maps of a given image using a pre-trained model.


    Arguments:
        model (torch.nn.Module): Pre-trained PyTorch model.
        image_path (str): Path to the input image.
        transform (callable, optional): Optional image transformations.
        device (str): Device to be used ('cpu' or 'cuda').


    Displays and saves the visualization of feature maps.
    """

    image = Image.open(image_path)
    image = transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)

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
        plt.imshow(activations[0, i, :, :], cmap='jet')
        plt.axis('off')

    plt.savefig("model/feature_maps.png")
    plt.show()


def grad_cam(model, image_path, transform=None, device="cpu"):
    """
    Description:
        This function computes the Gradient-weighted Class Activation Mapping (Grad-CAM) for a
        given input image using a pre-trained model. It computes the class activation map (CAM)
        and visualizes it overlaid on the input image. It returns the CAM and the input image as
        NumPy arrays.


    Arguments:
        model (torch.nn.Module): Pre-trained PyTorch model.
        image_path (str): Path to the input image.
        transform (callable, optional): Optional image transformations.
        device (str): Device to be used ('cpu', 'cuda').


    Returns:
        cam (np.ndarray): Gradient-weighted Class Activation Map (CAM) as a NumPy array.
        img (np.ndarray): Input image as a NumPy array.
    """

    # set the model to evaluation mode
    model.eval()

    # Unfreeze the desired layer
    model.unfreeze_layer('layer4.1.conv2.weight')
    model.unfreeze_layer('layer4.1.conv2.bias')

    # GradCam Operation
    target_layers = [model.base_model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    img_ = Image.open(image_path)
    img = transform(img_).unsqueeze(0)
    img = img.to(device)

    grayscale_cam = cam(input_tensor=img)

    grayscale_cam = grayscale_cam[0, :]

    # Overlay the heatmap on the original image
    plt.imshow(img.squeeze().cpu().numpy().transpose(1, 2, 0))

    # Overlay the heatmap with transparency
    plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)

    # Add a colorbar to show the intensity scale of the heatmap
    plt.colorbar()

    plt.show()


def calculator(x1, y1, x2, y2, w, h):
    """
    Description:
        Calculates the slope and intercept of a line and adjusts the line coordinates based on the image dimensions.


    Arguments:
        x1 (int): X-coordinate of the first point.
        y1 (int): Y-coordinate of the first point.
        x2 (int): X-coordinate of the second point.
        y2 (int): Y-coordinate of the second point.
        w (int): Width of the image.
        h (int): Height of the image.


    Returns:
        tuple: Adjusted coordinates (x1, y1, x2, y2) and the slope and intercept of the line.
    """

    # Calculate the slope and intercept of the line
    slope = (y2 - y1 + 1e-5) / (x2 - x1 + 1e-5)
    intercept = y1 - slope * x1

    # Set the starting and ending points for the line
    x1 = w
    y1 = int(slope * x1 + intercept)

    x2 = int((h - intercept) // slope)
    y2 = h

    return x1, y1, x2, y2, slope, intercept


def image_cropper(image):
    """

    :param image:
    :return:
    """
    h, w = image.shape[:2]
    w, h = w // 6, h // 6
    image = cv2.resize(image, (w, h))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    g_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # if gray_image.dtype != np.uint8:
    #     gray_image = gray_image.astype(np.uint8)

    edges = cv2.Canny(g_image, threshold1=100, threshold2=300)

    # Define the contour
    contour = np.array([[w // 4, h // 8], [3 * (w // 4), h // 8], [3 * (w // 4), 7 * (h // 8)], [w // 4, 7 * (h // 8)]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.int32([contour]), 255)
    outside_mask = cv2.bitwise_not(mask)

    # Apply the outside_mask to the edges image to keep the portion outside the contour
    masked_edges = cv2.bitwise_and(edges, outside_mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        maxLineGap=10
    )

    if lines is not None:
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x1 > 3 * (w // 4) and y1 < 7 * (h // 8):
                right_lines.append(line)
            elif x1 < w // 4 and y1 < 7 * (h // 8):
                left_lines.append(line)

        x1_l = 0
        x1_r = x2_r = w
        y_u = h // 2
        y_d = h

        if len(left_lines) > 0:
            left_line_avg = np.mean(left_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = left_line_avg[0]
            x1, y1, x2, y2, slope, intercept = calculator(x1, y1, x2, y2, w, h)
            x1_l = int((h // 2 - intercept) // slope)

            # cv2.line(image, (x1_l, h//2), (x2_l, h), (0, 0, 255), 2)

        if len(right_lines) > 0:
            right_line_avg = np.mean(right_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = right_line_avg[0]
            x1, y1, x2, y2, slope, intercept = calculator(x1, y1, x2, y2, w, h)
            x1_r = int((h // 2 - intercept) // slope)
            x2_r = int((h - intercept) // slope)

            # cv2.line(image, (x1_r, h//2), (x2_r, h), (0, 0, 255), 2)

        # original_points = np.float32([[x1_l, y_u], [x1_r, y_u], [x2_r, y_d], [x2_l, y_d]])
        original_points = np.float32([[x1_l, y_u], [x1_r, y_u], [x2_r, y_d]])
        target_points = np.float32([[0, 0], [w, 0], [w, h]])

        affine_matrix = cv2.getAffineTransform(original_points, target_points)

        # Apply the affine transformation to the image
        affine_image = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]))
        equalized_image = histogram_equalization_color(affine_image)

        return equalized_image

    image = image[h // 2:h, :]
    image = histogram_equalization_color(image)

    return image


def histogram_equalization_color(image):
    """
    Description:
        Crops and processes an image to emphasize specific regions of interest.


    Arguments:
        image (np.ndarray): Input image.


    Returns:
        np.ndarray: Processed image.
    """

    # Split the color image into individual channels
    b, g, r = cv2.split(image)

    # Apply histogram equalization to each channel separately
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # Merge the equalized channels back into a color image
    equalized_image = cv2.merge([b_eq, g_eq, r_eq])

    return equalized_image
