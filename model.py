# Import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# locals
from dataset import StoneDataset
from neural_net import Net

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Set batch size and number of epochs
batch_size = 50
num_epochs = 30

# Check for available device (GPU or CPU)
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Define image augmentation and transformation
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform = transforms.Compose([
    Resize((512, 512)),
    ToTensor()
])

# Load datasets
train_dataset = StoneDataset(r"data/train.csv", gp="train", transform=train_transform)
train_dataset, val_dataset = (train_test_split(train_dataset,
                                               test_size=0.2,
                                               random_state=42)
                              )  # split 20% of train data into validation set
test_dataset = StoneDataset(r"data/test.csv", gp="test", transform=transform)

# Create DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Instantiate the model and move it to the appropriate device (GPU or CPU)
model = Net()
model = model.to(device)

# Set loss function, optimizer, and learning rate scheduler
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# clear log file
open("mylog.txt", "w")

# Training loop with tqdm progress bar
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model.train()  # Set the model to training mode
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

    training_log = f'''Epoch: {epoch + 1}/{num_epochs},
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
#
#
# image = Image.open("data/train/C_processed_image_42.jpg")
# image = transform(image)
# image = image.to(device)
# image = image.unsqueeze(0)
# output = model(image)
#
# # Choose the layer whose feature maps you want to visualize
# target_layer = model.base_model.conv1
#
# # Get the activations (feature maps) from the target layer
# activations = target_layer(image).cpu()
#
# # Convert activations to numpy array
# activations = activations.detach().numpy()
#
# # Visualize the feature maps
# num_feature_maps = activations.shape[1]  # Number of feature maps
# plt.figure(figsize=(24, 24))
# for i in range(num_feature_maps):
#     plt.subplot(9, 8, i+1)  # Adjust the subplot layout based on the number of feature maps
#     plt.imshow(activations[0, i, :, :], cmap='gray')
#     plt.axis('off')
#
# plt.savefig("feature_maps.png")
# plt.show()

# set the model to evaluation mode
model.eval()
img_ = Image.open('data/train/A_processed_image_14.jpg')
img = transform(img_).unsqueeze(0)
img = img.to(device)

logits = model(img)
pred = torch.argmax(logits, dim=1)

# Compute the gradients of the predicted class output with respect to the activations of the target layer
model.zero_grad()
logits[:, pred].backward()

# Get the activations of the additional_conv layer
gradients = model.additional_conv.weight.grad  # Gradients of additional_conv with respect to the output
activations = model.additional_conv.weight

# Compute the gradient-weighted class activation map (CAM)
cam = torch.mean(gradients, dim=(2, 3))  # Global average pooling along spatial dimensions
cam = nn.functional.relu(cam)
cam = cam.detach().cpu().numpy()[0]  # Convert to numpy array

# # Normalize the CAM
cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)

# Resize the CAM to match the input image size
cam = np.uint8(255 * cam)
cam = np.uint8(Image.fromarray(cam).resize((512, 512), Image.Resampling.LANCZOS))

# Convert the input image to a numpy array
input_image_np = img.squeeze().cpu().numpy().transpose(1, 2, 0)

# Visualize the CAM overlaid on the input image
plt.imshow(input_image_np)
plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay the CAM on the input image using a jet colormap
plt.axis('off')
plt.show()
