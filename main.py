# Import necessary libraries
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# locals
from dataset import StoneDataset
from utils import grad_cam, feature_maps, train
from model import Net

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Set batch size and number of epochs
batch_size = 50
num_epochs = 30
image_size = (600, 600)

# Check for available device (GPU or CPU)
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Define image augmentation and transformation
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform = transforms.Compose([
    Resize(image_size),
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
# Unfreeze the desired layer
model.unfreeze_layer('layer4.1.conv2.weight')
model.unfreeze_layer('layer4.1.conv2.bias')
model = model.to(device)

# Set loss function, optimizer, and learning rate scheduler
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# clear log file
open("training_log.txt", "w")

# train and test the model
model = train(model,
              train_dataloader,
              val_dataloader,
              test_dataloader,
              optimizer,
              criterion,
              num_epochs,
              device,
              scheduler)

image_ = Image.open('data/test/C_processed_image_15.jpg')
image__ = transform(image_)
image = image__.to(device)
image = image.unsqueeze(0)
print(model(image))
target_layers = [model.base_model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=image)


# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
# visualization = show_cam_on_image(image_, grayscale_cam, use_rgb=True)
plt.imshow(grayscale_cam, cmap='jet')
plt.colorbar()
plt.show()
plt.imshow(image__.numpy().transpose(1, 2, 0), cmap='jet')
plt.show()
# You can also get the model outputs without having to re-inference
model_outputs = cam.outputs

# # save the model
# torch.save(model.state_dict(), "stone_model.pth")

# CAM, image = grad_cam(model,
#                       'data/train/A_processed_image_14.jpg',
#                       image_size=image_size,
#                       transform=transform, device=device)

# grad_cam(model, 'data/train/A_processed_image_14.jpg', transform=transform, device=device)

# feature_maps(model,
#              'data/train/A_processed_image_14.jpg',
#              transform=transform, device=device)

# # Visualize the CAM overlaid on the input image
# plt.imshow(image)
# plt.imshow(CAM, cmap='jet', alpha=0.5)  # Overlay the CAM on the input image using a jet colormap
# plt.axis('off')
# plt.show()
