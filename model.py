# Import necessary libraries
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
    transforms.ToTensor()
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
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    # print(f'Epoch: {epoch + 1}/{num_epochs}, '
    #       f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}, '
    #       f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}')
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
