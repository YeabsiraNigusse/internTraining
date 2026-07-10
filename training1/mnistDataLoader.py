# Import PyTorch and the MNIST dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Convert images to tensors and normalize pixel values to [0, 1]
transform = transforms.ToTensor()

# Download the training dataset (60,000 images)
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download the test dataset (10,000 images)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Create mini-batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Get one mini-batch
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)



