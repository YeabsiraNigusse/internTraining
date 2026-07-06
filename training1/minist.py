"""
Chapter 5
Linear MNIST Classifier

This program trains the simplest possible neural network:
784 inputs
    ↓
10 outputs

No hidden layers.

The goal is to understand the training pipeline,
not to achieve state-of-the-art accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Convert every image into a PyTorch tensor.
# Pixel values become floats in the range [0,1].
transform = transforms.ToTensor()

# Training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


class LinearClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # 784 pixels
        # ↓
        # 10 output scores
        self.linear = nn.Linear(784, 10)

    def forward(self, x):

        # x shape:
        #
        # (batch,1,28,28)

        # Flatten image
        x = x.view(x.size(0), -1)

        # Compute logits
        logits = self.linear(x)

        return logits
    
model = LinearClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)


epochs = 5

for epoch in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        # ------------------------------------
        # Step 1
        # Forward pass
        # ------------------------------------
        logits = model(images)

        # ------------------------------------
        # Step 2
        # Compute loss
        # ------------------------------------
        loss = criterion(logits, labels)

        # ------------------------------------
        # Step 3
        # Clear old gradients
        # ------------------------------------
        optimizer.zero_grad()

        # ------------------------------------
        # Step 4
        # Backpropagation
        # ------------------------------------
        loss.backward()

        # ------------------------------------
        # Step 5
        # Update parameters
        # ------------------------------------
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1} Loss = "
        f"{running_loss/len(train_loader):.4f}"
    )

correct = 0
total = 0

# Turn off gradient computation
# because we're only doing inference.
with torch.no_grad():

    for images, labels in test_loader:

        logits = model(images)

        # Find the class with the highest score.
        predictions = logits.argmax(dim=1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total

print(f"Accuracy = {accuracy:.2f}%")


# Get one image from the test set
image, label = test_dataset[0]

# Add a batch dimension.
# The model expects shape (batch_size, channels, height, width).
image = image.unsqueeze(0)

# Turn off gradients for inference
with torch.no_grad():
    logits = model(image)

# The predicted digit is the index of the largest logit
prediction = logits.argmax(dim=1).item()

print(f"True label     : {label}")
print(f"Predicted label: {prediction}")


