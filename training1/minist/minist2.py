"""
train.py

Part 1
- Load MNIST
- Create DataLoaders
- Inspect the dataset
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import MNISTClassifier
from pathlib import Path
# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

BATCH_SIZE = 64

# -------------------------------------------------------
# Image Transform
# -------------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
])

# -------------------------------------------------------
# Load Training Dataset
# -------------------------------------------------------

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# -------------------------------------------------------
# Load Test Dataset
# -------------------------------------------------------

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

# -------------------------------------------------------
# DataLoaders
# -------------------------------------------------------

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# -------------------------------------------------------
# Dataset Information
# -------------------------------------------------------

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print(f"Training Images : {len(train_dataset)}")
print(f"Testing Images  : {len(test_dataset)}")

print()

# -------------------------------------------------------
# First Sample
# -------------------------------------------------------

image, label = train_dataset[0]

print(f"Image Shape : {image.shape}")
print(f"Label       : {label}")

print()

# -------------------------------------------------------
# First Batch
# -------------------------------------------------------

images, labels = next(iter(train_loader))

print(f"Batch Images Shape : {images.shape}")
print(f"Batch Labels Shape : {labels.shape}")

print()

# -------------------------------------------------------
# Visualize First Image
# -------------------------------------------------------

plt.imshow(images[0].squeeze(), cmap="gray")
plt.title(f"Label : {labels[0].item()}")
plt.axis("off")
plt.show()



# -------------------------------------------------------
# Device
# -------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# -------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------

LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# -------------------------------------------------------
# Model
# -------------------------------------------------------

model = MNISTClassifier().to(device)

# -------------------------------------------------------
# Loss Function
# -------------------------------------------------------

criterion = torch.nn.CrossEntropyLoss()

# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# -------------------------------------------------------
# Model Summary
# -------------------------------------------------------

print(model)

print()

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Total Parameters     : {total_parameters}")
print(f"Trainable Parameters : {trainable_parameters}")

print()

# -------------------------------------------------------
# Test Forward Pass
# -------------------------------------------------------

images, labels = next(iter(train_loader))

images = images.to(device)

outputs = model(images)

print(f"Input Shape  : {images.shape}")
print(f"Output Shape : {outputs.shape}")

print()

# -------------------------------------------------------
# Predicted Classes
# -------------------------------------------------------

predictions = torch.argmax(outputs, dim=1)

print("Predictions")
print(predictions)

print()

print("Ground Truth")
print(labels)

print()

# -------------------------------------------------------
# Test Loss
# -------------------------------------------------------

loss = criterion(
    outputs,
    labels.to(device),
)

print(f"Initial Loss : {loss.item():.4f}")


# ============================================================
# PART 3
# Training Loop
#
# Append this to train.py
# ============================================================


# ------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------

SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

best_accuracy = 0.0

train_loss_history = []
train_accuracy_history = []

test_loss_history = []
test_accuracy_history = []

# ============================================================
# Training Function
# ============================================================

def train_one_epoch(model,
                    dataloader,
                    criterion,
                    optimizer,
                    device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        # ----------------------------------------
        # Clear previous gradients
        # ----------------------------------------
        optimizer.zero_grad()

        # ----------------------------------------
        # Forward pass
        # ----------------------------------------
        outputs = model(images)

        # ----------------------------------------
        # Compute loss
        # ----------------------------------------
        loss = criterion(outputs, labels)

        # ----------------------------------------
        # Backpropagation
        # ----------------------------------------
        loss.backward()

        # ----------------------------------------
        # Update parameters
        # ----------------------------------------
        optimizer.step()

        # ----------------------------------------
        # Statistics
        # ----------------------------------------
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Evaluation Function
# ============================================================

@torch.no_grad()
def evaluate(model,
             dataloader,
             criterion,
             device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Main Training Loop
# ============================================================

print("=" * 60)
print("Starting Training")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_acc)

    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_acc)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc:.2f}% "
        f"| Test Loss: {test_loss:.4f} "
        f"| Test Acc: {test_acc:.2f}%"
    )

    if test_acc > best_accuracy:

        best_accuracy = test_acc

        torch.save(
            model.state_dict(),
            SAVE_DIR / "best_model.pth",
        )

        print(
            f"New best model saved "
            f"(Accuracy: {best_accuracy:.2f}%)"
        )

print()

print("=" * 60)
print("Training Finished")
print("=" * 60)

print(f"Best Test Accuracy : {best_accuracy:.2f}%")
print()

# ============================================================
# Load Best Model
# ============================================================

model.load_state_dict(
    torch.load(
        SAVE_DIR / "best_model.pth",
        map_location=device,
    )
)

print("Best model loaded successfully.")

# ============================================================
# Final Evaluation
# ============================================================

final_loss, final_accuracy = evaluate(
    model,
    test_loader,
    criterion,
    device,
)

print()
print("=" * 60)
print("Final Evaluation")
print("=" * 60)

print(f"Loss     : {final_loss:.4f}")
print(f"Accuracy : {final_accuracy:.2f}%")

print()

# ============================================================
# Show Some Predictions
# ============================================================

model.eval()

images, labels = next(iter(test_loader))

images = images.to(device)

outputs = model(images)

predictions = outputs.argmax(dim=1)

print("=" * 60)
print("Sample Predictions")
print("=" * 60)

for i in range(10):

    print(
        f"Image {i:2d} | "
        f"Prediction: {predictions[i].item()} | "
        f"Ground Truth: {labels[i].item()}"
    )


