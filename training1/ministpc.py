from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader



# Step 1 — Downloading MNIST

transform = transforms.ToTensor()

train_dataset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

# Step 2 — Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)

# Step 3 — Look at One Batch

images, labels = next(iter(train_loader))

print(images.shape)

images = images.view(images.size(0), -1)

