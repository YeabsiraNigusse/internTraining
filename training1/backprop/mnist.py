#!/usr/bin/env python3
"""
Standard backprop MNIST baseline.

This is the ordinary neural-network baseline used by compare_pc_fluid_pc.py.
It trains MNISTClassifier from model.py with cross-entropy and Adam.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from model import MNISTClassifier
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def make_loaders(args: argparse.Namespace):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = maybe_subset(train_dataset, args.train_subset, args.seed)
    test_dataset = maybe_subset(test_dataset, args.test_subset, args.seed + 1)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, test_dataset, train_loader, test_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard backprop MNIST baseline")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_subset", type=int, default=0, help="0 means full train set")
    parser.add_argument("--test_subset", type=int, default=0, help="0 means full test set")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset, train_loader, test_loader = make_loaders(args)

    print("=" * 60)
    print("Standard Backprop MNIST")
    print("=" * 60)
    print(f"Using device: {device}")
    print(f"Training images used: {len(train_dataset)}")
    print(f"Testing images used : {len(test_dataset)}")

    model = MNISTClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(model)
    print(f"Total Parameters     : {total_parameters}")
    print(f"Trainable Parameters : {trainable_parameters}")

    save_dir = Path(args.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pth"
    best_accuracy = -1.0

    print("=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Test Loss: {test_loss:.4f} "
            f"| Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (Accuracy: {best_accuracy:.2f}%)")

    print()
    print("=" * 60)
    print("Training Finished")
    print("=" * 60)
    print(f"Best Test Accuracy : {best_accuracy:.2f}%")

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_loss, final_accuracy = evaluate(model, test_loader, criterion, device)

    print()
    print("=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    print(f"Loss     : {final_loss:.4f}")
    print(f"Accuracy : {final_accuracy:.2f}%")

    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(dim=1)

    print("=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    for i in range(min(10, images.size(0))):
        print(
            f"Image {i:2d} | "
            f"Prediction: {predictions[i].item()} | "
            f"Ground Truth: {labels[i].item()}"
        )


if __name__ == "__main__":
    main()
