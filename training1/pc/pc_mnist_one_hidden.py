"""
pc_mnist_one_hidden.py

A minimal, real MNIST predictive-coding example with ONE hidden layer.

This is deliberately written to look close to the pseudocode:

    predicted_input  = hidden @ W_x + b_x
    predicted_hidden = label_one_hot @ W_y + b_h
    eps_input        = input - predicted_input
    eps_hidden       = hidden - predicted_hidden
    hidden           = hidden - eta * dE/dhidden

The model is generative/top-down during training:

    label y  ---> hidden h ---> image x

During training, x and y are clamped, and h is inferred by an inner loop.
During testing, y is unknown, so we try every possible digit 0..9, infer h for
that candidate label, compute its free energy, and choose the label with the
lowest energy.

This is slower than standard backprop because every batch uses an inner
inference loop.

Run examples:

    python pc_mnist_one_hidden.py --epochs 3 --train_subset 10000 --test_subset 2000

Small debug run:

    python pc_mnist_one_hidden.py --epochs 1 --train_subset 1024 --test_subset 256 --debug_first_batch

No MNIST download smoke test:

    python pc_mnist_one_hidden.py --smoke_test
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ============================================================
# Configuration
# ============================================================


@dataclass
class PCConfig:
    image_dim: int = 28 * 28
    num_classes: int = 10
    hidden_dim: int = 128

    # Hidden-state inference loop
    inference_steps: int = 20
    eval_inference_steps: int = 25
    hidden_lr: float = 0.8
    hidden_clip: float = 5.0

    # Energy weights
    recon_weight: float = 1.0
    hidden_weight: float = 2.0

    # Slow weight learning
    weight_lr: float = 0.02
    weight_decay: float = 1e-5

    # Use local PC/Hebbian-style updates or Adam on the PC energy.
    # local is closest to the pseudocode.
    weight_update: str = "local"
    adam_lr: float = 1e-3


# ============================================================
# Utility helpers
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def flatten_mnist(images: torch.Tensor) -> torch.Tensor:
    """[B, 1, 28, 28] -> [B, 784]."""
    return images.view(images.size(0), -1)



def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()



def maybe_subset(dataset, subset_size: Optional[int], seed: int):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


# ============================================================
# One-hidden-layer predictive-coding model
# ============================================================


class OneHiddenPC(nn.Module):
    """
    One-hidden-layer generative PC model.

    Shapes:
        x_flat:      [B, 784]
        y_one_hot:   [B, 10]
        hidden h:    [B, H]

    Generative predictions:
        predicted_input  = h @ W_x + b_x
        predicted_hidden = y_one_hot @ W_y + b_h

    Matrix shapes:
        W_x: [H, 784]   hidden -> image
        W_y: [10, H]    label  -> hidden
    """

    def __init__(self, config: PCConfig):
        super().__init__()
        self.config = config

        # why do we need scaling here 
        scale_x = 1.0 / math.sqrt(config.hidden_dim)
        scale_y = 1.0 / math.sqrt(config.num_classes)

        self.W_x = nn.Parameter(torch.randn(config.hidden_dim, config.image_dim) * scale_x)
        self.b_x = nn.Parameter(torch.zeros(config.image_dim))

        self.W_y = nn.Parameter(torch.randn(config.num_classes, config.hidden_dim) * scale_y)
        self.b_h = nn.Parameter(torch.zeros(config.hidden_dim))

    def predict_input(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.W_x + self.b_x

    def predict_hidden_from_label(self, y_one_hot: torch.Tensor) -> torch.Tensor:
        return y_one_hot @ self.W_y + self.b_h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================
# PC energy and hidden-state inference
# ============================================================

# this function give us the total energy of the model
def pc_energy_parts(
    model: OneHiddenPC,
    x_flat: torch.Tensor,
    y_one_hot: torch.Tensor,
    hidden: torch.Tensor,
    config: PCConfig,
) -> Dict[str, torch.Tensor]:
    """
    Computes per-sample predictive-coding energy.

    E = recon_weight  * 0.5 * mean((x - x_hat)^2)
      + hidden_weight * 0.5 * mean((h - h_hat)^2)

    where:
        x_hat = predicted input from hidden
        h_hat = predicted hidden from clamped label
    """
    predicted_input = model.predict_input(hidden)
    predicted_hidden = model.predict_hidden_from_label(y_one_hot)

    eps_input = x_flat - predicted_input
    eps_hidden = hidden - predicted_hidden

    recon_per_sample = 0.5 * eps_input.pow(2).mean(dim=1)
    hidden_per_sample = 0.5 * eps_hidden.pow(2).mean(dim=1)

    total_per_sample = (
        config.recon_weight * recon_per_sample
        + config.hidden_weight * hidden_per_sample
    )

    return {
        "total_per_sample": total_per_sample,
        "recon_per_sample": recon_per_sample,
        "hidden_per_sample": hidden_per_sample,
        "eps_input": eps_input,
        "eps_hidden": eps_hidden,
        "predicted_input": predicted_input,
        "predicted_hidden": predicted_hidden,
    }


@torch.no_grad()
def infer_hidden_manual(
    model: OneHiddenPC,
    x_flat: torch.Tensor,
    y_one_hot: torch.Tensor,
    config: PCConfig,
    steps: Optional[int] = None,
    collect_trace: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float], Optional[list]]:
    """
    Fast PC inference loop: optimize hidden while keeping weights fixed.

    This uses the explicit gradient for the linear one-hidden-layer energy:

        grad_h = - W_x @ eps_input + eps_hidden

    with dimension-normalization and energy weights added.
    """
    if steps is None:
        steps = config.inference_steps

    # Initial hidden guess. During training y is clamped to the correct label.
    # So a natural initial guess is the top-down hidden prediction from y.
    hidden = model.predict_hidden_from_label(y_one_hot).detach().clone()

    trace = [] if collect_trace else None

    for step in range(steps + 1):
        parts = pc_energy_parts(model, x_flat, y_one_hot, hidden, config)
        eps_input = parts["eps_input"]
        eps_hidden = parts["eps_hidden"]

        if collect_trace:
            trace.append(
                {
                    "step": step,
                    "energy": parts["total_per_sample"].mean().item(),
                    "recon": parts["recon_per_sample"].mean().item(),
                    "hidden": parts["hidden_per_sample"].mean().item(),
                    "eps_input_rms": eps_input.pow(2).mean().sqrt().item(),
                    "eps_hidden_rms": eps_hidden.pow(2).mean().sqrt().item(),
                }
            )

        if step == steps:
            break

        # d/dh of 0.5 * mean((x - hW)^2)
        # = -(eps_input @ W_x.T) / image_dim
        grad_from_input_error = -config.recon_weight * (eps_input @ model.W_x.t()) / config.image_dim

        # d/dh of 0.5 * mean((h - h_hat)^2)
        # = eps_hidden / hidden_dim
        grad_from_hidden_error = config.hidden_weight * eps_hidden / config.hidden_dim

        grad_hidden = grad_from_input_error + grad_from_hidden_error

        hidden -= config.hidden_lr * grad_hidden
        hidden.clamp_(-config.hidden_clip, config.hidden_clip)

    final_parts = pc_energy_parts(model, x_flat, y_one_hot, hidden, config)
    stats = {
        "energy": final_parts["total_per_sample"].mean().item(),
        "recon": final_parts["recon_per_sample"].mean().item(),
        "hidden": final_parts["hidden_per_sample"].mean().item(),
        "eps_input_rms": final_parts["eps_input"].pow(2).mean().sqrt().item(),
        "eps_hidden_rms": final_parts["eps_hidden"].pow(2).mean().sqrt().item(),
    }

    return hidden.detach(), stats, trace


# ============================================================
# Weight learning
# ============================================================


@torch.no_grad()
def local_pc_weight_update(
    model: OneHiddenPC,
    x_flat: torch.Tensor,
    y_one_hot: torch.Tensor,
    hidden: torch.Tensor,
    config: PCConfig,
) -> Dict[str, float]:
    """
    Slow PC weight update after hidden inference converges.

    This is close to the update in your pseudocode:

        W_x += lr * hidden.T @ eps_input
        W_y += lr * label.T  @ eps_hidden

    Meaning:
        input error teaches the hidden->input weights
        hidden error teaches the label->hidden weights
    """
    batch_size = x_flat.size(0)

    parts = pc_energy_parts(model, x_flat, y_one_hot, hidden, config)
    eps_input = parts["eps_input"]
    eps_hidden = parts["eps_hidden"]

    lr = config.weight_lr

    if config.weight_decay > 0:
        model.W_x.mul_(1.0 - lr * config.weight_decay)
        model.W_y.mul_(1.0 - lr * config.weight_decay)

    # Local Hebbian/error updates.
    model.W_x.add_(lr * config.recon_weight * (hidden.t() @ eps_input) / batch_size)
    model.b_x.add_(lr * config.recon_weight * eps_input.mean(dim=0))

    model.W_y.add_(lr * config.hidden_weight * (y_one_hot.t() @ eps_hidden) / batch_size)
    model.b_h.add_(lr * config.hidden_weight * eps_hidden.mean(dim=0))

    return {
        "energy": parts["total_per_sample"].mean().item(),
        "recon": parts["recon_per_sample"].mean().item(),
        "hidden": parts["hidden_per_sample"].mean().item(),
        "eps_input_rms": eps_input.pow(2).mean().sqrt().item(),
        "eps_hidden_rms": eps_hidden.pow(2).mean().sqrt().item(),
    }


def adam_pc_weight_update(
    model: OneHiddenPC,
    optimizer: torch.optim.Optimizer,
    x_flat: torch.Tensor,
    y_one_hot: torch.Tensor,
    hidden: torch.Tensor,
    config: PCConfig,
) -> Dict[str, float]:
    """
    Alternative slow weight update using Adam on the same PC energy.

    The hidden state is detached, so Adam is only updating the model weights
    after the PC inference loop has settled.
    """
    optimizer.zero_grad(set_to_none=True)
    parts = pc_energy_parts(model, x_flat, y_one_hot, hidden.detach(), config)
    loss = parts["total_per_sample"].mean()
    loss.backward()
    optimizer.step()

    return {
        "energy": loss.detach().item(),
        "recon": parts["recon_per_sample"].mean().detach().item(),
        "hidden": parts["hidden_per_sample"].mean().detach().item(),
        "eps_input_rms": parts["eps_input"].pow(2).mean().sqrt().detach().item(),
        "eps_hidden_rms": parts["eps_hidden"].pow(2).mean().sqrt().detach().item(),
    }


# ============================================================
# Prediction by free energy
# ============================================================


@torch.no_grad()
def predict_by_free_energy(
    model: OneHiddenPC,
    images: torch.Tensor,
    config: PCConfig,
    steps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict labels without a feedforward classifier.

    For each candidate digit c in 0..9:
        1. clamp y=c
        2. infer hidden h
        3. compute final energy E(x, h, y=c)

    Choose the class with the lowest final energy.

    Returns:
        predictions: [B]
        energies:    [B, 10]
    """
    x_flat = flatten_mnist(images)
    batch_size = x_flat.size(0)
    energies = []

    for candidate in range(config.num_classes):
        labels = torch.full(
            (batch_size,),
            candidate,
            dtype=torch.long,
            device=images.device,
        )
        y_oh = one_hot(labels, config.num_classes)
        hidden, _, _ = infer_hidden_manual(
            model,
            x_flat,
            y_oh,
            config,
            steps=steps if steps is not None else config.eval_inference_steps,
            collect_trace=False,
        )
        parts = pc_energy_parts(model, x_flat, y_oh, hidden, config)
        energies.append(parts["total_per_sample"])

    energy_matrix = torch.stack(energies, dim=1)  # [B, 10]
    predictions = energy_matrix.argmin(dim=1)
    return predictions, energy_matrix


# ============================================================
# Data loading
# ============================================================


def build_mnist_loaders(args):
    # Import torchvision here so --smoke_test can run even in minimal environments.
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])

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
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_dataset, test_dataset, train_loader, test_loader


# ============================================================
# Training and evaluation
# ============================================================


def print_model_explanation(model: OneHiddenPC, config: PCConfig) -> None:
    print("=" * 70)
    print("One-hidden-layer Predictive Coding MNIST model")
    print("=" * 70)
    print(model)
    print()
    print(f"Total parameters: {model.num_parameters():,}")
    print()
    print("Shape meaning")
    print("-------------")
    print("MNIST image x:           [B, 1, 28, 28]")
    print("Flattened image x_flat:  [B, 784]")
    print(f"Label one-hot y:         [B, {config.num_classes}]")
    print(f"Hidden state h:          [B, {config.hidden_dim}]")
    print(f"W_x hidden->image:       [{config.hidden_dim}, 784]")
    print(f"W_y label->hidden:       [{config.num_classes}, {config.hidden_dim}]")
    print()
    print("PC equations")
    print("------------")
    print("predicted_input  = h @ W_x + b_x")
    print("predicted_hidden = y_one_hot @ W_y + b_h")
    print("eps_input        = x_flat - predicted_input")
    print("eps_hidden       = h - predicted_hidden")
    print("E_PC             = recon_weight * 0.5 * mean(eps_input^2)")
    print("                 + hidden_weight * 0.5 * mean(eps_hidden^2)")
    print()
    print("Training difference from backprop")
    print("---------------------------------")
    print("Backprop: x -> network -> logits -> CE loss -> loss.backward() -> optimizer.step()")
    print("PC here:  clamp x and y -> infer h for many steps -> update W_x and W_y from errors")
    print("=" * 70)



def debug_first_batch(
    model: OneHiddenPC,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: PCConfig,
) -> None:
    print("\n" + "=" * 70)
    print("Debugging one PC inference loop on the first batch")
    print("=" * 70)

    x_flat = flatten_mnist(images)
    y_oh = one_hot(labels, config.num_classes)

    print(f"images shape:       {tuple(images.shape)}")
    print(f"x_flat shape:       {tuple(x_flat.shape)}")
    print(f"labels shape:       {tuple(labels.shape)}")
    print(f"y_one_hot shape:    {tuple(y_oh.shape)}")

    hidden0 = model.predict_hidden_from_label(y_oh)
    print(f"initial hidden h0:   {tuple(hidden0.shape)}")
    print(f"predicted input:     {tuple(model.predict_input(hidden0).shape)}")

    hidden, stats, trace = infer_hidden_manual(
        model,
        x_flat,
        y_oh,
        config,
        steps=min(config.inference_steps, 10),
        collect_trace=True,
    )

    print(f"settled hidden h:    {tuple(hidden.shape)}")
    print()
    print("Inner inference trace")
    print("step | energy    | recon     | hidden    | eps_x_rms | eps_h_rms")
    print("-----+-----------+-----------+-----------+-----------+----------")
    for row in trace:
        print(
            f"{row['step']:4d} | "
            f"{row['energy']:.6f} | "
            f"{row['recon']:.6f} | "
            f"{row['hidden']:.6f} | "
            f"{row['eps_input_rms']:.6f} | "
            f"{row['eps_hidden_rms']:.6f}"
        )
    print("=" * 70 + "\n")



def train_one_epoch(
    model: OneHiddenPC,
    dataloader: DataLoader,
    config: PCConfig,
    device: torch.device,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    log_interval: int = 100,
) -> Dict[str, float]:
    model.train()

    totals = {
        "energy": 0.0,
        "recon": 0.0,
        "hidden": 0.0,
        "eps_input_rms": 0.0,
        "eps_hidden_rms": 0.0,
    }
    n_samples = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        x_flat = flatten_mnist(images)
        y_oh = one_hot(labels, config.num_classes)

        # ----------------------------------------------------
        # Fast loop: infer hidden state while weights are fixed
        # ----------------------------------------------------
        hidden, infer_stats, _ = infer_hidden_manual(
            model,
            x_flat,
            y_oh,
            config,
            steps=config.inference_steps,
            collect_trace=False,
        )

        # ----------------------------------------------------
        # Slow loop: update weights after hidden settles
        # ----------------------------------------------------
        if config.weight_update == "local":
            update_stats = local_pc_weight_update(
                model, x_flat, y_oh, hidden, config
            )
        elif config.weight_update == "adam":
            if optimizer is None:
                raise ValueError("Adam optimizer must be provided when weight_update='adam'.")
            update_stats = adam_pc_weight_update(
                model, optimizer, x_flat, y_oh, hidden, config
            )
        else:
            raise ValueError(f"Unknown weight_update: {config.weight_update}")

        batch_size = images.size(0)
        n_samples += batch_size
        for key in totals:
            totals[key] += update_stats[key] * batch_size

        if batch_idx % log_interval == 0:
            # Quick free-energy accuracy on a small part of this batch.
            eval_images = images[: min(32, images.size(0))]
            eval_labels = labels[: min(32, labels.size(0))]
            preds, _ = predict_by_free_energy(
                model,
                eval_images,
                config,
                steps=min(config.eval_inference_steps, 10),
            )
            mini_acc = 100.0 * (preds == eval_labels).float().mean().item()

            print(
                f"Epoch {epoch:02d} | Batch {batch_idx:04d}/{len(dataloader):04d} | "
                f"E={update_stats['energy']:.5f} | "
                f"Recon={update_stats['recon']:.5f} | "
                f"Hidden={update_stats['hidden']:.5f} | "
                f"eps_x={update_stats['eps_input_rms']:.4f} | "
                f"eps_h={update_stats['eps_hidden_rms']:.4f} | "
                f"mini free-energy acc={mini_acc:.1f}%"
            )

    return {key: totals[key] / max(1, n_samples) for key in totals}


@torch.no_grad()
def evaluate(
    model: OneHiddenPC,
    dataloader: DataLoader,
    config: PCConfig,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    total = 0
    correct = 0
    total_energy = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        preds, energy_matrix = predict_by_free_energy(
            model,
            images,
            config,
            steps=config.eval_inference_steps,
        )

        winning_energy = energy_matrix.min(dim=1).values
        total_energy += winning_energy.sum().item()
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return {
        "loss": total_energy / max(1, total),
        "accuracy": 100.0 * correct / max(1, total),
    }


@torch.no_grad()
def show_sample_predictions(
    model: OneHiddenPC,
    dataloader: DataLoader,
    config: PCConfig,
    device: torch.device,
    n: int = 10,
) -> None:
    model.eval()
    images, labels = next(iter(dataloader))
    images = images[:n].to(device)
    labels = labels[:n].to(device)

    preds, energies = predict_by_free_energy(model, images, config)

    print("\n" + "=" * 70)
    print("Sample predictions by lowest predictive-coding free energy")
    print("=" * 70)
    for i in range(images.size(0)):
        winning_energy = energies[i, preds[i]].item()
        true_energy = energies[i, labels[i]].item()
        print(
            f"Image {i:02d} | Prediction: {preds[i].item()} | "
            f"Ground Truth: {labels[i].item()} | "
            f"E_pred={winning_energy:.5f} | E_true={true_energy:.5f}"
        )


# ============================================================
# Smoke test
# ============================================================


def run_smoke_test(device: torch.device) -> None:
    print("Running smoke test with random MNIST-shaped tensors.")
    config = PCConfig(hidden_dim=32, inference_steps=5, eval_inference_steps=5)
    model = OneHiddenPC(config).to(device)

    images = torch.rand(8, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (8,), device=device)
    x_flat = flatten_mnist(images)
    y_oh = one_hot(labels, config.num_classes)

    hidden, stats, trace = infer_hidden_manual(
        model, x_flat, y_oh, config, collect_trace=True
    )
    update_stats = local_pc_weight_update(model, x_flat, y_oh, hidden, config)
    preds, energies = predict_by_free_energy(model, images, config)

    print(f"images:       {tuple(images.shape)}")
    print(f"x_flat:       {tuple(x_flat.shape)}")
    print(f"y_one_hot:    {tuple(y_oh.shape)}")
    print(f"hidden:       {tuple(hidden.shape)}")
    print(f"energies:     {tuple(energies.shape)}")
    print(f"predictions:  {tuple(preds.shape)}")
    print(f"final stats:  {stats}")
    print(f"update stats: {update_stats}")
    print("Smoke test passed.")


# ============================================================
# Main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-hidden-layer PC model on MNIST")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--eval_inference_steps", type=int, default=25)
    parser.add_argument("--hidden_lr", type=float, default=0.8)
    parser.add_argument("--weight_lr", type=float, default=0.02)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--hidden_weight", type=float, default=2.0)
    parser.add_argument("--weight_update", type=str, default="local", choices=["local", "adam"])
    parser.add_argument("--train_subset", type=int, default=10000)
    parser.add_argument("--test_subset", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="pc_checkpoints")
    parser.add_argument("--debug_first_batch", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.smoke_test:
        run_smoke_test(device)
        return

    config = PCConfig(
        hidden_dim=args.hidden_dim,
        inference_steps=args.inference_steps,
        eval_inference_steps=args.eval_inference_steps,
        hidden_lr=args.hidden_lr,
        weight_lr=args.weight_lr,
        adam_lr=args.adam_lr,
        recon_weight=args.recon_weight,
        hidden_weight=args.hidden_weight,
        weight_update=args.weight_update,
    )

    train_dataset, test_dataset, train_loader, test_loader = build_mnist_loaders(args)

    print("=" * 70)
    print("Dataset Information")
    print("=" * 70)
    print(f"Training images used: {len(train_dataset)}")
    print(f"Testing images used:  {len(test_dataset)}")
    first_image, first_label = train_dataset[0]
    print(f"First image shape:    {tuple(first_image.shape)}")
    print(f"First label:          {first_label}")

    model = OneHiddenPC(config).to(device)
    print_model_explanation(model, config)

    optimizer = None
    if config.weight_update == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
        print(f"Using Adam for slow PC weight updates with lr={config.adam_lr}")
    else:
        print(f"Using local PC error-based weight updates with lr={config.weight_lr}")

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    print("\nInitial one-batch free-energy prediction check")
    init_preds, _ = predict_by_free_energy(
        model,
        images[: min(32, images.size(0))],
        config,
        steps=min(config.eval_inference_steps, 10),
    )
    init_acc = 100.0 * (init_preds == labels[: init_preds.numel()]).float().mean().item()
    print(f"Initial mini free-energy accuracy: {init_acc:.2f}%")

    if args.debug_first_batch:
        debug_first_batch(model, images, labels, config)

    save_dir = Path(args.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_path = save_dir / "best_pc_one_hidden.pth"

    print("\n" + "=" * 70)
    print("Starting Predictive Coding Training")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            config,
            device,
            epoch=epoch,
            optimizer=optimizer,
            log_interval=args.log_interval,
        )

        test_stats = evaluate(model, test_loader, config, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train E: {train_stats['energy']:.5f} | "
            f"Train Recon: {train_stats['recon']:.5f} | "
            f"Train Hidden: {train_stats['hidden']:.5f} | "
            f"Test E: {test_stats['loss']:.5f} | "
            f"Test Acc: {test_stats['accuracy']:.2f}%"
        )

        if test_stats["accuracy"] > best_accuracy:
            best_accuracy = test_stats["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "args": vars(args),
                    "best_accuracy": best_accuracy,
                },
                best_path,
            )
            print(f"New best PC model saved: {best_path} | Acc={best_accuracy:.2f}%")

    print("\n" + "=" * 70)
    print("Training Finished")
    print("=" * 70)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_stats = evaluate(model, test_loader, config, device)

    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    print(f"Final free-energy loss: {final_stats['loss']:.5f}")
    print(f"Final accuracy:         {final_stats['accuracy']:.2f}%")

    show_sample_predictions(model, test_loader, config, device, n=10)


if __name__ == "__main__":
    main()
