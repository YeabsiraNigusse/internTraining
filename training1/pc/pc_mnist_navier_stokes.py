"""
Predictive Coding MNIST with a Navier-Stokes-style hidden-state regularizer.

Run examples:
    python pc_mnist_navier_stokes.py --epochs 3 --train_subset 10000
    python pc_mnist_navier_stokes.py --epochs 1 --train_subset 1024 --debug_grid

Core idea:
    hidden grid h has shape [B, 3, 28, 28]

    h[:, 0, :, :] = u_x, horizontal latent velocity
    h[:, 1, :, :] = u_y, vertical latent velocity
    h[:, 2, :, :] = p, latent pressure

    The same hidden state is used in two views:
        1. Grid view [B, 3, 28, 28] for Navier-Stokes residuals.
        2. Flat view [B, 3*28*28] for classification.

This is a real trainable implementation, but note the physics interpretation:
MNIST digits are not physical fluids. The Navier-Stokes term is used here as a
spatial regularizer over a latent field, not as a claim that digit pixels obey
fluid mechanics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 2-D finite differences on a spatial grid
# -----------------------------------------------------------------------------


def center(z: torch.Tensor) -> torch.Tensor:
    """
    z: [B, H, W]

    Returns interior grid cells only: [B, H-2, W-2].
    We use interior cells because central differences need left/right/top/bottom
    neighbors.
    """
    return z[:, 1:-1, 1:-1]


def ddx(z: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    Central finite difference in the horizontal x direction.

    z: [B, H, W]
    output: [B, H-2, W-2]

    d z / d x at cell (row=i, col=j):
        (z[i, j+1] - z[i, j-1]) / (2 dx)
    """
    return (z[:, 1:-1, 2:] - z[:, 1:-1, :-2]) / (2.0 * dx)


def ddy(z: torch.Tensor, dy: float = 1.0) -> torch.Tensor:
    """
    Central finite difference in the vertical y direction.

    z: [B, H, W]
    output: [B, H-2, W-2]

    d z / d y at cell (row=i, col=j):
        (z[i+1, j] - z[i-1, j]) / (2 dy)
    """
    return (z[:, 2:, 1:-1] - z[:, :-2, 1:-1]) / (2.0 * dy)


def laplacian(z: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """
    2-D Laplacian using central finite differences.

    Laplacian(z) = d2z/dx2 + d2z/dy2

    z: [B, H, W]
    output: [B, H-2, W-2]
    """
    z_c = center(z)
    d2x = (z[:, 1:-1, 2:] - 2.0 * z_c + z[:, 1:-1, :-2]) / (dx * dx)
    d2y = (z[:, 2:, 1:-1] - 2.0 * z_c + z[:, :-2, 1:-1]) / (dy * dy)
    return d2x + d2y


# -----------------------------------------------------------------------------
# Navier-Stokes residual on the hidden grid
# -----------------------------------------------------------------------------


@dataclass
class NSConfig:
    nu: float = 0.01
    dx: float = 1.0
    dy: float = 1.0
    momentum_weight: float = 1.0
    divergence_weight: float = 5.0


def navier_stokes_loss(
    h_grid: torch.Tensor,
    config: NSConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute a steady incompressible Navier-Stokes residual on a hidden grid.

    h_grid shape:
        [B, 3, H, W]

    Channel interpretation:
        h_grid[:, 0] = u_x, horizontal latent velocity
        h_grid[:, 1] = u_y, vertical latent velocity
        h_grid[:, 2] = p, latent pressure

    Steady incompressible Navier-Stokes, density rho=1, no external force:

        (u . grad)u + grad(p) - nu * Laplacian(u) = 0
        div(u) = 0

    Momentum residual:
        r_x = u_x * du_x/dx + u_y * du_x/dy + dp/dx - nu * Laplacian(u_x)
        r_y = u_x * du_y/dx + u_y * du_y/dy + dp/dy - nu * Laplacian(u_y)

    Divergence residual:
        r_div = du_x/dx + du_y/dy

    If the hidden field obeys this simplified fluid law, these residuals should
    be near zero.
    """
    if h_grid.ndim != 4 or h_grid.shape[1] < 3:
        raise ValueError(
            f"Expected h_grid shape [B, at least 3, H, W], got {tuple(h_grid.shape)}"
        )
    if h_grid.shape[-1] < 3 or h_grid.shape[-2] < 3:
        raise ValueError("Navier-Stokes residual needs H >= 3 and W >= 3.")

    ux = h_grid[:, 0]  # [B, H, W]
    uy = h_grid[:, 1]  # [B, H, W]
    p = h_grid[:, 2]   # [B, H, W]

    ux_c = center(ux)
    uy_c = center(uy)

    dux_dx = ddx(ux, config.dx)
    dux_dy = ddy(ux, config.dy)
    duy_dx = ddx(uy, config.dx)
    duy_dy = ddy(uy, config.dy)

    dp_dx = ddx(p, config.dx)
    dp_dy = ddy(p, config.dy)

    lap_ux = laplacian(ux, config.dx, config.dy)
    lap_uy = laplacian(uy, config.dx, config.dy)

    # Convective acceleration terms: (u . grad) u_x and (u . grad) u_y.
    conv_x = ux_c * dux_dx + uy_c * dux_dy
    conv_y = ux_c * duy_dx + uy_c * duy_dy

    # Momentum residual: should be zero if momentum is conserved.
    r_x = conv_x + dp_dx - config.nu * lap_ux
    r_y = conv_y + dp_dy - config.nu * lap_uy

    # Divergence residual: should be zero for incompressible flow.
    r_div = dux_dx + duy_dy

    momentum = (r_x.pow(2) + r_y.pow(2)).mean()
    divergence = r_div.pow(2).mean()

    total = config.momentum_weight * momentum + config.divergence_weight * divergence

    parts = {
        "momentum": momentum,
        "divergence": divergence,
        "r_x": r_x,
        "r_y": r_y,
        "r_div": r_div,
        "dux_dx": dux_dx,
        "duy_dy": duy_dy,
        "dp_dx": dp_dx,
        "dp_dy": dp_dy,
    }
    return total, parts


# -----------------------------------------------------------------------------
# One-hidden-layer predictive-coding MNIST model
# -----------------------------------------------------------------------------


class OneHiddenPCMNIST(nn.Module):
    """
    One spatial hidden layer predictive-coding model for MNIST.

    Input image:
        x: [B, 1, 28, 28]

    Hidden state:
        h: [B, 3, 28, 28]

    Hidden channel interpretation for the physics regularizer:
        h[:, 0] = u_x
        h[:, 1] = u_y
        h[:, 2] = pressure p

    The model has three learned maps:
        initializer: image -> initial hidden guess h0
        decoder: hidden -> reconstructed image
        classifier: hidden -> digit logits

    During the PC inference loop, h is optimized while these weights stay fixed.
    During the slow learning loop, weights are optimized after h settles.
    """

    def __init__(self, hidden_channels: int = 3) -> None:
        super().__init__()
        if hidden_channels < 3:
            raise ValueError("hidden_channels must be at least 3 for ux, uy, p.")
        self.hidden_channels = hidden_channels

        # The hidden layer is spatial: same H,W as the image.
        self.initializer = nn.Conv2d(1, hidden_channels, kernel_size=5, padding=2)

        # Hidden predicts the input image. This is the PC reconstruction error.
        self.decoder = nn.Conv2d(hidden_channels, 1, kernel_size=5, padding=2)

        # Hidden predicts the label. We flatten the same hidden grid for classifier.
        self.classifier = nn.Linear(hidden_channels * 28 * 28, 10)

    def initial_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.initializer(x))

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        # MNIST pixels are in [0, 1], so sigmoid keeps reconstruction in [0, 1].
        return torch.sigmoid(self.decoder(h))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        # Grid view [B, C, H, W] -> flat matrix [B, C*H*W].
        return self.classifier(h.flatten(start_dim=1))


@dataclass
class PCConfig:
    inference_steps: int = 8
    hidden_lr: float = 0.25
    recon_weight: float = 1.0
    class_weight: float = 1.0
    prior_weight: float = 0.10
    ns_weight: float = 0.02
    hidden_clip: float = 5.0


# -----------------------------------------------------------------------------
# Predictive-coding energy and hidden-state inference
# -----------------------------------------------------------------------------


def pc_prediction_energy(
    model: OneHiddenPCMNIST,
    h: torch.Tensor,
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    h0: torch.Tensor,
    config: PCConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    PC prediction energy for one hidden state.

    Terms:
        image prediction error: hidden predicts the input image
        label prediction error: hidden predicts the digit label, if y is given
        hidden prior error: hidden should not move infinitely far from h0
    """
    x_hat = model.reconstruct(h)
    logits = model.logits(h)

    recon = F.mse_loss(x_hat, x)

    if y is None:
        cls = torch.zeros((), device=h.device, dtype=h.dtype)
    else:
        # For categorical output, cross-entropy is the label prediction error.
        cls = F.cross_entropy(logits, y)

    prior = F.mse_loss(h, h0)

    total = (
        config.recon_weight * recon
        + config.class_weight * cls
        + config.prior_weight * prior
    )

    parts = {
        "recon": recon,
        "class": cls,
        "prior": prior,
    }
    return total, parts


def infer_hidden(
    model: OneHiddenPCMNIST,
    x: torch.Tensor,
    y: Optional[torch.Tensor],
    pc_config: PCConfig,
    ns_config: NSConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Fast predictive-coding inference loop.

    We freeze weights conceptually and optimize only h:

        h <- h - eta * dE_total/dh

    where

        E_total = PC prediction error + ns_weight * Navier-Stokes residual

    Returns a detached settled hidden state.
    """
    # h0 is the network's first guess. It is not updated inside this inference loop.
    h0 = model.initial_hidden(x).detach()

    # h is now the variable being inferred.
    h = h0.clone().requires_grad_(True)

    last_stats: Dict[str, float] = {}

    for _ in range(pc_config.inference_steps):
        pc_loss, pc_parts = pc_prediction_energy(model, h, x, y, h0, pc_config)
        ns_loss, ns_parts = navier_stokes_loss(h, ns_config)
        energy = pc_loss + pc_config.ns_weight * ns_loss

        # Gradient only with respect to h, not with respect to weights.
        (grad_h,) = torch.autograd.grad(energy, h, create_graph=False)

        with torch.no_grad():
            h -= pc_config.hidden_lr * grad_h
            h.clamp_(-pc_config.hidden_clip, pc_config.hidden_clip)

        # Detach so the next iteration builds a fresh graph.
        h = h.detach().requires_grad_(True)

        last_stats = {
            "pc_total": float(pc_loss.detach().cpu()),
            "energy_total": float(energy.detach().cpu()),
            "recon": float(pc_parts["recon"].detach().cpu()),
            "class": float(pc_parts["class"].detach().cpu()),
            "prior": float(pc_parts["prior"].detach().cpu()),
            "ns_total": float(ns_loss.detach().cpu()),
            "ns_momentum": float(ns_parts["momentum"].detach().cpu()),
            "ns_divergence": float(ns_parts["divergence"].detach().cpu()),
        }

    return h.detach(), last_stats


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def make_loaders(
    batch_size: int,
    train_subset: int,
    test_subset: int,
    data_dir: str,
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if train_subset > 0:
        train_set = Subset(train_set, range(min(train_subset, len(train_set))))
    if test_subset > 0:
        test_set = Subset(test_set, range(min(test_subset, len(test_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def debug_grid_mapping(h: torch.Tensor, ns_parts: Dict[str, torch.Tensor]) -> None:
    """Prints the exact matrix-to-grid/channel interpretation once."""
    print("\n--- Grid mapping debug ---")
    print(f"hidden h shape: {tuple(h.shape)}")
    print("h[:, 0, :, :] is u_x, horizontal latent velocity")
    print("h[:, 1, :, :] is u_y, vertical latent velocity")
    print("h[:, 2, :, :] is p, latent pressure")
    print(f"u_x grid shape: {tuple(h[:, 0].shape)}")
    print(f"u_y grid shape: {tuple(h[:, 1].shape)}")
    print(f"p grid shape:   {tuple(h[:, 2].shape)}")
    print("classifier receives h.flatten(1), shape:", tuple(h.flatten(1).shape))
    print("Navier-Stokes residuals are computed on interior cells only.")
    print(f"r_x shape:   {tuple(ns_parts['r_x'].shape)}")
    print(f"r_y shape:   {tuple(ns_parts['r_y'].shape)}")
    print(f"r_div shape: {tuple(ns_parts['r_div'].shape)}")
    print("For a 28x28 grid, central differences produce 26x26 residual grids.")
    sample = h[0]
    row, col = 14, 14
    print(
        f"Example cell: h[0, 0, {row}, {col}] = u_x at pixel row={row}, col={col}: "
        f"{float(sample[0, row, col].detach().cpu()):.4f}"
    )
    print("--- End debug ---\n")


def train_one_epoch(
    model: OneHiddenPCMNIST,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pc_config: PCConfig,
    ns_config: NSConfig,
    amort_weight: float,
    ns_init_weight: float,
    epoch: int,
    debug_grid: bool = False,
) -> None:
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_count = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # Fast PC loop: infer hidden states with weights fixed.
        h_settled, infer_stats = infer_hidden(model, x, y, pc_config, ns_config)

        if debug_grid and epoch == 1 and batch_idx == 0:
            _, ns_parts = navier_stokes_loss(h_settled, ns_config)
            debug_grid_mapping(h_settled, ns_parts)

        # Slow learning loop: update weights after hidden state settled.
        optimizer.zero_grad(set_to_none=True)

        # Decoder and classifier learn from the settled hidden state.
        x_hat = model.reconstruct(h_settled)
        logits = model.logits(h_settled)
        recon_loss = F.mse_loss(x_hat, x)
        class_loss = F.cross_entropy(logits, y)

        # Initializer learns to produce the settled hidden state directly next time.
        h0 = model.initial_hidden(x)
        amort_loss = F.mse_loss(h0, h_settled)

        # Also gently regularize the initial guess itself so weights internalize
        # the physics regularizer, instead of only relying on the inference loop.
        ns_h0, _ = navier_stokes_loss(h0, ns_config)

        weight_loss = (
            pc_config.recon_weight * recon_loss
            + pc_config.class_weight * class_loss
            + amort_weight * amort_loss
            + ns_init_weight * ns_h0
        )

        weight_loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            running_correct += int((pred == y).sum().item())
            running_count += int(y.numel())
            running_loss += float(weight_loss.detach().cpu()) * y.numel()

        if batch_idx % 100 == 0:
            acc = 100.0 * running_correct / max(running_count, 1)
            avg_loss = running_loss / max(running_count, 1)
            print(
                f"epoch={epoch:02d} batch={batch_idx:04d} "
                f"loss={avg_loss:.4f} acc={acc:.2f}% "
                f"inner_E={infer_stats['energy_total']:.4f} "
                f"recon={infer_stats['recon']:.4f} "
                f"cls={infer_stats['class']:.4f} "
                f"NSmom={infer_stats['ns_momentum']:.4f} "
                f"NSdiv={infer_stats['ns_divergence']:.4f}"
            )


@torch.no_grad()
def evaluate_without_pc_inference(
    model: OneHiddenPCMNIST,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Fast evaluation: use only the initializer h0, no iterative inference.
    This tests whether the slow weights have learned useful representations.
    """
    model.eval()
    correct = 0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        h0 = model.initial_hidden(x)
        logits = model.logits(h0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        count += int(y.numel())
    return 100.0 * correct / max(count, 1)


def evaluate_with_pc_inference(
    model: OneHiddenPCMNIST,
    loader: DataLoader,
    device: torch.device,
    pc_config: PCConfig,
    ns_config: NSConfig,
) -> float:
    """
    Slower evaluation: infer h from image reconstruction + NS only.
    The label y is not given to the inference loop at test time.
    """
    model.eval()
    correct = 0
    count = 0

    # Use a copy of the config with no classification energy at test time.
    test_pc_config = PCConfig(
        inference_steps=pc_config.inference_steps,
        hidden_lr=pc_config.hidden_lr,
        recon_weight=pc_config.recon_weight,
        class_weight=0.0,
        prior_weight=pc_config.prior_weight,
        ns_weight=pc_config.ns_weight,
        hidden_clip=pc_config.hidden_clip,
    )

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        h, _ = infer_hidden(model, x, None, test_pc_config, ns_config)
        with torch.no_grad():
            logits = model.logits(h)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            count += int(y.numel())

    return 100.0 * correct / max(count, 1)


# -----------------------------------------------------------------------------
# A tiny self-test without downloading MNIST
# -----------------------------------------------------------------------------


def smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneHiddenPCMNIST(hidden_channels=3).to(device)
    pc_config = PCConfig(inference_steps=2)
    ns_config = NSConfig()
    x = torch.rand(4, 1, 28, 28, device=device)
    y = torch.tensor([0, 1, 2, 3], device=device)
    h, stats = infer_hidden(model, x, y, pc_config, ns_config)
    ns, parts = navier_stokes_loss(h, ns_config)
    assert h.shape == (4, 3, 28, 28)
    assert parts["r_x"].shape == (4, 26, 26)
    assert parts["r_y"].shape == (4, 26, 26)
    assert parts["r_div"].shape == (4, 26, 26)
    print("Smoke test passed.")
    print("hidden shape:", tuple(h.shape))
    print("stats:", stats)
    print("NS loss:", float(ns.detach().cpu()))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_subset", type=int, default=10000)
    parser.add_argument("--test_subset", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--inference_steps", type=int, default=8)
    parser.add_argument("--hidden_lr", type=float, default=0.25)
    parser.add_argument("--ns_weight", type=float, default=0.02)
    parser.add_argument("--prior_weight", type=float, default=0.10)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--class_weight", type=float, default=1.0)

    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--momentum_weight", type=float, default=1.0)
    parser.add_argument("--divergence_weight", type=float, default=5.0)
    parser.add_argument("--amort_weight", type=float, default=0.50)
    parser.add_argument("--ns_init_weight", type=float, default=0.005)

    parser.add_argument("--debug_grid", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_test:
        smoke_test()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        data_dir=args.data_dir,
    )

    model = OneHiddenPCMNIST(hidden_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pc_config = PCConfig(
        inference_steps=args.inference_steps,
        hidden_lr=args.hidden_lr,
        recon_weight=args.recon_weight,
        class_weight=args.class_weight,
        prior_weight=args.prior_weight,
        ns_weight=args.ns_weight,
    )
    ns_config = NSConfig(
        nu=args.nu,
        dx=1.0,
        dy=1.0,
        momentum_weight=args.momentum_weight,
        divergence_weight=args.divergence_weight,
    )

    print("PC config:", pc_config)
    print("NS config:", ns_config)
    print("Hidden grid shape for MNIST will be [B, 3, 28, 28].")
    print("Physics channels: 0=u_x, 1=u_y, 2=pressure.")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pc_config=pc_config,
            ns_config=ns_config,
            amort_weight=args.amort_weight,
            ns_init_weight=args.ns_init_weight,
            epoch=epoch,
            debug_grid=args.debug_grid,
        )

        acc_fast = evaluate_without_pc_inference(model, test_loader, device)
        acc_pc = evaluate_with_pc_inference(model, test_loader, device, pc_config, ns_config)
        print(
            f"epoch={epoch:02d} test_acc_no_inner_loop={acc_fast:.2f}% "
            f"test_acc_with_pc_inference={acc_pc:.2f}%"
        )


if __name__ == "__main__":
    main()
