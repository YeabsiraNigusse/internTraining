#!/usr/bin/env python3
"""
fluid_pc_mnist_paper_like.py

Educational "paper-like" Fluid Predictive Coding MNIST implementation.

This is NOT meant to be a state-of-the-art MNIST classifier.
It is meant to show how the ideas fit together in code:

    1. Hidden state is an activation density a on a 2D grid.
    2. Predictive-coding reaction step reduces local prediction/reconstruction errors.
    3. A learned stream/value head creates a velocity field u.
    4. Helmholtz-Hodge / Leray projection makes u approximately divergence-free.
    5. Conservative advection moves activation mass using u.
    6. A readout predicts the MNIST digit from the final hidden density.

Main loop inside each inference step:

    PC reaction:
        a <- a - eta * dE_PC/da

    value/control:
        psi = stream_net([a, x])
        u = curl(psi)

    projection:
        u <- project_div_free(u)

    advection:
        a <- conservative_advect(a, u)

    normalize:
        a <- positive density with sum 1

Run:

    python fluid_pc_mnist_paper_like.py --epochs 3 --train_subset 10000 --test_subset 2000

Quick debug:

    python fluid_pc_mnist_paper_like.py --epochs 1 --train_subset 1024 --test_subset 256 --batch_size 64 --inner_steps 4 --print_every 20

Smoke test without MNIST download:

    python fluid_pc_mnist_paper_like.py --smoke_test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_density(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Make activation nonnegative and mass-normalized.

    a shape: [B, 1, H, W]

    After this:
        a >= 0
        a.sum over H,W = 1 for each sample

    This represents the paper's "activation budget".
    """
    a = F.softplus(a)
    mass = a.sum(dim=(1, 2, 3), keepdim=True)
    return a / (mass + eps)


def image_to_density(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Initialize hidden activation density from input image.

    x shape: [B, 1, 28, 28]

    This is the simplest paper-like initialization:
        the image provides the initial activation distribution.
    """
    a = x.clamp_min(0.0) + eps
    return a / a.sum(dim=(1, 2, 3), keepdim=True)


def roll_x(z: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(z, shifts=shift, dims=-1)


def roll_y(z: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(z, shifts=shift, dims=-2)


def ddx_periodic(z: torch.Tensor) -> torch.Tensor:
    """Central difference in x direction with periodic boundary."""
    return 0.5 * (roll_x(z, -1) - roll_x(z, 1))


def ddy_periodic(z: torch.Tensor) -> torch.Tensor:
    """Central difference in y direction with periodic boundary."""
    return 0.5 * (roll_y(z, -1) - roll_y(z, 1))


def laplacian_periodic(z: torch.Tensor) -> torch.Tensor:
    """5-point periodic Laplacian."""
    return roll_x(z, -1) + roll_x(z, 1) + roll_y(z, -1) + roll_y(z, 1) - 4.0 * z


def divergence_periodic(u: torch.Tensor) -> torch.Tensor:
    """
    Divergence of velocity field.

    u shape: [B, 2, H, W]
        u[:, 0] = u_x
        u[:, 1] = u_y

    returns: [B, 1, H, W]
    """
    ux = u[:, 0:1]
    uy = u[:, 1:2]
    return ddx_periodic(ux) + ddy_periodic(uy)


def velocity_from_stream_function(psi: torch.Tensor) -> torch.Tensor:
    """
    Build a divergence-free velocity from a stream function.

    psi shape: [B, 1, H, W]

    In 2D:
        u_x =  d psi / dy
        u_y = -d psi / dx
    """
    ux = ddy_periodic(psi)
    uy = -ddx_periodic(psi)
    return torch.cat([ux, uy], dim=1)


def project_div_free_fft(u: torch.Tensor) -> torch.Tensor:
    """
    Helmholtz-Hodge / Leray projection on a periodic 2D grid.

    This removes the divergent component of u in Fourier space.

    This is the grid version of:
        u = g - grad(p)
        Delta p = div(g)
    """
    batch, channels, height, width = u.shape
    assert channels == 2

    ux = u[:, 0]
    uy = u[:, 1]

    ux_hat = torch.fft.fftn(ux, dim=(-2, -1))
    uy_hat = torch.fft.fftn(uy, dim=(-2, -1))

    ky = torch.fft.fftfreq(height, device=u.device, dtype=u.dtype) * (2.0 * torch.pi)
    kx = torch.fft.fftfreq(width, device=u.device, dtype=u.dtype) * (2.0 * torch.pi)

    ky = ky.view(1, height, 1)
    kx = kx.view(1, 1, width)

    denom = kx.pow(2) + ky.pow(2)
    denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))

    # Fourier projection:
    # u_hat <- u_hat - k * (k dot u_hat) / |k|^2
    k_dot_u = kx * ux_hat + ky * uy_hat

    ux_hat_proj = ux_hat - kx * k_dot_u / denom_safe
    uy_hat_proj = uy_hat - ky * k_dot_u / denom_safe

    # Leave zero-frequency mode unchanged.
    zero = denom == 0
    ux_hat_proj = torch.where(zero, ux_hat, ux_hat_proj)
    uy_hat_proj = torch.where(zero, uy_hat, uy_hat_proj)

    ux_proj = torch.fft.ifftn(ux_hat_proj, dim=(-2, -1)).real
    uy_proj = torch.fft.ifftn(uy_hat_proj, dim=(-2, -1)).real

    return torch.stack([ux_proj, uy_proj], dim=1)


def scale_to_cfl(
    u: torch.Tensor,
    dt: float,
    target_cfl: float,
    max_scale: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rescale velocity so maximum displacement per step is near target_cfl.

    CFL = dt * max(abs(u))
    """
    with torch.no_grad():
        max_speed = u.abs().amax(dim=(1, 2, 3), keepdim=True)
        scale = target_cfl / (dt * max_speed + 1e-8)
        scale = scale.clamp(max=max_scale)
    u_scaled = u * scale
    cfl = dt * u_scaled.abs().amax(dim=(1, 2, 3))
    return u_scaled, cfl


def conservative_advect_periodic(a: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Conservative finite-volume advection with periodic boundary.

    a shape: [B, 1, H, W]
    u shape: [B, 2, H, W]

    The update is:
        a_new = a - dt * div(flux)

    Flux through a face is:
        flux = face_velocity * donor_cell_density

    This approximately implements:
        partial_t a + div(a u) = 0
    """
    ux = u[:, 0:1]
    uy = u[:, 1:2]

    # Velocity at right face between current cell and right neighbor.
    ux_right = 0.5 * (ux + roll_x(ux, -1))
    a_right = roll_x(a, -1)
    flux_right = torch.where(ux_right >= 0, ux_right * a, ux_right * a_right)
    flux_left = roll_x(flux_right, 1)

    # Velocity at down face between current cell and lower neighbor.
    uy_down = 0.5 * (uy + roll_y(uy, -1))
    a_down = roll_y(a, -1)
    flux_down = torch.where(uy_down >= 0, uy_down * a, uy_down * a_down)
    flux_up = roll_y(flux_down, 1)

    div_flux = (flux_right - flux_left) + (flux_down - flux_up)
    return a - dt * div_flux


@dataclass
class InnerConfig:
    inner_steps: int = 5
    reaction_lr: float = 0.25
    dt: float = 0.5
    target_cfl: float = 0.35
    diffusion: float = 0.002
    pc_pred_weight: float = 0.20
    pc_recon_weight: float = 0.20
    pc_entropy_weight: float = 0.00
    differentiable_inner: bool = True
    use_projection: bool = True


class FluidPCMNIST(nn.Module):
    """
    One-hidden-layer paper-like Fluid Predictive Coding model.

    Hidden state:
        a : [B, 1, 28, 28]
        This is activation density / conserved hidden mass.

    Velocity:
        u : [B, 2, 28, 28]
        This moves a during advection.

    Pressure:
        Not a learned channel here. It is implicit in the Hodge projection:
            u = g - grad(p)
    """

    def __init__(self, img_size: int = 28):
        super().__init__()
        self.img_size = img_size
        n = img_size * img_size

        # Local predictor for PC reaction:
        # predicts what hidden density should be from local neighborhood + input image.
        self.local_predictor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # Decoder: hidden density -> reconstructed image.
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # Readout: final hidden density -> digit logits.
        self.readout = nn.Linear(n, 10)

        # Stream/value head:
        # produces a stream function psi. The curl of psi is velocity.
        self.stream_head = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def classify(self, a: torch.Tensor) -> torch.Tensor:
        return self.readout(a.flatten(start_dim=1))

    def reconstruct(self, a: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(a))

    def predict_density(self, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raw = self.local_predictor(torch.cat([a, x], dim=1))
        return normalize_density(raw)

    def make_velocity(self, a: torch.Tensor, x: torch.Tensor, config: InnerConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        psi = self.stream_head(torch.cat([a, x], dim=1))
        u = velocity_from_stream_function(psi)

        if config.use_projection:
            # Projection is used as a forward-pass safety correction.
            # We use a straight-through estimator so training does not need
            # expensive second-order gradients through the FFT Poisson solve.
            with torch.no_grad():
                u_projected = project_div_free_fft(u)
            u = u + (u_projected - u).detach()

        u, cfl = scale_to_cfl(u, dt=config.dt, target_cfl=config.target_cfl)

        div = divergence_periodic(u)
        stats = {
            "div_abs": div.abs().mean().detach(),
            "div_rms": torch.sqrt(div.pow(2).mean()).detach(),
            "cfl": cfl.mean().detach(),
            "speed": u.pow(2).sum(dim=1).sqrt().mean().detach(),
        }
        return u, stats

    def pc_reaction_energy(self, a: torch.Tensor, x: torch.Tensor, config: InnerConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        PC reaction energy.

        This is the prediction error part.

        It contains:
            1. local hidden prediction error: a - a_hat
            2. image reconstruction error: x - x_hat
            3. optional entropy term for density sharpness
        """
        a_hat = self.predict_density(a, x)
        x_hat = self.reconstruct(a)

        pred_loss = F.mse_loss(a, a_hat)
        recon_loss = F.mse_loss(x_hat, x)

        entropy = -(a * (a + 1e-8).log()).sum(dim=(1, 2, 3)).mean()

        energy = (
            config.pc_pred_weight * pred_loss
            + config.pc_recon_weight * recon_loss
            + config.pc_entropy_weight * entropy
        )

        stats = {
            "pc_pred": pred_loss.detach(),
            "pc_recon": recon_loss.detach(),
            "pc_entropy": entropy.detach(),
            "pc_energy": energy.detach(),
        }
        return energy, stats

    def infer(self, x: torch.Tensor, config: InnerConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Paper-like inner inference loop.

        x shape: [B, 1, 28, 28]

        Returns:
            final density a
            stats dictionary
        """
        # Initial hidden density from input image.
        a = image_to_density(x)

        stats_accum = {
            "pc_pred": [],
            "pc_recon": [],
            "pc_energy": [],
            "div_abs": [],
            "div_rms": [],
            "cfl": [],
            "speed": [],
            "mass_error": [],
        }

        for _ in range(config.inner_steps):
            # 1. PC reaction step: update hidden density by prediction error gradient.
            a = a.requires_grad_(True)
            pc_energy, pc_stats = self.pc_reaction_energy(a, x, config)

            # Straight-through PC reaction gradient.
            # We do not build second-order graphs through this hidden-state
            # gradient, which keeps the example fast enough for MNIST.
            # The later advection/readout path is still differentiable.
            grad_a = torch.autograd.grad(
                pc_energy,
                a,
                create_graph=False,
                retain_graph=False,
            )[0]

            a = a - config.reaction_lr * grad_a
            a = normalize_density(a)

            # 2. Value/stream -> velocity.
            u, vel_stats = self.make_velocity(a, x, config)

            # 3. Conservative advection of hidden activation mass.
            a = conservative_advect_periodic(a, u, dt=config.dt)

            # 4. Optional diffusion smoothing.
            if config.diffusion > 0:
                a = a + config.dt * config.diffusion * laplacian_periodic(a)

            # 5. Keep a valid density.
            a = normalize_density(a)

            if not config.differentiable_inner:
                a = a.detach()

            with torch.no_grad():
                mass = a.sum(dim=(1, 2, 3))
                stats_accum["mass_error"].append((mass - 1.0).abs().mean().detach())

                for k, v in pc_stats.items():
                    if k in stats_accum:
                        stats_accum[k].append(v.detach())

                for k, v in vel_stats.items():
                    if k in stats_accum:
                        stats_accum[k].append(v.detach())

        stats = {}
        for k, values in stats_accum.items():
            stats[k] = torch.stack(values).mean().item() if values else 0.0
        return a, stats


def make_loaders(args: argparse.Namespace):
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    if args.train_subset > 0:
        train_data = Subset(train_data, list(range(args.train_subset)))
    if args.test_subset > 0:
        test_data = Subset(test_data, list(range(args.test_subset)))

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, config, epoch, args):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_recon = 0.0
    total_correct = 0
    total_seen = 0
    stat_sums: Dict[str, float] = {}

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Inner PC + fluid inference.
        a, stats = model.infer(images, config)

        logits = model.classify(a)
        x_hat = model.reconstruct(a)

        ce_loss = F.cross_entropy(logits, labels)
        recon_loss = F.mse_loss(x_hat, images)
        loss = ce_loss + args.outer_recon_weight * recon_loss

        loss.backward()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)

            total_loss += loss.item() * bs
            total_ce += ce_loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_correct += correct
            total_seen += bs

            for k, v in stats.items():
                stat_sums[k] = stat_sums.get(k, 0.0) + v * bs

        if args.print_every > 0 and batch_idx % args.print_every == 0:
            acc = 100.0 * correct / bs
            print(
                f"epoch={epoch:02d} batch={batch_idx:04d} "
                f"loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                f"recon={recon_loss.item():.4f} acc={acc:5.1f}% "
                f"div_rms={stats.get('div_rms', 0.0):.3e} "
                f"mass_err={stats.get('mass_error', 0.0):.3e} "
                f"cfl={stats.get('cfl', 0.0):.3f}"
            )

    out = {
        "loss": total_loss / total_seen,
        "ce": total_ce / total_seen,
        "recon": total_recon / total_seen,
        "acc": 100.0 * total_correct / total_seen,
    }
    for k, v in stat_sums.items():
        out[k] = v / total_seen
    return out


def evaluate(model, loader, device, config):
    model.eval()

    eval_config = InnerConfig(**vars(config))
    eval_config.differentiable_inner = False

    total_loss = 0.0
    total_recon = 0.0
    total_correct = 0
    total_seen = 0
    stat_sums: Dict[str, float] = {}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # We need gradients with respect to a inside inference, even in eval.
        with torch.enable_grad():
            a, stats = model.infer(images, eval_config)

        with torch.no_grad():
            logits = model.classify(a)
            x_hat = model.reconstruct(a)

            ce_loss = F.cross_entropy(logits, labels)
            recon_loss = F.mse_loss(x_hat, images)

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)

            total_loss += ce_loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_correct += correct
            total_seen += bs

            for k, v in stats.items():
                stat_sums[k] = stat_sums.get(k, 0.0) + v * bs

    out = {
        "loss": total_loss / total_seen,
        "recon": total_recon / total_seen,
        "acc": 100.0 * total_correct / total_seen,
    }
    for k, v in stat_sums.items():
        out[k] = v / total_seen
    return out


def show_predictions(model, loader, device, config, count=10):
    model.eval()

    eval_config = InnerConfig(**vars(config))
    eval_config.differentiable_inner = False

    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.enable_grad():
        a, _ = model.infer(images, eval_config)

    with torch.no_grad():
        logits = model.classify(a)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    print("=" * 72)
    print("Sample predictions")
    print("=" * 72)

    for i in range(min(count, images.size(0))):
        pred = preds[i].item()
        truth = labels[i].item()
        conf = probs[i, pred].item()
        mark = "OK" if pred == truth else "WRONG"
        print(f"sample={i:02d} pred={pred} truth={truth} confidence={conf:.3f} {mark}")


def smoke_test(device: torch.device):
    print("Running smoke test with random MNIST-shaped data...")

    model = FluidPCMNIST().to(device)
    config = InnerConfig(inner_steps=2, differentiable_inner=True)

    x = torch.rand(8, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    a, stats = model.infer(x, config)
    logits = model.classify(a)
    x_hat = model.reconstruct(a)

    loss = F.cross_entropy(logits, y) + F.mse_loss(x_hat, x)
    loss.backward()

    print("x shape      :", tuple(x.shape))
    print("a shape      :", tuple(a.shape))
    print("logits shape :", tuple(logits.shape))
    print("x_hat shape  :", tuple(x_hat.shape))
    print("mass per item:", a.sum(dim=(1, 2, 3)).detach().cpu())
    print("stats        :", stats)
    print("Smoke test passed.")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--train_subset", type=int, default=10000, help="0 means full train set")
    p.add_argument("--test_subset", type=int, default=2000, help="0 means full test set")

    # Inner loop settings
    p.add_argument("--inner_steps", type=int, default=5)
    p.add_argument("--reaction_lr", type=float, default=0.25)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--target_cfl", type=float, default=0.35)
    p.add_argument("--diffusion", type=float, default=0.002)
    p.add_argument("--pc_pred_weight", type=float, default=0.20)
    p.add_argument("--pc_recon_weight", type=float, default=0.20)
    p.add_argument("--pc_entropy_weight", type=float, default=0.00)

    p.add_argument("--no_projection", action="store_true")
    p.add_argument("--no_differentiable_inner", action="store_true")

    # Outer training settings
    p.add_argument("--outer_recon_weight", type=float, default=0.10)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--smoke_test", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("Paper-like Fluid Predictive Coding MNIST")
    print("=" * 72)
    print(f"Device: {device}")

    config = InnerConfig(
        inner_steps=args.inner_steps,
        reaction_lr=args.reaction_lr,
        dt=args.dt,
        target_cfl=args.target_cfl,
        diffusion=args.diffusion,
        pc_pred_weight=args.pc_pred_weight,
        pc_recon_weight=args.pc_recon_weight,
        pc_entropy_weight=args.pc_entropy_weight,
        differentiable_inner=(not args.no_differentiable_inner),
        use_projection=(not args.no_projection),
    )

    print("Inner loop config:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")

    model = FluidPCMNIST().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {total_params:,}")

    if args.smoke_test:
        smoke_test(device)
        return

    train_loader, test_loader = make_loaders(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, epoch, args)
        test_metrics = evaluate(model, test_loader, device, config)

        best_acc = max(best_acc, test_metrics["acc"])

        print("=" * 72)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Train acc {train_metrics['acc']:.2f}% | "
            f"Test loss {test_metrics['loss']:.4f} | "
            f"Test acc {test_metrics['acc']:.2f}% | "
            f"Best {best_acc:.2f}%"
        )
        print(
            f"Diagnostics | "
            f"train div_rms {train_metrics.get('div_rms', 0.0):.3e} | "
            f"test div_rms {test_metrics.get('div_rms', 0.0):.3e} | "
            f"train mass_err {train_metrics.get('mass_error', 0.0):.3e} | "
            f"test mass_err {test_metrics.get('mass_error', 0.0):.3e} | "
            f"train CFL {train_metrics.get('cfl', 0.0):.3f} | "
            f"test CFL {test_metrics.get('cfl', 0.0):.3f}"
        )
        print("=" * 72)

    show_predictions(model, test_loader, device, config, count=10)


if __name__ == "__main__":
    main()
