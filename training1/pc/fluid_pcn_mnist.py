"""
fluid_pcn_mnist.py

A practical, discriminative Fluid-PCN experiment for MNIST.

This is not an exact solver of the HJB equation on SDiff(M).  It implements the
paper's practical recipe:

1. One conserved PC hidden state z1 on a 28x28 periodic manifold, constrained
   to be the physical activation density used by the fluid solver.
2. A two-hidden-layer generative predictive-coding chain z2 -> z1 -> x, with
   local latent inference and manual local PC parameter learning.
3. A learned value-like cost-to-go map supervised by the task.
4. A learned stream function psi, giving a divergence-free velocity
       u = (d psi / dy, -d psi / dx).
5. CFL-targeted conservative upwind advection.
6. Diffusion warmup followed by annealing to zero.
7. Nonlocal credit through a distance-to-goal shaping objective and a short
   receding-horizon rollout.
8. A real routing bottleneck: each class is represented by a small target region,
   and the classifier reads only the mass delivered to those ten regions.
9. No backpropagation through the long PC inference loop.  PC parameters are
   updated manually from local errors; value and flow parameters keep their
   existing fixed-state objectives after inference.

Run a smoke test:
    python fluid_pcn_mnist.py --smoke_test

Small experiment:
    python fluid_pcn_mnist.py --epochs 1 --train_subset 2000 --test_subset 500

Ablation without fluid transport:
    python fluid_pcn_mnist.py --epochs 1 --train_subset 2000 --test_subset 500 --disable_flow
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG = 28
PIXELS = IMG * IMG
NUM_CLASSES = 10
EPS = 1e-8


@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 5
    train_subset: int = 40000
    test_subset: int = 8000

    # PC inference
    infer_steps: int = 5
    hidden_lr: float = 0.01
    z2_channels: int = 8
    pc_weight_decay: float = 1e-4
    pc_latent_clip: float = 1.0

    # Short receding-horizon controller training
    control_horizon: int = 5

    # Fluid numerics
    dt: float = 0.5
    target_cfl: float = 0.35
    max_cfl_scale: float = 50.0
    diffusion_start: float = 0.015
    diffusion_anneal_fraction: float = 0.5

    # Separate local learning rates
    pc_lr: float = 2e-3
    value_lr: float = 2e-3
    flow_lr: float = 2e-3

    # Composite free-energy weights
    pc_weight: float = 1.0
    value_weight: float = 1.0
    class_weight: float = 1.0
    shape_weight: float = 2.0
    control_weight: float = 1e-3
    divergence_weight: float = 1e-2

    region_logit_scale: float = 40.0
    disable_flow: bool = False


# -----------------------------------------------------------------------------
# Density and periodic-grid operators
# -----------------------------------------------------------------------------


def normalize_density(a: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Project a nonnegative tensor to unit mass per sample.

    This is used after the PC reaction, which is not inherently conservative.
    It is not an incompressibility operation; incompressibility applies to u.
    """

    a = a.clamp_min(eps)
    mass = a.sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return a / mass


def roll_x(z: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(z, shifts=shift, dims=-1)


def roll_y(z: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(z, shifts=shift, dims=-2)


def ddx(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (roll_x(z, -1) - roll_x(z, 1))


def ddy(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (roll_y(z, -1) - roll_y(z, 1))


def laplacian(z: torch.Tensor) -> torch.Tensor:
    """Five-point periodic Laplacian with unit grid spacing."""

    return (
        roll_x(z, -1)
        + roll_x(z, 1)
        + roll_y(z, -1)
        + roll_y(z, 1)
        - 4.0 * z
    )


def velocity_from_stream(psi: torch.Tensor) -> torch.Tensor:
    """Construct a 2-D divergence-free velocity from a stream function.

    u_x = d psi / dy
    u_y = -d psi / dx
    """

    ux = ddy(psi)
    uy = -ddx(psi)
    return torch.cat([ux, uy], dim=1)


def divergence(u: torch.Tensor) -> torch.Tensor:
    ux = u[:, 0:1]
    uy = u[:, 1:2]
    return ddx(ux) + ddy(uy)


def cfl_number(u: torch.Tensor, dt: float) -> torch.Tensor:
    speed = torch.sqrt(u[:, 0:1].square() + u[:, 1:2].square() + EPS)
    return dt * speed.amax(dim=(1, 2, 3))


def scale_velocity_to_cfl(
    u: torch.Tensor,
    dt: float,
    target_cfl: float,
    max_scale: float,
) -> torch.Tensor:
    """Rescale each sample toward a target Courant number."""

    speed = torch.sqrt(u[:, 0:1].square() + u[:, 1:2].square() + EPS)
    max_speed = speed.amax(dim=(1, 2, 3), keepdim=True)
    scale = target_cfl / (dt * max_speed + EPS)
    return u * scale.clamp(max=max_scale)


def advect_upwind(a: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """Conservative donor-cell update for d_t a + div(a u) = 0.

    The periodic flux differences telescope, so total mass is conserved up to
    floating-point roundoff.  With a suitable CFL, the update is positivity
    preserving.
    """

    ux = u[:, 0:1]
    uy = u[:, 1:2]

    ux_right = 0.5 * (ux + roll_x(ux, -1))
    a_right = roll_x(a, -1)
    flux_right = torch.where(ux_right >= 0, ux_right * a, ux_right * a_right)
    flux_left = roll_x(flux_right, 1)

    uy_down = 0.5 * (uy + roll_y(uy, -1))
    a_down = roll_y(a, -1)
    flux_down = torch.where(uy_down >= 0, uy_down * a, uy_down * a_down)
    flux_up = roll_y(flux_down, 1)

    div_flux = (flux_right - flux_left) + (flux_down - flux_up)
    return a - dt * div_flux


def diffusion_at_step(step: int, total_steps: int, cfg: Config) -> float:
    """Exploration-to-sharpening schedule kappa_0 -> 0."""

    anneal_steps = max(1, int(math.ceil(total_steps * cfg.diffusion_anneal_fraction)))
    if step >= anneal_steps:
        return 0.0
    return cfg.diffusion_start * (1.0 - step / anneal_steps)


def transport_step(
    a: torch.Tensor,
    u: torch.Tensor,
    step: int,
    total_steps: int,
    cfg: Config,
) -> torch.Tensor:
    a_next = advect_upwind(a, u, cfg.dt)
    kappa = diffusion_at_step(step, total_steps, cfg)
    if kappa > 0.0:
        a_next = a_next + cfg.dt * kappa * laplacian(a_next)
    return a_next


# -----------------------------------------------------------------------------
# Routing geometry: one target region per class
# -----------------------------------------------------------------------------


def make_routing_geometry() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return source mask, ten target masks, and normalized distance maps.

    Targets lie on a ring around the central source.  Classification therefore
    requires moving conserved mass from the source region to one target region.
    """

    source = torch.zeros(1, 1, IMG, IMG)
    source_size = 6
    y0 = IMG // 2 - source_size // 2
    x0 = IMG // 2 - source_size // 2
    source[:, :, y0 : y0 + source_size, x0 : x0 + source_size] = 1.0

    target_masks = torch.zeros(NUM_CLASSES, 1, IMG, IMG)
    center = (IMG - 1) / 2.0
    radius = 8.0
    target_centers = []
    for k in range(NUM_CLASSES):
        angle = 2.0 * math.pi * k / NUM_CLASSES - math.pi / 2.0
        cy = int(round(center + radius * math.sin(angle)))
        cx = int(round(center + radius * math.cos(angle)))
        target_centers.append((cy, cx))
        target_masks[k, 0, cy - 1 : cy + 2, cx - 1 : cx + 2] = 1.0

    yy = torch.arange(IMG, dtype=torch.float32).view(IMG, 1)
    xx = torch.arange(IMG, dtype=torch.float32).view(1, IMG)
    distance_maps = torch.empty(NUM_CLASSES, 1, IMG, IMG)
    for k, (cy, cx) in enumerate(target_centers):
        # Distance to the 3x3 target patch, not merely its center.
        dy = (yy - float(cy)).abs().sub(1.0).clamp_min(0.0)
        dx = (xx - float(cx)).abs().sub(1.0).clamp_min(0.0)
        dist = torch.sqrt(dx.square() + dy.square())
        distance_maps[k, 0] = dist / dist.max().clamp_min(EPS)

    return source, target_masks, distance_maps


SOURCE_MASK, TARGET_MASKS, DISTANCE_MAPS = make_routing_geometry()


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class FluidPCN(nn.Module):
    """One conserved hidden density with local PC and learned fluid control."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Generative PC predictors, top-down:
        #   W0: z1 -> z0=x      with sigmoid output for MNIST pixels
        #   W1: z2 -> z1        with tanh hidden prediction
        # PC learning updates these parameters manually from local errors, so
        # they are deliberately excluded from autograd optimizers.
        self.pc_to_image = nn.Conv2d(
            1,
            1,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        self.pc_to_density = nn.Conv2d(
            cfg.z2_channels,
            1,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=True,
        )
        with torch.no_grad():
            nn.init.normal_(self.pc_to_image.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.pc_to_image.bias)
            nn.init.normal_(self.pc_to_density.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.pc_to_density.bias)
        for module in (self.pc_to_image, self.pc_to_density):
            for param in module.parameters():
                param.requires_grad_(False)

        # Value-like cost-to-go map.  During training it learns the distance map
        # to the task-relevant class region.  This is a practical finite-grid
        # surrogate for the paper's abstract HJB value function.
        self.value_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), # [B,1,28,28]→[B,16,28,28].
            nn.GELU(),
            nn.MaxPool2d(2), # reducing the spatial size by 2:
            nn.Conv2d(16, 32, 3, padding=1), # gives [B,32,14,14].
            nn.GELU(),
            nn.MaxPool2d(2), # pool again
            nn.Flatten(), # [B,32,7,7]→[B,1568].
            nn.Linear(32 * 7 * 7, 256),
            nn.GELU(),
            nn.Linear(256, PIXELS), # 1568→256→784.
        )

        # Stream-function controller.  It receives current mass, the image, and
        # the learned value map.  Curling psi places u in the solenoidal
        # (divergence-free) subspace by construction.
        self.stream_controller = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(24, 1, 3, padding=1, padding_mode="circular"),
        )

        self.register_buffer("source_mask", SOURCE_MASK.clone())
        self.register_buffer("target_masks", TARGET_MASKS.clone())
        self.register_buffer("distance_maps", DISTANCE_MAPS.clone())

    def random_density(self, x: torch.Tensor) -> torch.Tensor:
        """Random central unit-mass hidden state, independent of image content."""

        # x is used only for batch/device/dtype/shape; no pixel values seed z1.
        raw = (torch.rand_like(x) + EPS) * self.source_mask
        return normalize_density(raw)

    def random_top_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Small random z2 latent, independent of image content."""

        return 0.01 * torch.randn(
            x.shape[0],
            self.cfg.z2_channels,
            IMG,
            IMG,
            device=x.device,
            dtype=x.dtype,
        )

    def pc_predictions(self, z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Top-down generative predictions: z2 -> z1 -> z0.

        Final equations:
          r0 = W0 * z1 + b0,  mu0 = sigmoid(r0)
          r1 = W1 * z2 + b1,  mu1 = tanh(r1)
        x is intentionally absent here; it is only the clamped target z0 used
        later when computing eps0.
        """

        r0 = self.pc_to_image(z1)
        mu0 = torch.sigmoid(r0)
        r1 = self.pc_to_density(z2)
        mu1 = torch.tanh(r1)
        return {"r0": r0, "mu0": mu0, "r1": r1, "mu1": mu1}

    def value_map(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.value_encoder(x).view(-1, 1, IMG, IMG)
        return torch.sigmoid(raw)

    # learned feature might respond to:
        # high density + decreasing value toward the right.

    # Another might respond to:
        # image stroke + density edge.

    # Another might respond to:
        # density sitting in a high-cost area.

    # We cannot say exactly what they learn without inspecting trained weights/activations,
    # but mathematically that's what they're able to represent.

    def velocity(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        value_map: torch.Tensor,
    ) -> torch.Tensor:
        # It generates one scalar field ψ, then converts it into an incompressible velocity.
        # That is the substitute for explicit pressure projection in this implementation.
        
        psi = self.stream_controller(torch.cat([a, x, value_map], dim=1)) # concatenates across the channel dimension.
        # Given where mass currently is, 
            # which image I'm looking at, and where the low-cost destination appears to be, 
            # what flow field should I generate?
            
        u = velocity_from_stream(psi)
        return scale_velocity_to_cfl(
            u,
            dt=self.cfg.dt,
            target_cfl=self.cfg.target_cfl,
            max_scale=self.cfg.max_cfl_scale,
        )

    def region_logits(self, a: torch.Tensor) -> torch.Tensor:
        masses = (a.unsqueeze(1) * self.target_masks.unsqueeze(0)).sum(dim=(2, 3, 4))
        return self.cfg.region_logit_scale * masses

    def target_distance(self, labels: torch.Tensor) -> torch.Tensor:
        return self.distance_maps[labels]


# -----------------------------------------------------------------------------
# Local generative-PC operators
# -----------------------------------------------------------------------------


def sigmoid_prime_from_mu(mu: torch.Tensor) -> torch.Tensor:
    return mu * (1.0 - mu)


def tanh_prime_from_mu(mu: torch.Tensor) -> torch.Tensor:
    return 1.0 - mu.square()


def pc_errors(
    x: torch.Tensor,
    z1: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """README convention: eps_k = target - prediction."""

    return {
        "eps0": x - predictions["mu0"],
        "eps1": z1 - predictions["mu1"],
    }


def pc_error_derivatives(
    errors: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Local child error times that layer's activation derivative."""

    d0 = errors["eps0"] * sigmoid_prime_from_mu(predictions["mu0"])
    d1 = errors["eps1"] * tanh_prime_from_mu(predictions["mu1"])
    return d0, d1


def circular_conv2d(
    parent: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    pad_h = weight.shape[-2] // 2
    pad_w = weight.shape[-1] // 2
    padded = F.pad(parent, (pad_w, pad_w, pad_h, pad_h), mode="circular")
    return F.conv2d(padded, weight, bias=bias, stride=1, padding=0)


def circular_conv_transpose_local(delta: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Adjoint of the same-padding circular Conv2d used by the PC predictors."""

    pad_h = weight.shape[-2] // 2
    pad_w = weight.shape[-1] // 2
    flipped = torch.flip(weight, dims=(-2, -1)).transpose(0, 1)
    padded = F.pad(delta, (pad_w, pad_w, pad_h, pad_h), mode="circular")
    return F.conv2d(padded, flipped, bias=None, stride=1, padding=0)


def project_physical_density(z1: torch.Tensor) -> torch.Tensor:
    """Constrain z1 to the fluid solver's nonnegative unit-mass density.

    This projection is an added physical constraint on top of the ordinary PC
    latent update.  It is used inside no-grad inference and is not a learned
    autograd operation.
    """

    return normalize_density(z1)


def clip_latent_delta(delta: torch.Tensor, max_abs: float) -> torch.Tensor:
    if max_abs <= 0.0:
        return delta
    return delta.clamp(min=-max_abs, max=max_abs)


def pc_latent_updates(
    model: FluidPCN,
    x: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One local PC reaction step with fixed weights.

    Implemented equations:
      eps0 = x - mu0
      eps1 = z1 - mu1
      delta_z1 = transpose(W0, eps0 * sigmoid_prime(r0)) - eps1
      delta_z2 = transpose(W1, eps1 * tanh_prime(r1))
    """

    predictions = model.pc_predictions(z1, z2)
    errors = pc_errors(x, z1, predictions)
    d0, d1 = pc_error_derivatives(errors, predictions)

    bottom_up_0 = circular_conv_transpose_local(d0, model.pc_to_image.weight)
    bottom_up_1 = circular_conv_transpose_local(d1, model.pc_to_density.weight)
    delta_z1 = clip_latent_delta(bottom_up_0 - errors["eps1"], cfg.pc_latent_clip)
    delta_z2 = clip_latent_delta(bottom_up_1, cfg.pc_latent_clip)

    z1 = project_physical_density(z1 + cfg.hidden_lr * delta_z1)
    z2 = z2 + cfg.hidden_lr * delta_z2
    return z1, z2


def pc_diagnostics_from_errors(errors: Dict[str, torch.Tensor]) -> Dict[str, float]:
    pc_recon_loss = 0.5 * errors["eps0"].square().mean()
    pc_hidden_loss = 0.5 * errors["eps1"].square().mean()
    return {
        "pc_recon_loss": pc_recon_loss.item(),
        "pc_hidden_loss": pc_hidden_loss.item(),
        "pc_energy": (pc_recon_loss + pc_hidden_loss).item(),
    }


def pc_state_diagnostics(
    model: FluidPCN,
    x: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> Dict[str, float]:
    predictions = model.pc_predictions(z1, z2)
    errors = pc_errors(x, z1, predictions)
    return pc_diagnostics_from_errors(errors)


def conv_local_weight_grads(
    parent: torch.Tensor,
    child_delta: torch.Tensor,
    weight_shape: torch.Size,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Local convolutional correlation: parent patches times child error."""

    kernel_h, kernel_w = weight_shape[-2], weight_shape[-1]
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    padded = F.pad(parent, (pad_w, pad_w, pad_h, pad_h), mode="circular")
    patches = F.unfold(padded, kernel_size=(kernel_h, kernel_w))
    delta_flat = child_delta.flatten(2)
    batch = float(parent.shape[0])

    grad_w = torch.einsum("bol,bil->oi", delta_flat, patches) / batch
    grad_w = grad_w.view(weight_shape)
    grad_b = delta_flat.sum(dim=(0, 2)) / batch
    return grad_w, grad_b


@torch.no_grad()
def pc_local_weight_updates(
    model: FluidPCN,
    x: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    cfg: Config,
) -> Dict[str, float]:
    """Manual local PC learning; no backward/autograd is used here.

    Implemented equations:
      dW0 = correlation(z1, eps0 * sigmoid_prime(r0)) / batch_size
      dW1 = correlation(z2, eps1 * tanh_prime(r1)) / batch_size
      Wk += pc_lr * (dWk - pc_weight_decay * Wk)
      bk += pc_lr * dbk
    """

    predictions = model.pc_predictions(z1, z2)
    errors = pc_errors(x, z1, predictions)
    d0, d1 = pc_error_derivatives(errors, predictions)

    dW0, db0 = conv_local_weight_grads(z1, d0, model.pc_to_image.weight.shape)
    dW1, db1 = conv_local_weight_grads(z2, d1, model.pc_to_density.weight.shape)

    model.pc_to_image.weight.add_(cfg.pc_lr * (dW0 - cfg.pc_weight_decay * model.pc_to_image.weight))
    model.pc_to_image.bias.add_(cfg.pc_lr * db0)
    model.pc_to_density.weight.add_(cfg.pc_lr * (dW1 - cfg.pc_weight_decay * model.pc_to_density.weight))
    model.pc_to_density.bias.add_(cfg.pc_lr * db1)

    diagnostics = pc_diagnostics_from_errors(errors)
    diagnostics.update(
        {
            "pc_dW0_norm": dW0.norm().item(),
            "pc_dW1_norm": dW1.norm().item(),
        }
    )
    return diagnostics


def validate_manual_pc_gradients(device: torch.device, dtype: torch.dtype = torch.float32) -> None:
    """Tiny isolated autograd check for the manual local conv update.

    Autograd is used only in this smoke-test helper to verify that the explicit
    local correlation equals the negative gradient of 0.5||target - mu||^2.
    Actual PC inference and PC parameter learning never call autograd.
    """

    parent = torch.randn(2, 2, 5, 5, device=device, dtype=dtype)
    target = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)
    weight = (0.02 * torch.randn(3, 2, 3, 3, device=device, dtype=dtype)).requires_grad_()
    bias = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

    r = circular_conv2d(parent, weight, bias)
    mu = torch.sigmoid(r)
    eps = target - mu
    d = eps * sigmoid_prime_from_mu(mu)
    manual_w, manual_b = conv_local_weight_grads(parent, d, weight.shape)

    loss = 0.5 * eps.square().sum() / parent.shape[0]
    loss.backward()

    if not torch.allclose(manual_w, -weight.grad, rtol=1e-4, atol=1e-5):
        raise AssertionError("manual PC convolution weight gradient does not match autograd check")
    if not torch.allclose(manual_b, -bias.grad, rtol=1e-4, atol=1e-5):
        raise AssertionError("manual PC convolution bias gradient does not match autograd check")


# -----------------------------------------------------------------------------
# PC inference and local post-inference learning
# -----------------------------------------------------------------------------


@torch.no_grad()
def infer_density(
    model: FluidPCN,
    x: torch.Tensor,
    cfg: Config,
    steps: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Run local PC inference for z1/z2 plus learned incompressible advection."""

    total_steps = cfg.infer_steps if steps is None else steps
    z1 = model.random_density(x)
    z2 = model.random_top_latent(x)
    value = model.value_map(x)

    max_mass_error = 0.0
    min_density = float("inf")
    mean_divergence = 0.0
    mean_cfl = 0.0

    for step in range(total_steps):
        z1, z2 = pc_latent_updates(model, x, z1, z2, cfg)
        min_density = min(min_density, z1.min().item())

        if cfg.disable_flow:
            continue

        u = model.velocity(z1, x, value)
        mass_before = z1.sum(dim=(1, 2, 3))
        z1 = transport_step(z1, u, step, total_steps, cfg)
        mass_after = z1.sum(dim=(1, 2, 3))

        mass_error = (mass_after - mass_before).abs().max().item()
        max_mass_error = max(max_mass_error, mass_error)
        min_density = min(min_density, z1.min().item())
        mean_divergence += divergence(u).square().mean().sqrt().item()
        mean_cfl += cfl_number(u, cfg.dt).mean().item()

    denom = max(total_steps, 1)
    diagnostics = {
        "mass_error": max_mass_error,
        "min_density": min_density if min_density != float("inf") else z1.min().item(),
        "divergence_rms": mean_divergence / denom,
        "cfl": mean_cfl / denom,
        **pc_state_diagnostics(model, x, z1, z2),
    }
    return z1, z2, diagnostics


def train_batch(
    model: FluidPCN,
    optimizers: Dict[str, torch.optim.Optimizer],
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
) -> Dict[str, float]:
    x = images.to(DEVICE)
    y = labels.to(DEVICE)

    # Fixed-point inference is detached: no backpropagation through the long
    # reaction/advection chain.
    z1_star, z2_star, inference_diag = infer_density(model, x, cfg)
    z1_fixed = z1_star.detach()
    z2_fixed = z2_star.detach()

    # ------------------------------------------------------------------
    # 1. Local generative-PC parameter update
    # ------------------------------------------------------------------
    pc_diag = pc_local_weight_updates(model, x, z1_fixed, z2_fixed, cfg)

    # ------------------------------------------------------------------
    # 2. Value-like cost-to-go update
    # ------------------------------------------------------------------
    optimizers["value"].zero_grad(set_to_none=True)
    value_pred = model.value_map(x)
    value_target = model.target_distance(y)
    value_loss = F.mse_loss(value_pred, value_target)
    weighted_value_loss = cfg.value_weight * value_loss
    weighted_value_loss.backward()
    optimizers["value"].step()

    # ------------------------------------------------------------------
    # 3. Short receding-horizon flow update
    # ------------------------------------------------------------------
    optimizers["flow"].zero_grad(set_to_none=True)
    value_for_flow = model.value_map(x).detach()
    a_roll = z1_fixed

    shape_terms = []
    control_terms = []
    divergence_terms = []
    mass_terms = []

    for step in range(cfg.control_horizon):
        if cfg.disable_flow:
            u = torch.zeros(x.shape[0], 2, IMG, IMG, device=x.device, dtype=x.dtype)
            a_next = a_roll
        else:
            u = model.velocity(a_roll, x, value_for_flow)
            a_next = transport_step(a_roll, u, step, cfg.control_horizon, cfg)

        # Nonlocal credit: reduce expected learned cost-to-go before mass has
        # reached the target region.
        shape_terms.append((a_next * value_for_flow).sum(dim=(1, 2, 3)))
        control_terms.append(0.5 * u.square().mean(dim=(1, 2, 3)))
        divergence_terms.append(divergence(u).square().mean(dim=(1, 2, 3)))
        mass_terms.append((a_next.sum(dim=(1, 2, 3)) - 1.0).square())
        a_roll = a_next

    logits = model.region_logits(a_roll)
    class_per_sample = F.cross_entropy(logits, y, reduction="none")
    class_loss = class_per_sample.mean()

    shape_loss = torch.stack(shape_terms, dim=0).mean()
    control_loss = torch.stack(control_terms, dim=0).mean()
    divergence_loss = torch.stack(divergence_terms, dim=0).mean()
    mass_loss = torch.stack(mass_terms, dim=0).mean()

    flow_loss = (
        cfg.class_weight * class_loss
        + cfg.shape_weight * shape_loss
        + cfg.control_weight * control_loss
        + cfg.divergence_weight * divergence_loss
    )

    if not cfg.disable_flow:
        flow_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stream_controller.parameters(), max_norm=5.0)
        optimizers["flow"].step()

    total_energy = (
        cfg.pc_weight * pc_diag["pc_energy"]
        + weighted_value_loss.item()
        + flow_loss.item()
    )

    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item() * 100.0

    return {
        "energy": total_energy,
        "pc_recon_loss": pc_diag["pc_recon_loss"],
        "pc_hidden_loss": pc_diag["pc_hidden_loss"],
        "pc_energy": pc_diag["pc_energy"],
        "pc_dW0_norm": pc_diag["pc_dW0_norm"],
        "pc_dW1_norm": pc_diag["pc_dW1_norm"],
        "value": value_loss.item(),
        "ce": class_loss.item(),
        "shape": shape_loss.item(),
        "control": control_loss.item(),
        "div": divergence_loss.item(),
        "mass_loss": mass_loss.item(),
        "acc": acc,
        **inference_diag,
    }


@torch.no_grad()
def evaluate(model: FluidPCN, loader: DataLoader, cfg: Config) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_correct = 0
    total_seen = 0
    diag_sum = {
        "mass_error": 0.0,
        "min_density": 0.0,
        "divergence_rms": 0.0,
        "cfl": 0.0,
        "pc_recon_loss": 0.0,
        "pc_hidden_loss": 0.0,
        "pc_energy": 0.0,
    }
    batches = 0

    for images, labels in loader:
        x = images.to(DEVICE)
        y = labels.to(DEVICE)
        z1, _z2, diag = infer_density(model, x, cfg)
        logits = model.region_logits(z1)
        pred = logits.argmax(dim=1)

        total_correct += (pred == y).sum().item()
        total_seen += y.numel()
        for key in diag_sum:
            diag_sum[key] += diag[key]
        batches += 1

    for key in diag_sum:
        diag_sum[key] /= max(batches, 1)
    return 100.0 * total_correct / max(total_seen, 1), diag_sum


# -----------------------------------------------------------------------------
# Data, diagnostics, and CLI
# -----------------------------------------------------------------------------


def make_loader(train: bool, subset: int, batch_size: int) -> DataLoader:
    data = datasets.MNIST(
        "./data",
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    if subset > 0:
        subset = min(subset, len(data))
        data = Subset(data, list(range(subset)))
    return DataLoader(data, batch_size=batch_size, shuffle=train, num_workers=0)


def make_optimizers(model: FluidPCN, cfg: Config) -> Dict[str, torch.optim.Optimizer]:
    return {
        "value": torch.optim.Adam(model.value_encoder.parameters(), lr=cfg.value_lr),
        "flow": torch.optim.Adam(model.stream_controller.parameters(), lr=cfg.flow_lr),
    }


def smoke_test(cfg: Config) -> None:
    # Keep the smoke test fast even when full training uses many inference steps.
    smoke_cfg = Config(**{**cfg.__dict__, "infer_steps": min(cfg.infer_steps, 3), "control_horizon": min(cfg.control_horizon, 2)})
    model = FluidPCN(smoke_cfg).to(DEVICE)
    optimizers = make_optimizers(model, smoke_cfg)
    validate_manual_pc_gradients(DEVICE)

    x = torch.rand(8, 1, IMG, IMG)
    y = torch.randint(0, NUM_CLASSES, (8,))
    metrics = train_batch(model, optimizers, x, y, smoke_cfg)

    model.eval()
    with torch.no_grad():
        z1, z2, diag = infer_density(model, x.to(DEVICE), smoke_cfg, steps=3)
        logits = model.region_logits(z1)

    assert z1.shape == (8, 1, IMG, IMG)
    assert z2.shape == (8, smoke_cfg.z2_channels, IMG, IMG)
    assert logits.shape == (8, NUM_CLASSES)
    assert torch.isfinite(z1).all()
    assert torch.isfinite(z2).all()
    assert torch.isfinite(logits).all()
    assert z1.min().item() >= -1e-7
    assert torch.allclose(z1.sum(dim=(1, 2, 3)), torch.ones(8, device=z1.device), atol=1e-4)
    assert diag["mass_error"] < 1e-4
    assert diag["divergence_rms"] < 1e-4

    print("smoke test passed")
    print("manual PC gradient check passed")
    print(
        " ".join(
            [
                f"energy={metrics['energy']:.4f}",
                f"pc_recon={metrics['pc_recon_loss']:.6f}",
                f"pc_hidden={metrics['pc_hidden_loss']:.6f}",
                f"value={metrics['value']:.4f}",
                f"ce={metrics['ce']:.4f}",
                f"shape={metrics['shape']:.4f}",
                f"acc={metrics['acc']:.1f}%",
            ]
        )
    )
    print(
        f"mass_error={diag['mass_error']:.3e} "
        f"min_density={diag['min_density']:.3e} "
        f"div_rms={diag['divergence_rms']:.3e} "
        f"cfl={diag['cfl']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--train_subset", type=int, default=40000)
    p.add_argument("--test_subset", type=int, default=10000)
    p.add_argument("--infer_steps", type=int, default=20)
    p.add_argument("--hidden_lr", type=float, default=0.01)
    p.add_argument("--z2_channels", type=int, default=8)
    p.add_argument("--pc_weight_decay", type=float, default=1e-4)
    p.add_argument("--pc_latent_clip", type=float, default=1.0)
    p.add_argument("--control_horizon", type=int, default=5)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--target_cfl", type=float, default=0.35)
    p.add_argument("--diffusion_start", type=float, default=0.015)
    p.add_argument("--pc_lr", type=float, default=2e-3)
    p.add_argument("--value_lr", type=float, default=2e-3)
    p.add_argument("--flow_lr", type=float, default=2e-3)
    p.add_argument("--pc_weight", type=float, default=1.0)
    p.add_argument("--value_weight", type=float, default=1.0)
    p.add_argument("--class_weight", type=float, default=1.0)
    p.add_argument("--shape_weight", type=float, default=2.0)
    p.add_argument("--control_weight", type=float, default=1e-3)
    p.add_argument("--divergence_weight", type=float, default=1e-2)
    p.add_argument("--disable_flow", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        infer_steps=args.infer_steps,
        hidden_lr=args.hidden_lr,
        z2_channels=args.z2_channels,
        pc_weight_decay=args.pc_weight_decay,
        pc_latent_clip=args.pc_latent_clip,
        control_horizon=args.control_horizon,
        dt=args.dt,
        target_cfl=args.target_cfl,
        diffusion_start=args.diffusion_start,
        pc_lr=args.pc_lr,
        value_lr=args.value_lr,
        flow_lr=args.flow_lr,
        pc_weight=args.pc_weight,
        value_weight=args.value_weight,
        class_weight=args.class_weight,
        shape_weight=args.shape_weight,
        control_weight=args.control_weight,
        divergence_weight=args.divergence_weight,
        disable_flow=args.disable_flow,
    )


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)

    print("Paper-aligned discriminative Fluid-PCN routing experiment")
    print(f"device={DEVICE}")
    print(
        f"infer_steps={cfg.infer_steps} z2_channels={cfg.z2_channels} "
        f"control_horizon={cfg.control_horizon} "
        f"target_cfl={cfg.target_cfl} diffusion_start={cfg.diffusion_start} "
        f"flow={'off' if cfg.disable_flow else 'on'}"
    )

    if args.smoke_test:
        smoke_test(cfg)
        return

    model = FluidPCN(cfg).to(DEVICE)
    optimizers = make_optimizers(model, cfg)
    train_loader = make_loader(True, cfg.train_subset, cfg.batch_size)
    test_loader = make_loader(False, cfg.test_subset, cfg.batch_size)

    best_test = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            metrics = train_batch(model, optimizers, images, labels, cfg)
            if batch_idx % 20 == 0:
                print(
                    f"epoch={epoch} batch={batch_idx:03d} "
                    f"E={metrics['energy']:.4f} "
                    f"pc0={metrics['pc_recon_loss']:.6f} pc1={metrics['pc_hidden_loss']:.6f} "
                    f"V={metrics['value']:.4f} ce={metrics['ce']:.4f} "
                    f"shape={metrics['shape']:.4f} acc={metrics['acc']:.1f}% "
                    f"mass_err={metrics['mass_error']:.1e} "
                    f"div={metrics['divergence_rms']:.1e} cfl={metrics['cfl']:.3f}"
                )

        test_acc, diag = evaluate(model, test_loader, cfg)
        best_test = max(best_test, test_acc)
        print("=" * 90)
        print(
            f"epoch={epoch} test_acc={test_acc:.2f}% best={best_test:.2f}% "
            f"pc0={diag['pc_recon_loss']:.6f} pc1={diag['pc_hidden_loss']:.6f} "
            f"mass_err={diag['mass_error']:.2e} div={diag['divergence_rms']:.2e} "
            f"cfl={diag['cfl']:.3f}"
        )
        print("=" * 90)


if __name__ == "__main__":
    main()



# let me put my questions here

# 1 what is cnn and what are its difference from standared feedforwared neural network
# 2 is there a difference between conv and conv2d cuz i have seen parameter difference intake 
# 3 inside pc_pridiction archtecture we take a density and the actual image being sourced, 
# what exactly does the changing a in every inference step give to the convolution layer and what exactlly is the conv learning


# 4 why with the actuall image and doesnt normalization create loss of infomation, even when we source it,
# what is the point of concentrating the image mass to the center with 6 by 6 dimention it to 28 by 28 source


# 5 ofc this is totally different pc archtecture from what i know like even with one hidden layer there should be things that happened
# 1 we have our image then our single layer and output layer, if we are considering implementing generative pc we take the final output prediction
# which could be w1z2 + b1 with activation will predict z1 which is our hidden layer hidden state and we predict w0z1 + b0 to predict
# the actual image then using the two prediction error we find the gradient and update the z1 and for z2 we only use the prediction error 
# taken from the difference between w1z2 + b1 with activation and actual z1, and as fara as i know intially all this parameter will be 
# intialized randomelly at least when we start then errors and learning from the actuall input train the hidden states

# i am not sure the use of conv is related with the need to preprsent fluidic behviour cuz we have the circular padding
# even having the convolution not seen this sequention prediction of z1 and z2, all the prediction error calculation makes me dout the pc_prediction function
# regardless of its number of layer, so i want explantion what exactlly is happning or what kind of learning is happning inside the function

# tbh i dont get the following line
#   asks: Given the density in this location and its neighboring cells, what density pattern should be locally expected?
# so like what exactly are we learning/predicting here, i know a is changing and we have the actual image both contered in the
# center what does the conv2d bring to the table plus that does the cobination of the actula source image and pc local bring

# this all could be due to i dont have any knowuldge about cnn archtecure but i want to ask check wheather we are doing the actuall pc with intention fluid dynamic
