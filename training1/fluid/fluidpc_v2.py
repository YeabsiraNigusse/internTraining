"""
fluid_pcn_mnist.py

A practical, discriminative Fluid-PCN experiment for MNIST.

This is not an exact solver of the HJB equation on SDiff(M).  It implements the
paper's practical recipe:

1. Two-hidden-state generative predictive coding: z2 -> z1 -> z0=x.
2. z1 is constrained to a nonnegative unit-mass 28x28 density and transported by the fluid solver.
3. A learned value-like cost-to-go map supervised by the task.
4. A learned stream function psi, giving a divergence-free velocity
       u = (d psi / dy, -d psi / dx).
5. CFL-targeted conservative upwind advection.
6. Diffusion warmup followed by annealing to zero.
7. Nonlocal credit through a distance-to-goal shaping objective and a short
   receding-horizon rollout.
8. A real routing bottleneck: each class is represented by a small target region,
   and the classifier reads only the mass delivered to those ten regions.
9. PC latent and parameter updates are explicit/local and do not use autograd;
   value and flow parameters retain their original optimizer-based learning.

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
    pc_reaction_weight: float = 1.0

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
    """Two-hidden-state generative PC coupled to the existing fluid controller.

    z0 = x is clamped sensory input.
    z1 is the inferred predictive-coding state nearest the image and is also the
    physical nonnegative unit-mass density transported by the fluid solver.
    z2 is an unconstrained upper predictive-coding latent.

    Generative direction: z2 -> z1 -> z0.
    PC parameters are updated explicitly from local parent-activity x child-error
    correlations.  They are not trained by autograd or an optimizer object.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pc_hidden_channels = 8

        # ------------------------------------------------------------------
        # Generative predictive-coding parameters.
        # W0: z1 [B,1,H,W] -> mu0, a prediction of x [B,1,H,W].
        # W1: z2 [B,8,H,W] -> mu1, a prediction of z1 [B,1,H,W].
        # requires_grad=False is deliberate: PC uses explicit local updates.
        # ------------------------------------------------------------------
        self.pc_W0 = nn.Parameter(
            0.05 * torch.randn(1, 1, 3, 3), requires_grad=False
        )
        self.pc_b0 = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.pc_W1 = nn.Parameter(
            0.05 * torch.randn(1, self.pc_hidden_channels, 3, 3),
            requires_grad=False,
        )
        self.pc_b1 = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Value-like cost-to-go map.  This branch is intentionally unchanged.
        self.value_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.GELU(),
            nn.Linear(256, PIXELS),
        )

        # Stream-function controller.  This branch is intentionally unchanged.
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

    @staticmethod
    def _periodic_conv(parent: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """3x3 periodic convolution used by the generative PC predictions."""
        parent_pad = F.pad(parent, (1, 1, 1, 1), mode="circular")
        return F.conv2d(parent_pad, weight, bias=bias)

    @staticmethod
    def _periodic_conv_transpose_error(
        child_delta: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit local transpose of the periodic 3x3 prediction operator.

        If child = Conv(parent, W), this computes the local error projection
        dE/d(parent) up to the sign convention used by PC.  No autograd is used.
        """
        batch, out_channels, height, width = child_delta.shape
        _, in_channels, kh, kw = weight.shape
        result = torch.zeros(
            batch, in_channels, height, width,
            device=child_delta.device,
            dtype=child_delta.dtype,
        )
        cy, cx = kh // 2, kw // 2
        for oy in range(kh):
            for ox in range(kw):
                shift_y = oy - cy
                shift_x = ox - cx
                shifted = torch.roll(
                    child_delta,
                    shifts=(shift_y, shift_x),
                    dims=(-2, -1),
                )
                # weight[:, :, oy, ox] is [out_channels, in_channels].
                result += torch.einsum(
                    "bohw,oi->bihw",
                    shifted,
                    weight[:, :, oy, ox],
                )
        return result

    @staticmethod
    def _periodic_conv_local_grads(
        parent: torch.Tensor,
        child_delta: torch.Tensor,
        weight_shape: torch.Size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Local convolutional Hebbian/error correlation.

        For each shared kernel weight:
            dW ~ mean(parent activity in that kernel position * child delta)
        and db ~ mean(child delta).

        These are explicit local PC updates; no backward()/autograd.grad is used.
        """
        out_channels, in_channels, kh, kw = weight_shape
        dW = torch.zeros(
            out_channels, in_channels, kh, kw,
            device=parent.device,
            dtype=parent.dtype,
        )
        cy, cx = kh // 2, kw // 2
        for oy in range(kh):
            for ox in range(kw):
                off_y = oy - cy
                off_x = ox - cx
                # forward output[y,x] reads parent[y+off_y, x+off_x]
                parent_at_kernel_offset = torch.roll(
                    parent,
                    shifts=(-off_y, -off_x),
                    dims=(-2, -1),
                )
                dW[:, :, oy, ox] = torch.einsum(
                    "bohw,bihw->oi",
                    child_delta,
                    parent_at_kernel_offset,
                ) / (parent.shape[0] * parent.shape[2] * parent.shape[3])
        db = child_delta.mean(dim=(0, 2, 3))
        return dW, db

    def random_density(self, x: torch.Tensor) -> torch.Tensor:
        """Random central unit-mass z1, independent of image content."""
        raw = (torch.rand_like(x) + EPS) * self.source_mask
        return normalize_density(raw)

    def initial_z2(self, x: torch.Tensor) -> torch.Tensor:
        """Image-independent top latent initialization."""
        return torch.zeros(
            x.shape[0], self.pc_hidden_channels, IMG, IMG,
            device=x.device,
            dtype=x.dtype,
        )

    def pc_predictions(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Top-down generative predictions z2 -> z1 -> x."""
        r0 = self._periodic_conv(z1, self.pc_W0, self.pc_b0)
        mu0 = torch.sigmoid(r0)

        r1 = self._periodic_conv(z2, self.pc_W1, self.pc_b1)
        mu1 = torch.tanh(r1)

        return {"r0": r0, "mu0": mu0, "r1": r1, "mu1": mu1}

    @staticmethod
    def pc_errors(
        x: torch.Tensor,
        z1: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """README convention: epsilon = target state - top-down prediction."""
        eps0 = x - predictions["mu0"]
        eps1 = z1 - predictions["mu1"]
        return {"eps0": eps0, "eps1": eps1}

    def pc_latent_updates(
        self,
        x: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        cfg: Config,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """One explicit local PC inference step for z1 and z2."""
        pred = self.pc_predictions(z1, z2)
        err = self.pc_errors(x, z1, pred)

        # z1 predicts x.  Bottom image path uses sigmoid'.
        d0 = err["eps0"] * pred["mu0"] * (1.0 - pred["mu0"])
        bottom_up_0 = self._periodic_conv_transpose_error(d0, self.pc_W0)

        # z1 is also predicted by z2, so its two adjacent errors compete.
        delta_z1 = bottom_up_0 - err["eps1"]

        # z2 is the top latent: only its prediction error below updates it.
        d1 = err["eps1"] * (1.0 - pred["mu1"].square())
        delta_z2 = self._periodic_conv_transpose_error(d1, self.pc_W1)

        # Keep the explicit PC state update numerically bounded.
        delta_z1 = delta_z1.clamp(-5.0, 5.0)
        delta_z2 = delta_z2.clamp(-5.0, 5.0)
        step_size = cfg.hidden_lr * cfg.pc_reaction_weight

        z1_new = z1 + step_size * delta_z1
        z2_new = z2 + step_size * delta_z2

        # Physical coupling constraint: z1 must be a nonnegative unit-mass density
        # before it is given to the unchanged fluid transport solver.
        z1_new = normalize_density(z1_new.clamp_min(EPS))

        return z1_new, z2_new, {**pred, **err, "d0": d0, "d1": d1}

    @torch.no_grad()
    def apply_local_pc_weight_update(
        self,
        x: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        cfg: Config,
    ) -> Dict[str, float]:
        """Update W0,b0,W1,b1 using only local parent activity and child error."""
        pred = self.pc_predictions(z1, z2)
        err = self.pc_errors(x, z1, pred)

        d0 = err["eps0"] * pred["mu0"] * (1.0 - pred["mu0"])
        d1 = err["eps1"] * (1.0 - pred["mu1"].square())

        dW0, db0 = self._periodic_conv_local_grads(z1, d0, self.pc_W0.shape)
        dW1, db1 = self._periodic_conv_local_grads(z2, d1, self.pc_W1.shape)

        # eps = target - prediction, so these local correlations are ascent
        # directions for prediction quality; use +lr as in the README.
        lr = cfg.pc_lr * cfg.pc_weight
        weight_decay = 0.0
        self.pc_W0.add_(lr * (dW0 - weight_decay * self.pc_W0))
        self.pc_b0.add_(lr * db0)
        self.pc_W1.add_(lr * (dW1 - weight_decay * self.pc_W1))
        self.pc_b1.add_(lr * db1)

        recon = 0.5 * err["eps0"].square().mean()
        hidden = 0.5 * err["eps1"].square().mean()
        return {
            "pc_recon": recon.item(),
            "pc_hidden": hidden.item(),
            "pc_energy": (recon + hidden).item(),
        }

    def value_map(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.value_encoder(x).view(-1, 1, IMG, IMG)
        return torch.sigmoid(raw)

    def velocity(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        value_map: torch.Tensor,
    ) -> torch.Tensor:
        psi = self.stream_controller(torch.cat([a, x, value_map], dim=1))
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
# PC inference and local post-inference learning
# -----------------------------------------------------------------------------


@torch.no_grad()
def infer_density(
    model: FluidPCN,
    x: torch.Tensor,
    cfg: Config,
    steps: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Run local generative PC reaction + unchanged incompressible advection."""

    total_steps = cfg.infer_steps if steps is None else steps
    z1 = model.random_density(x)
    z2 = model.initial_z2(x)
    value = model.value_map(x)

    max_mass_error = 0.0
    min_density = float("inf")
    mean_divergence = 0.0
    mean_cfl = 0.0

    for step in range(total_steps):
        # Local PC reaction:
        #   mu0 = sigmoid(W0*z1+b0), eps0 = x-mu0
        #   mu1 = tanh(W1*z2+b1),   eps1 = z1-mu1
        #   delta_z1 = W0^T(eps0*sigmoid') - eps1
        #   delta_z2 = W1^T(eps1*tanh')
        z1, z2, _ = model.pc_latent_updates(x, z1, z2, cfg)

        if cfg.disable_flow:
            continue

        # Existing fluid path: z1 is the density transported by u.
        u = model.velocity(z1, x, value)
        mass_before = z1.sum(dim=(1, 2, 3))
        z1 = transport_step(z1, u, step, total_steps, cfg)
        mass_after = z1.sum(dim=(1, 2, 3))

        mass_error = (mass_after - mass_before).abs().max().item()
        max_mass_error = max(max_mass_error, mass_error)
        min_density = min(min_density, z1.min().item())
        mean_divergence += divergence(u).square().mean().sqrt().item()
        mean_cfl += cfl_number(u, cfg.dt).mean().item()

    final_pred = model.pc_predictions(z1, z2)
    final_err = model.pc_errors(x, z1, final_pred)
    recon = 0.5 * final_err["eps0"].square().mean().item()
    hidden = 0.5 * final_err["eps1"].square().mean().item()

    denom = max(total_steps, 1)
    diagnostics = {
        "mass_error": max_mass_error,
        "min_density": min_density if min_density != float("inf") else z1.min().item(),
        "divergence_rms": mean_divergence / denom,
        "cfl": mean_cfl / denom,
        "infer_pc_recon": recon,
        "infer_pc_hidden": hidden,
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

    # Fixed-state inference.  No autograd graph is built through PC or transport.
    z1_star, z2_star, inference_diag = infer_density(model, x, cfg)
    a_fixed = z1_star.detach()
    z2_fixed = z2_star.detach()

    # ------------------------------------------------------------------
    # 1. LOCAL PC parameter update: no backward(), no Adam.
    # ------------------------------------------------------------------
    pc_diag = model.apply_local_pc_weight_update(x, a_fixed, z2_fixed, cfg)
    pc_energy = torch.tensor(pc_diag["pc_energy"], device=x.device, dtype=x.dtype)
    weighted_pc_loss = cfg.pc_weight * pc_energy

    # ------------------------------------------------------------------
    # 2. Value-like cost-to-go update (unchanged; autograd intentionally kept).
    # ------------------------------------------------------------------
    optimizers["value"].zero_grad(set_to_none=True)
    value_pred = model.value_map(x)
    value_target = model.target_distance(y)
    value_loss = F.mse_loss(value_pred, value_target)
    weighted_value_loss = cfg.value_weight * value_loss
    weighted_value_loss.backward()
    optimizers["value"].step()

    # ------------------------------------------------------------------
    # 3. Short receding-horizon flow update (unchanged).
    # ------------------------------------------------------------------
    optimizers["flow"].zero_grad(set_to_none=True)
    value_for_flow = model.value_map(x).detach()
    a_roll = a_fixed

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

    total_energy = weighted_pc_loss + weighted_value_loss + flow_loss

    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item() * 100.0

    return {
        "energy": total_energy.item(),
        "pc": pc_diag["pc_energy"],
        "pc_recon": pc_diag["pc_recon"],
        "pc_hidden": pc_diag["pc_hidden"],
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
    diag_sum = {"mass_error": 0.0, "min_density": 0.0, "divergence_rms": 0.0, "cfl": 0.0}
    batches = 0

    for images, labels in loader:
        x = images.to(DEVICE)
        y = labels.to(DEVICE)
        a, _z2, diag = infer_density(model, x, cfg)
        logits = model.region_logits(a)
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
    # PC parameters are intentionally absent: they use explicit local updates.
    return {
        "value": torch.optim.Adam(model.value_encoder.parameters(), lr=cfg.value_lr),
        "flow": torch.optim.Adam(model.stream_controller.parameters(), lr=cfg.flow_lr),
    }



def validate_pc_local_math() -> Dict[str, float]:
    """Numerically compare explicit PC local rules with autograd on a tiny test.

    Autograd is used only here as a verification oracle.  The actual PC training
    path never calls backward() or autograd.grad for PC latent/weight updates.
    """
    torch.manual_seed(123)
    batch, height, width = 2, 7, 7
    results: Dict[str, float] = {}

    for name, in_channels, activation in (
        ("bottom", 1, "sigmoid"),
        ("hidden", 8, "tanh"),
    ):
        parent = torch.randn(batch, in_channels, height, width, requires_grad=True)
        weight = torch.randn(1, in_channels, 3, 3, requires_grad=True)
        bias = torch.randn(1, requires_grad=True)
        target = (
            torch.rand(batch, 1, height, width)
            if activation == "sigmoid"
            else torch.randn(batch, 1, height, width)
        )

        r = FluidPCN._periodic_conv(parent, weight, bias)
        if activation == "sigmoid":
            mu = torch.sigmoid(r)
            deriv = mu * (1.0 - mu)
        else:
            mu = torch.tanh(r)
            deriv = 1.0 - mu.square()
        eps = target - mu
        delta = eps * deriv

        manual_dw, manual_db = FluidPCN._periodic_conv_local_grads(
            parent.detach(), delta.detach(), weight.shape
        )
        manual_bu = FluidPCN._periodic_conv_transpose_error(
            delta.detach(), weight.detach()
        )

        # Mean energy validates local parameter updates.  Since eps=target-mu,
        # the explicit PC update direction equals -dE/dtheta.
        loss = 0.5 * eps.square().mean()
        loss.backward()
        results[f"{name}_weight_err"] = (manual_dw + weight.grad).abs().max().item()
        results[f"{name}_bias_err"] = (manual_db + bias.grad).abs().max().item()

        # Sum energy validates the unnormalized per-state bottom-up direction.
        parent2 = parent.detach().clone().requires_grad_(True)
        r2 = FluidPCN._periodic_conv(parent2, weight.detach(), bias.detach())
        mu2 = torch.sigmoid(r2) if activation == "sigmoid" else torch.tanh(r2)
        eps2 = target - mu2
        (0.5 * eps2.square().sum()).backward()
        results[f"{name}_state_err"] = (manual_bu + parent2.grad).abs().max().item()

    return results


def smoke_test(cfg: Config) -> None:
    local_check = validate_pc_local_math()
    assert max(local_check.values()) < 1e-5, local_check

    # Keep the smoke test fast even when full training uses many inference steps.
    smoke_cfg = Config(**{**cfg.__dict__, "infer_steps": min(cfg.infer_steps, 3), "control_horizon": min(cfg.control_horizon, 2)})
    model = FluidPCN(smoke_cfg).to(DEVICE)
    optimizers = make_optimizers(model, smoke_cfg)

    x = torch.rand(8, 1, IMG, IMG)
    y = torch.randint(0, NUM_CLASSES, (8,))
    metrics = train_batch(model, optimizers, x, y, smoke_cfg)

    model.eval()
    with torch.no_grad():
        a, z2, diag = infer_density(model, x.to(DEVICE), smoke_cfg, steps=3)
        logits = model.region_logits(a)

    assert a.shape == (8, 1, IMG, IMG)
    assert z2.shape == (8, model.pc_hidden_channels, IMG, IMG)
    assert (a >= 0).all()
    assert logits.shape == (8, NUM_CLASSES)
    assert torch.isfinite(a).all()
    assert torch.isfinite(logits).all()
    assert abs(a.sum(dim=(1, 2, 3)).mean().item() - 1.0) < 1e-4
    assert diag["mass_error"] < 1e-4
    assert diag["divergence_rms"] < 1e-4

    print("smoke test passed")
    print(f"pc_local_math_max_error={max(local_check.values()):.3e}")
    print(
        " ".join(
            [
                f"energy={metrics['energy']:.4f}",
                f"pc={metrics['pc']:.6f}",
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
    p.add_argument("--hidden_lr", type=float, default=0.05)
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
        f"infer_steps={cfg.infer_steps} control_horizon={cfg.control_horizon} "
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
                    f"E={metrics['energy']:.4f} pc={metrics['pc']:.6f} "
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
            f"mass_err={diag['mass_error']:.2e} div={diag['divergence_rms']:.2e} "
            f"cfl={diag['cfl']:.3f}"
        )
        print("=" * 90)


if __name__ == "__main__":
    main()
