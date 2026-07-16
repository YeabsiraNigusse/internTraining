"""
fixed_discriminative_fluid_pcn_manual.py

Clean DISCRIMINATIVE Fluid-PCN MNIST example.

This fixes the earlier mixed code.

Model type:
    image x -> hidden density a -> class logits

Errors used:
    1. hidden/density prediction error: eps_a = a - a_hat
    2. class/output error: softmax cross-entropy

Errors NOT used:
    - no image reconstruction error
    - no decoder
    - no recon_loss

Training style:
    - no loss.backward()
    - no optimizer.step()
    - manual hidden-density inference
    - manual PC-style weight updates

Fluid part:
    - after the PC reaction step, density is moved by advection
    - velocity is produced from a simple stream function
    - stream weights are kept fixed in this educational manual version

Run:
    python fixed_discriminative_fluid_pcn_manual.py --smoke_test

Small run:
    python fixed_discriminative_fluid_pcn_manual.py --epochs 1 --train_subset 1000 --test_subset 200
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG = 28
PIXELS = IMG * IMG
BATCH = 128
EPOCHS = 3
TRAIN_SUBSET = 5000
TEST_SUBSET = 1000

# Inner inference settings
INFER_STEPS = 5
H_LR = 0.25

# Manual parameter update rate
W_LR = 0.02

# Fluid settings
DT = 0.5
TARGET_CFL = 0.35
DIFFUSION = 0.001

# Energy weights
DENSITY_WEIGHT = 1.0
CLASS_WEIGHT = 1.0


# ============================================================
# Density and grid operators
# ============================================================

def normalize_density(a, eps=1e-8):
    """
    Make hidden activation positive and mass-normalized.

    a shape: [B, 1, 28, 28]

    After this:
        a > 0
        a.sum(dim=(1,2,3)) = 1
    """
    a = F.softplus(a)
    mass = a.sum(dim=(1, 2, 3), keepdim=True)
    return a / (mass + eps)


def image_to_density(x, eps=1e-6):
    """
    Turn image pixels into initial hidden density.

    x shape: [B, 1, 28, 28]
    """
    a = x.clamp_min(0.0) + eps
    return a / a.sum(dim=(1, 2, 3), keepdim=True)


def roll_x(z, shift):
    return torch.roll(z, shifts=shift, dims=-1)


def roll_y(z, shift):
    return torch.roll(z, shifts=shift, dims=-2)


def ddx(z):
    return 0.5 * (roll_x(z, -1) - roll_x(z, 1))


def ddy(z):
    return 0.5 * (roll_y(z, -1) - roll_y(z, 1))


def laplacian(z):
    return roll_x(z, -1) + roll_x(z, 1) + roll_y(z, -1) + roll_y(z, 1) - 4.0 * z


def velocity_from_stream(psi):
    """
    u_x =  d psi / dy
    u_y = -d psi / dx
    """
    ux = ddy(psi)
    uy = -ddx(psi)
    return torch.cat([ux, uy], dim=1)


def scale_velocity(u):
    max_speed = u.abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = TARGET_CFL / (DT * max_speed + 1e-8)
    return u * scale.clamp(max=10.0)


def advect(a, u):
    """
    Conservative density transport:
        a_new = a - dt * div(a*u)

    This is the discrete continuity-equation step.
    """
    ux = u[:, 0:1]
    uy = u[:, 1:2]

    # horizontal fluxes
    ux_right = 0.5 * (ux + roll_x(ux, -1))
    a_right = roll_x(a, -1)
    flux_right = torch.where(ux_right >= 0, ux_right * a, ux_right * a_right)
    flux_left = roll_x(flux_right, 1)

    # vertical fluxes
    uy_down = 0.5 * (uy + roll_y(uy, -1))
    a_down = roll_y(a, -1)
    flux_down = torch.where(uy_down >= 0, uy_down * a, uy_down * a_down)
    flux_up = roll_y(flux_down, 1)

    div_flux = (flux_right - flux_left) + (flux_down - flux_up)
    return a - DT * div_flux


# ============================================================
# Parameters
# ============================================================

# Discriminative direction:
#     x -> a_hat -> class logits

# input image -> predicted hidden density, linear PC predictor
W_in = torch.randn(PIXELS, PIXELS, device=DEVICE) / PIXELS**0.5
b_a = torch.zeros(PIXELS, device=DEVICE)

# hidden density -> class logits
W_cls = torch.randn(PIXELS, 10, device=DEVICE) / PIXELS**0.5
b_cls = torch.zeros(10, device=DEVICE)

# stream function predictor for velocity.
# Fixed in this manual educational version.
W_stream = 0.02 * torch.randn(1, 2, 3, 3, device=DEVICE)
b_stream = torch.zeros(1, device=DEVICE)


# ============================================================
# Model pieces
# ============================================================

def predict_density_from_image(x):
    """
    Discriminative local prediction:
        image x predicts hidden density a_hat.

    This is the fluid-grid version of:
        h_hat = x @ W_x + b_h
    """
    x_flat = x.flatten(1)
    a_hat_flat = x_flat @ W_in + b_a
    return a_hat_flat.view(-1, 1, IMG, IMG)


def classify_density(a):
    """
    Hidden density predicts class logits.
    """
    a_flat = a.flatten(1)
    return a_flat @ W_cls + b_cls


def softmax_manual(logits):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(shifted)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)


def cross_entropy_manual(probs, y_onehot):
    return -(y_onehot * torch.log(probs + 1e-8)).sum(dim=1).mean()


def make_velocity(a, x):
    """
    Fixed stream function for fluid transport.
    Input channels are [density, image].
    """
    psi = F.conv2d(torch.cat([a, x], dim=1), W_stream, b_stream, padding=1)
    u = velocity_from_stream(psi)
    return scale_velocity(u)


# ============================================================
# Inference and manual learning
# ============================================================

def infer_density(x, y_onehot):
    """
    Inner predictive-coding inference loop.

    Clean discriminative PCN energy:

        E = E_density + E_class

    where:
        E_density = 0.5 / PIXELS * ||a - a_hat||^2
        E_class   = CE(softmax(classify(a)), y)

    No reconstruction error.
    """

    # Feedforward-style initialization.
    # We start from the image density because it is already positive and spatial.
    a = image_to_density(x)

    for _ in range(INFER_STEPS):
        # -----------------------------
        # 1. PC reaction gradient
        # -----------------------------

        a_hat = predict_density_from_image(x)
        eps_a = a - a_hat

        logits = classify_density(a)
        probs = softmax_manual(logits)

        # Manual gradient of density MSE wrt a:
        # d/d a [0.5/PIXELS * ||a - a_hat||^2] = (a - a_hat)/PIXELS
        grad_density = DENSITY_WEIGHT * eps_a / PIXELS

        # Manual gradient of CE wrt logits:
        # d CE / d logits = probs - y
        #
        # logits = flatten(a) @ W_cls + b_cls
        # so:
        # d CE / d flatten(a) = (probs - y) @ W_cls.T
        grad_class_flat = CLASS_WEIGHT * ((probs - y_onehot) @ W_cls.t())
        grad_class = grad_class_flat.view_as(a)

        # Total manual hidden-density gradient
        grad_a = grad_density + grad_class

        # PC reaction update
        a = normalize_density(a - H_LR * grad_a)

        # -----------------------------
        # 2. Fluid advection step
        # -----------------------------
        u = make_velocity(a, x)
        a = advect(a, u)

        # Optional diffusion/smoothing
        a = a + DT * DIFFUSION * laplacian(a)

        # Keep valid density
        a = normalize_density(a)

    return a


def train_batch(images, labels):
    """
    Manual PC/IL update.

    No autograd.
    No backward.
    No optimizer.
    """
    global W_in, b_a, W_cls, b_cls

    x = images.to(DEVICE)
    y_idx = labels.to(DEVICE)
    y_onehot = F.one_hot(y_idx, 10).float()

    B = x.size(0)

    # -----------------------------
    # 1. Infer hidden density
    # -----------------------------
    a = infer_density(x, y_onehot)

    # -----------------------------
    # 2. Recompute final local errors
    # -----------------------------
    a_hat = predict_density_from_image(x)
    eps_a = a - a_hat

    logits = classify_density(a)
    probs = softmax_manual(logits)

    ce = cross_entropy_manual(probs, y_onehot)
    density_energy = 0.5 * eps_a.pow(2).sum(dim=(1, 2, 3)).mean() / PIXELS
    total_energy = density_energy + ce

    # -----------------------------
    # 3. Manual local weight updates
    # -----------------------------

    x_flat = x.flatten(1)
    a_flat = a.flatten(1)
    eps_a_flat = eps_a.flatten(1)

    # x -> hidden-density predictor update.
    # Local PC rule:
    #     W_in += lr * x.T @ eps_a
    W_in += W_LR * (x_flat.t() @ eps_a_flat) / B
    b_a += W_LR * eps_a_flat.mean(dim=0)

    # hidden density -> class update using CE gradient.
    #
    # Gradient descent on CE:
    #     dW = a.T @ (probs - y)
    #     W -= lr * dW
    #
    # Same as:
    #     W += lr * a.T @ (y - probs)
    W_cls += W_LR * (a_flat.t() @ (y_onehot - probs)) / B
    b_cls += W_LR * (y_onehot - probs).mean(dim=0)

    # -----------------------------
    # 4. Metrics
    # -----------------------------
    pred = probs.argmax(dim=1)
    acc = (pred == y_idx).float().mean().item() * 100.0

    return total_energy.item(), ce.item(), density_energy.item(), acc


@torch.no_grad()
def evaluate(loader):
    """
    Testing for discriminative PCN.

    During testing the label is unknown.
    For clean discriminative PCN, we do a feedforward prediction:
        x -> a -> logits

    We use normalized image density as stable visible hidden state.
    """
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        x = images.to(DEVICE)
        y = labels.to(DEVICE)

        # Simple discriminative testing.
        # This is stable because a is a valid density.
        a = image_to_density(x)

        logits = classify_density(a)
        probs = softmax_manual(logits)
        pred = probs.argmax(dim=1)

        total_correct += (pred == y).sum().item()
        total_seen += y.numel()

    return 100.0 * total_correct / total_seen


# ============================================================
# Data and running
# ============================================================

def make_loader(train, subset):
    data = datasets.MNIST(
        "./data",
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    if subset > 0:
        data = Subset(data, list(range(subset)))
    return DataLoader(data, batch_size=BATCH, shuffle=train)


def smoke_test():
    x = torch.rand(8, 1, IMG, IMG)
    y = torch.randint(0, 10, (8,))
    energy, ce, density_e, acc = train_batch(x, y)

    print("smoke test passed")
    print(f"energy={energy:.4f} ce={ce:.4f} density={density_e:.6f} acc={acc:.1f}%")

    # Check density mass after inference
    y_onehot = F.one_hot(y.to(DEVICE), 10).float()
    a = infer_density(x.to(DEVICE), y_onehot)
    print("density shape:", tuple(a.shape))
    print("mass per item:", a.sum(dim=(1, 2, 3)).detach().cpu())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--train_subset", type=int, default=TRAIN_SUBSET)
    p.add_argument("--test_subset", type=int, default=TEST_SUBSET)
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("Clean Discriminative Fluid-PCN MNIST")
    print(f"device={DEVICE}")
    print("Errors: density prediction error + class CE error")
    print("No reconstruction error. No loss.backward(). No optimizer.step().")

    if args.smoke_test:
        smoke_test()
        return

    train_loader = make_loader(train=True, subset=args.train_subset)
    test_loader = make_loader(train=False, subset=args.test_subset)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, (images, labels) in enumerate(train_loader):
            energy, ce, density_e, acc = train_batch(images, labels)

            if batch_idx % 20 == 0:
                print(
                    f"epoch={epoch} batch={batch_idx:03d} "
                    f"energy={energy:.4f} ce={ce:.4f} "
                    f"density={density_e:.6f} acc={acc:.1f}%"
                )

        test_acc = evaluate(test_loader)
        print("=" * 70)
        print(f"epoch={epoch} test_acc={test_acc:.2f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
