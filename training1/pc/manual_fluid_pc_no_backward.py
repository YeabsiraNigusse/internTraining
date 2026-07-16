import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG = 28
BATCH = 128
EPOCHS = 3
TRAIN_SUBSET = 5000
TEST_SUBSET = 1000

# Inner hidden-state inference settings.
STEPS = 5
REACTION_LR = 0.25
DT = 0.5
TARGET_CFL = 0.35
DIFFUSION = 0.001

# Manual parameter update learning rate.
LR = 1e-2

# Energy weights used inside hidden-state inference and manual parameter updates.
CLASS_WEIGHT = 0.20
RECON_WEIGHT = 0.20
PRED_WEIGHT = 0.20


def normalize_density(a, eps=1e-8):
    """Keep hidden activation positive and make each image's mass sum to 1."""
    a = F.softplus(a)
    mass = a.sum(dim=(1, 2, 3), keepdim=True)
    return a / (mass + eps)


def image_to_density(x, eps=1e-6):
    """Use the input image as the first hidden density."""
    a = x.clamp_min(0.0) + eps
    return a / a.sum(dim=(1, 2, 3), keepdim=True)


def roll_x(z, shift):
    return torch.roll(z, shifts=shift, dims=-1)


def roll_y(z, shift):
    return torch.roll(z, shifts=shift, dims=-2)


def ddx(z):
    """Central difference in the horizontal direction."""
    return 0.5 * (roll_x(z, -1) - roll_x(z, 1))


def ddy(z):
    """Central difference in the vertical direction."""
    return 0.5 * (roll_y(z, -1) - roll_y(z, 1))


def laplacian(z):
    """Small smoothing operator used as diffusion."""
    return roll_x(z, -1) + roll_x(z, 1) + roll_y(z, -1) + roll_y(z, 1) - 4 * z


def velocity_from_stream(psi):
    ux = ddy(psi)
    uy = -ddx(psi)
    return torch.cat([ux, uy], dim=1)


def scale_velocity(u):
    max_speed = u.abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = TARGET_CFL / (DT * max_speed + 1e-8)
    return u * scale.clamp(max=10.0)


def advect(a, u):
    """Conservative upwind advection: a_new = a - dt * div(a * u)."""
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
    return a - DT * div_flux


# -------------------------------------------------------------------
# Trainable tensors.
# Important difference from the previous file:
# W_pred predicts density from x only, not from a and x.
# This makes the manual hidden gradient simple:
#     d/da 0.5 ||a - a_hat(x)||^2 = a - a_hat(x)
# -------------------------------------------------------------------
W_pred = (0.02 * torch.randn(1, 1, 3, 3, device=DEVICE))
b_pred = torch.zeros(1, device=DEVICE)

W_dec = (0.02 * torch.randn(1, 1, 5, 5, device=DEVICE))
b_dec = torch.zeros(1, device=DEVICE)

W_stream = (0.02 * torch.randn(1, 2, 3, 3, device=DEVICE))
b_stream = torch.zeros(1, device=DEVICE)

W_cls = (0.02 * torch.randn(IMG * IMG, 10, device=DEVICE))
b_cls = torch.zeros(10, device=DEVICE)


def predict_density_from_x(x):
    """Input image predicts hidden density. This is the PC prior/prediction for a."""
    pre = F.conv2d(x, W_pred, b_pred, padding=1)
    a_hat = torch.sigmoid(pre)
    return pre, a_hat


def reconstruct_with_pre(a):
    """Hidden density predicts/reconstructs image."""
    pre = F.conv2d(a, W_dec, b_dec, padding=2)
    x_hat = torch.sigmoid(pre)
    return pre, x_hat


def classify(a):
    """Class logits from final hidden density."""
    return a.flatten(1) @ W_cls + b_cls


def manual_softmax_cross_entropy(logits, labels):
    """
    Manual CE and its gradient.

    CE = mean_i -log softmax(logits_i)[label_i]
    dCE/dlogits = (probs - one_hot(labels)) / batch
    """
    y_onehot = F.one_hot(labels, 10).float()
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(shifted)
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    ce = -(y_onehot * torch.log(probs + 1e-8)).sum(dim=1).mean()
    grad_logits_mean = (probs - y_onehot) / logits.size(0)
    grad_logits_per_sample = probs - y_onehot
    return ce, probs, y_onehot, grad_logits_mean, grad_logits_per_sample


def make_velocity(a, x):
    """Learn a stream image from [hidden density, input image], then curl it."""
    psi = F.conv2d(torch.cat([a, x], dim=1), W_stream, b_stream, padding=1)
    return scale_velocity(velocity_from_stream(psi))


def conv2d_weight_grad(input_tensor, grad_out, kernel_size, padding):
    """
    Manual conv2d weight gradient for stride=1.

    input_tensor: [B, C_in, H, W]
    grad_out:     [B, C_out, H, W]
    returns:      [C_out, C_in, K, K]
    """
    patches = F.unfold(input_tensor, kernel_size=kernel_size, padding=padding)
    # patches: [B, C_in*K*K, H*W]
    grad_flat = grad_out.flatten(2)
    # grad_flat: [B, C_out, H*W]
    dW = torch.einsum("bol,bil->oi", grad_flat, patches)
    return dW.view(grad_out.shape[1], input_tensor.shape[1], kernel_size, kernel_size)


def manual_hidden_gradient(a, x, labels=None):
    """
    Manual version of grad_a = dE/da.

    E(a) =
        PRED_WEIGHT  * 0.5 * mean_grid((a - a_hat(x))^2)
      + RECON_WEIGHT * 0.5 * mean_grid((x_hat(a) - x)^2)
      + CLASS_WEIGHT * CE(classify(a), y)      if labels is not None

    This does not use torch.autograd.grad.
    """
    B = x.size(0)
    n_grid = IMG * IMG

    # Hidden prediction error term.
    _, a_hat = predict_density_from_x(x)
    eps_a = a - a_hat
    grad_pred = PRED_WEIGHT * eps_a / n_grid
    pred_loss = 0.5 * eps_a.pow(2).mean()

    # Reconstruction term.
    pre_x, x_hat = reconstruct_with_pre(a)
    eps_x = x_hat - x
    # d/dpre sigmoid(pre) = x_hat * (1 - x_hat)
    grad_pre_x = RECON_WEIGHT * eps_x * x_hat * (1.0 - x_hat) / n_grid
    grad_recon = F.conv_transpose2d(grad_pre_x, W_dec, padding=2)
    recon_loss = 0.5 * eps_x.pow(2).mean()

    # Optional class term. During training, labels are clamped and push hidden density.
    # During testing, labels are unknown, so this term is omitted.
    class_loss = torch.tensor(0.0, device=x.device)
    grad_class = torch.zeros_like(a)
    if labels is not None:
        logits = classify(a)
        class_loss, probs, y_onehot, _, grad_logits_per_sample = manual_softmax_cross_entropy(logits, labels)
        grad_class = CLASS_WEIGHT * (grad_logits_per_sample @ W_cls.t()).view(B, 1, IMG, IMG)

    grad_a = grad_pred + grad_recon + grad_class

    stats = {
        "class_loss": float(class_loss.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "pred_loss": float(pred_loss.detach().cpu()),
    }
    return grad_a, stats


def infer(x, labels=None):
    """
    Inner PC + fluid loop.

    Training: labels are supplied, so classification error pushes hidden density.
    Testing: labels are None, so hidden density is inferred from image/reconstruction only.
    """
    a = image_to_density(x)

    last_stats = {"class_loss": 0.0, "recon_loss": 0.0, "pred_loss": 0.0}

    for _ in range(STEPS):
        # 1. PC reaction: manual hidden-state gradient, no autograd.grad.
        grad_a, last_stats = manual_hidden_gradient(a, x, labels)
        a = normalize_density(a - REACTION_LR * grad_a)

        # 2. Fluid advection: move density by divergence-free-ish velocity.
        u = make_velocity(a, x)
        a = advect(a, u)
        a = a + DT * DIFFUSION * laplacian(a)
        a = normalize_density(a)

    return a, last_stats


def manual_parameter_update(a, x, labels):
    """
    Manual local-ish parameter updates, no loss.backward and no optimizer.step.

    This updates:
      W_cls, b_cls from CE output error
      W_dec, b_dec from reconstruction error
      W_pred, b_pred from hidden prediction error

    W_stream is intentionally not updated here because a true manual gradient through
    advection/normalization is much more complex. This file is for learning the PC
    math without backprop. You can later add a separate flow-control rule for W_stream.
    """
    global W_cls, b_cls, W_dec, b_dec, W_pred, b_pred

    B = x.size(0)
    n_total = B * IMG * IMG

    # ----- Classifier update -----
    logits = classify(a)
    class_loss, probs, y_onehot, grad_logits_mean, _ = manual_softmax_cross_entropy(logits, labels)
    a_flat = a.flatten(1)
    dW_cls = a_flat.t() @ grad_logits_mean
    db_cls = grad_logits_mean.sum(dim=0)

    # ----- Decoder update -----
    pre_x, x_hat = reconstruct_with_pre(a)
    eps_x = x_hat - x
    # E_recon = 0.5 * mean((x_hat - x)^2)
    grad_pre_x = RECON_WEIGHT * eps_x * x_hat * (1.0 - x_hat) / n_total
    dW_dec = conv2d_weight_grad(a, grad_pre_x, kernel_size=5, padding=2)
    db_dec = grad_pre_x.sum(dim=(0, 2, 3))
    recon_loss = 0.5 * eps_x.pow(2).mean()

    # ----- Density predictor update -----
    pre_a, a_hat = predict_density_from_x(x)
    eps_pred = a_hat - a
    # E_pred = 0.5 * mean((a_hat - a)^2)
    grad_pre_a = PRED_WEIGHT * eps_pred * a_hat * (1.0 - a_hat) / n_total
    dW_pred = conv2d_weight_grad(x, grad_pre_a, kernel_size=3, padding=1)
    db_pred = grad_pre_a.sum(dim=(0, 2, 3))
    pred_loss = 0.5 * eps_pred.pow(2).mean()

    with torch.no_grad():
        W_cls -= LR * dW_cls
        b_cls -= LR * db_cls
        W_dec -= LR * dW_dec
        b_dec -= LR * db_dec
        W_pred -= LR * dW_pred
        b_pred -= LR * db_pred

    total_loss = float((class_loss + RECON_WEIGHT * recon_loss + PRED_WEIGHT * pred_loss).detach().cpu())
    return total_loss, float(class_loss.detach().cpu()), float(recon_loss.detach().cpu()), float(pred_loss.detach().cpu())


def train_batch(images, labels):
    x = images.to(DEVICE)
    y = labels.to(DEVICE)

    # PC inference with label clamped. No optimizer.zero_grad, no loss.backward.
    a, infer_stats = infer(x, labels=y)

    # Manual parameter updates after inference equilibrium.
    loss, class_loss, recon_loss, pred_loss = manual_parameter_update(a.detach(), x, y)

    with torch.no_grad():
        logits = classify(a)
        acc = (logits.argmax(dim=1) == y).float().mean().item() * 100

    return loss, class_loss, recon_loss, pred_loss, acc


@torch.no_grad()
def evaluate(loader):
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        x = images.to(DEVICE)
        y = labels.to(DEVICE)

        # Test: label is not clamped, so labels=None.
        a, _ = infer(x, labels=None)
        logits = classify(a)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += y.numel()

    return 100.0 * total_correct / total_seen


def make_loader(train, subset):
    data = datasets.MNIST("./data", train=train, download=True, transform=transforms.ToTensor())
    if subset > 0:
        data = Subset(data, list(range(subset)))
    return DataLoader(data, batch_size=BATCH, shuffle=train)


def smoke_test():
    x = torch.rand(8, 1, IMG, IMG, device=DEVICE)
    y = torch.randint(0, 10, (8,), device=DEVICE)
    loss, class_loss, recon_loss, pred_loss, acc = train_batch(x, y)
    print("smoke test passed")
    print(f"loss={loss:.4f} class={class_loss:.4f} recon={recon_loss:.4f} pred={pred_loss:.4f} acc={acc:.1f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--train_subset", type=int, default=TRAIN_SUBSET)
    p.add_argument("--test_subset", type=int, default=TEST_SUBSET)
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("Manual Fluid Predictive Coding MNIST")
    print(f"device={DEVICE}")
    print("No torch.autograd.grad, no loss.backward, no optimizer.step")

    if args.smoke_test:
        smoke_test()
        return

    train_loader = make_loader(train=True, subset=args.train_subset)
    test_loader = make_loader(train=False, subset=args.test_subset)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, (images, labels) in enumerate(train_loader):
            loss, class_loss, recon_loss, pred_loss, acc = train_batch(images, labels)

            if batch_idx % 20 == 0:
                print(
                    f"epoch={epoch} batch={batch_idx:03d} "
                    f"loss={loss:.4f} class={class_loss:.4f} "
                    f"recon={recon_loss:.4f} pred={pred_loss:.4f} acc={acc:.1f}%"
                )

        test_acc = evaluate(test_loader)
        print(f"epoch={epoch} test_acc={test_acc:.2f}%")


if __name__ == "__main__":
    main()
