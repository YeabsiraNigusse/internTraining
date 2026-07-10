import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128
EPOCHS = 3
STEPS = 8
H_LR = 0.25
W_LR = 1e-3
NS_W = 0.02


def ns_loss(h):
    ux, uy, p = h[:, 0], h[:, 1], h[:, 2]
    ux_c, uy_c = ux[:, 1:-1, 1:-1], uy[:, 1:-1, 1:-1]
    dux_dx = (ux[:, 1:-1, 2:] - ux[:, 1:-1, :-2]) / 2
    dux_dy = (ux[:, 2:, 1:-1] - ux[:, :-2, 1:-1]) / 2
    duy_dx = (uy[:, 1:-1, 2:] - uy[:, 1:-1, :-2]) / 2
    duy_dy = (uy[:, 2:, 1:-1] - uy[:, :-2, 1:-1]) / 2
    dp_dx = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / 2
    dp_dy = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) / 2
    lap_ux = ux[:, 1:-1, 2:] + ux[:, 1:-1, :-2] + ux[:, 2:, 1:-1] + ux[:, :-2, 1:-1] - 4 * ux_c
    lap_uy = uy[:, 1:-1, 2:] + uy[:, 1:-1, :-2] + uy[:, 2:, 1:-1] + uy[:, :-2, 1:-1] - 4 * uy_c
    rx = ux_c * dux_dx + uy_c * dux_dy + dp_dx - 0.01 * lap_ux
    ry = ux_c * duy_dx + uy_c * duy_dy + dp_dy - 0.01 * lap_uy
    div = dux_dx + duy_dy
    return (rx.square() + ry.square()).mean() + 5 * div.square().mean()


train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)

W_i = (0.1 * torch.randn(3, 1, 5, 5, device=DEVICE)).requires_grad_()
b_i = torch.zeros(3, device=DEVICE, requires_grad=True)
W_d = (0.1 * torch.randn(1, 3, 5, 5, device=DEVICE)).requires_grad_()
b_d = torch.zeros(1, device=DEVICE, requires_grad=True)
W_c = (0.02 * torch.randn(3 * 28 * 28, 10, device=DEVICE)).requires_grad_()
b_c = torch.zeros(10, device=DEVICE, requires_grad=True)
params = [W_i, b_i, W_d, b_d, W_c, b_c]

for _ in range(EPOCHS):
    for images, labels in train_loader:
        x = images.to(DEVICE)
        y = labels.to(DEVICE)

        h0 = torch.tanh(F.conv2d(x, W_i, b_i, padding=2)).detach()
        h = h0.clone().requires_grad_(True)

        for _ in range(STEPS):
            x_hat = torch.sigmoid(F.conv2d(h, W_d, b_d, padding=2))
            logits = h.flatten(1) @ W_c + b_c
            energy = F.mse_loss(x_hat, x) + F.cross_entropy(logits, y)
            energy = energy + 0.1 * F.mse_loss(h, h0) + NS_W * ns_loss(h)
            (dh,) = torch.autograd.grad(energy, h)
            with torch.no_grad():
                h = (h - H_LR * dh).clamp(-5, 5)
            h = h.detach().requires_grad_(True)

        h = h.detach()
        h0 = torch.tanh(F.conv2d(x, W_i, b_i, padding=2))
        x_hat = torch.sigmoid(F.conv2d(h, W_d, b_d, padding=2))
        logits = h.flatten(1) @ W_c + b_c
        loss = F.mse_loss(x_hat, x) + F.cross_entropy(logits, y)
        loss = loss + 0.5 * F.mse_loss(h0, h) + NS_W * ns_loss(h0)

        loss.backward()
        with torch.no_grad():
            for param in params:
                param -= W_LR * param.grad
                param.grad = None
