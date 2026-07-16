import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH = 128
EPOCHS = 3
HIDDEN = 128
INFER_STEPS = 20
H_LR = 0.2
W_LR = 0.02

train_data = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.MNIST(
    "./data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH, shuffle=False)

# x -> h
W_x = torch.randn(784, HIDDEN, device=DEVICE) / 784**0.5
b_h = torch.zeros(HIDDEN, device=DEVICE)

# h -> class logits
W_y = torch.randn(HIDDEN, 10, device=DEVICE) / HIDDEN**0.5
b_y = torch.zeros(10, device=DEVICE)


def softmax_manual(logits):
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)


def cross_entropy_manual(probs, y_onehot):
    return -(y_onehot * torch.log(probs + 1e-8)).sum(dim=1).mean()


def infer_hidden(x, y):
    """
    Discriminative PC inference with cross-entropy output energy.

    x is clamped.
    y is clamped.
    h is inferred.
    """

    h_hat = x @ W_x + b_h
    h = h_hat.clone()

    for _ in range(INFER_STEPS):
        # hidden prediction from image
        h_hat = x @ W_x + b_h

        # class prediction from hidden
        logits = h @ W_y + b_y
        probs = softmax_manual(logits)

        # local errors
        eps_h = h - h_hat

        # cross-entropy gradient wrt logits
        grad_logits = probs - y

        # hidden gradient:
        # E = hidden MSE + class CE
        grad_h = eps_h / HIDDEN + grad_logits @ W_y.t()

        h = (h - H_LR * grad_h).clamp(-5, 5)

    return h


def train_one_epoch(epoch):
    global W_x, b_h, W_y, b_y

    total_correct = 0
    total_seen = 0
    total_ce = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        x = images.view(images.size(0), -1).to(DEVICE)
        y = F.one_hot(labels.to(DEVICE), 10).float()

        batch_size = x.size(0)

        # -----------------------------
        # 1. PC inference phase
        # -----------------------------
        h = infer_hidden(x, y)

        # -----------------------------
        # 2. Recompute final errors
        # -----------------------------
        h_hat = x @ W_x + b_h
        logits = h @ W_y + b_y
        probs = softmax_manual(logits)

        eps_h = h - h_hat

        ce = cross_entropy_manual(probs, y)

        # manual cross-entropy gradient
        grad_logits = probs - y

        # -----------------------------
        # 3. Manual PC weight updates
        # -----------------------------

        # x -> h update
        # hidden prediction error says:
        # make x predict inferred h better
        W_x += W_LR * (x.t() @ eps_h) / batch_size
        b_h += W_LR * eps_h.mean(dim=0)

        # h -> class update with cross-entropy
        # gradient descent:
        # W_y -= lr * h.T @ (probs - y)
        # equivalent:
        # W_y += lr * h.T @ (y - probs)
        W_y += W_LR * (h.t() @ (y - probs)) / batch_size
        b_y += W_LR * (y - probs).mean(dim=0)

        # -----------------------------
        # 4. Accuracy
        # -----------------------------
        preds = probs.argmax(dim=1)
        total_correct += (preds == labels.to(DEVICE)).sum().item()
        total_seen += labels.size(0)
        total_ce += ce.item() * batch_size

        if batch_idx % 100 == 0:
            acc = 100.0 * total_correct / total_seen
            avg_ce = total_ce / total_seen

            print(
                f"epoch={epoch} batch={batch_idx:04d} "
                f"ce={avg_ce:.4f} "
                f"train_acc={acc:.2f}%"
            )

    return total_ce / total_seen, 100.0 * total_correct / total_seen


@torch.no_grad()
def test():
    """
    Testing for discriminative PCN.

    During testing, y is unknown.
    So we just do feedforward:

        x -> h -> logits
    """

    total_correct = 0
    total_seen = 0

    for images, labels in test_loader:
        x = images.view(images.size(0), -1).to(DEVICE)

        h = x @ W_x + b_h
        logits = h @ W_y + b_y
        probs = softmax_manual(logits)

        preds = probs.argmax(dim=1)

        total_correct += (preds == labels.to(DEVICE)).sum().item()
        total_seen += labels.size(0)

    return 100.0 * total_correct / total_seen


for epoch in range(1, EPOCHS + 1):
    train_ce, train_acc = train_one_epoch(epoch)
    test_acc = test()

    print("=" * 70)
    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train CE: {train_ce:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Test Acc: {test_acc:.2f}%"
    )
    print("=" * 70)