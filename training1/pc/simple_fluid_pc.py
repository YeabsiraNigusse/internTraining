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

# Weight optimizer learning rate.
LR = 1e-3


# How fluid PC is different from normal PC:
# Normal PC only changes the hidden state by prediction-error gradient descent:
#     h = h - lr * dE/dh
#
# Fluid PC still does that PC error-correction step, but its hidden state is a
# 2D activation density a, shaped like an image: [batch, 1, 28, 28].
# After the PC step, a learned velocity field u moves that density across the
# grid using advection, like a simple fluid:
#     reaction:  a = a - lr * dE/da
#     advection: a = a - dt * div(a * u)
#
# So the simplest mental model is:
#     normal PC = hidden state corrects its prediction errors
#     fluid PC  = hidden density corrects errors AND flows across space


def normalize_density(a, eps=1e-8):
    """Keep hidden activation positive and make each image's mass sum to 1."""
    a = F.softplus(a) # a smooth version of ReLU, It makes every value positive.
    mass = a.sum(dim=(1, 2, 3), keepdim=True)
    return a / (mass + eps)



def image_to_density(x, eps=1e-6):
    """Use the input image as the first hidden density."""
    a = x.clamp_min(0.0) + eps
    return a / a.sum(dim=(1, 2, 3), keepdim=True)

# we are doing shift to the left or right depending on the sign of the parameter
def roll_x(z, shift):
    return torch.roll(z, shifts=shift, dims=-1)

# up or down shift 
def roll_y(z, shift):
    return torch.roll(z, shifts=shift, dims=-2)


# we need them cuz of the following computation need

# spatial derivative
# divergence
# velocity gradient
# pressure gradient
# laplacian
# advection

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
    """
    Turn one learned image psi into a 2-channel velocity field u.

    This is the simplest way to make a fluid-like velocity in 2D:
        u_x =  d psi / dy
        u_y = -d psi / dx

    The full paper-like file also has an FFT projection step. This simple file
    skips it because curl(psi) is already the easy study version of an
    approximately divergence-free velocity.
    """
    ux = ddy(psi)
    uy = -ddx(psi)
    return torch.cat([ux, uy], dim=1)


def scale_velocity(u):
    """Keep the fluid step small enough that density does not jump too far."""
    max_speed = u.abs().amax(dim=(1, 2, 3), keepdim=True)
    scale = TARGET_CFL / (DT * max_speed + 1e-8)
    return u * scale.clamp(max=10.0)


def advect(a, u):
    """
    Move density a through the velocity field u.

    This implements the simple conservation equation:
        a_new = a - dt * div(a * u)

    Upwind flux means: when flow crosses a cell edge, use the density from the
    side the flow is coming from.
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
    return a - DT * div_flux


# Tiny trainable pieces.
# These are intentionally raw tensors, like simple_pc.py, so the math is easy
# to inspect.
W_pred = (0.02 * torch.randn(1, 2, 3, 3, device=DEVICE)).requires_grad_()
b_pred = torch.zeros(1, device=DEVICE, requires_grad=True)

W_dec = (0.02 * torch.randn(1, 1, 5, 5, device=DEVICE)).requires_grad_()
b_dec = torch.zeros(1, device=DEVICE, requires_grad=True)

W_stream = (0.02 * torch.randn(1, 2, 3, 3, device=DEVICE)).requires_grad_()
b_stream = torch.zeros(1, device=DEVICE, requires_grad=True)

W_cls = (0.02 * torch.randn(IMG * IMG, 10, device=DEVICE)).requires_grad_()
b_cls = torch.zeros(10, device=DEVICE, requires_grad=True)

PARAMS = [W_pred, b_pred, W_dec, b_dec, W_stream, b_stream, W_cls, b_cls]


def predict_density(a, x):
    # density(a) will be constracted directly from the image
    
    """Predict what the hidden density should look like locally."""
    raw = F.conv2d(torch.cat([a, x], dim=1), W_pred, b_pred, padding=1)
    return normalize_density(raw)


def reconstruct(a):
    """Decode hidden density back into an image."""
    return torch.sigmoid(F.conv2d(a, W_dec, b_dec, padding=2))


def classify(a):
    """Classify using the final hidden density."""
    return a.flatten(1) @ W_cls + b_cls


def make_velocity(a, x):
    """Learn a stream image from [hidden density, input image], then curl it."""
    psi = F.conv2d(torch.cat([a, x], dim=1), W_stream, b_stream, padding=1)
    return scale_velocity(velocity_from_stream(psi))


def pc_reaction_energy(a, x):
    """
    The normal predictive-coding part.

    a_hat asks: what hidden density should local neighborhoods predict?
    x_hat asks: what image does the hidden density reconstruct?
    """
    a_hat = predict_density(a, x)
    x_hat = reconstruct(a)
    return 0.20 * F.mse_loss(a, a_hat) + 0.20 * F.mse_loss(x_hat, x)


def infer(x):
    """
    Inner loop for one batch.

    Each step has:
        1. PC reaction: change a to reduce prediction errors.
        2. Fluid advection: move a through a learned velocity field.
        3. Normalize: keep a as a valid density.
    """
    a = image_to_density(x)

    for _ in range(STEPS):
        # Normal PC part: update hidden state by gradient descent on PC energy.
        a = a.detach().requires_grad_(True) # can this function update the hidden state 
        energy = pc_reaction_energy(a, x)
        (grad_a,) = torch.autograd.grad(energy, a) # are you sure can this simple function do accurate gradient of density
                                                   # tbh i need this to be manual calcualtion
        with torch.no_grad():
            a = normalize_density(a - REACTION_LR * grad_a) # normalization after gradient fix

        # Fluid part: velocity moves the hidden density across the 2D grid.
        u = make_velocity(a, x) # idk this is somehow the idea of stream function and HJB optimal transport happen
        a = advect(a, u) # additional advectional change on density
        a = a + DT * DIFFUSION * laplacian(a) # adding somekind of divergence to the density
        a = normalize_density(a) # then normalize the density again

    return a


def train_batch(images, labels, optimizer):
    x = images.to(DEVICE)
    y = labels.to(DEVICE)

    optimizer.zero_grad(set_to_none=True) # i am not sure why exactly i need this

    a = infer(x) # we get trained density value according to IFN(incompressible fluid network)
    logits = classify(a) # is this target pridiction
    x_hat = reconstruct(a) # this is image pridiction or reconstruction
    a_hat = predict_density(a, x) # this is density pridiction itself

    class_loss = F.cross_entropy(logits, y) # i need this to be manual calulation rather than being corss_entropy call
                                            # and do we exactlly add the cross entropy in pc weight optimization??
    recon_loss = F.mse_loss(x_hat, x)
    pred_loss = F.mse_loss(a, a_hat)
    loss = class_loss + 0.10 * recon_loss + 0.05 * pred_loss

    loss.backward() # what the hell is backward for, i really need this to be manual calculation
    optimizer.step() # i need this optimization to be adam, idk if HJB can be used as optimizer
                     # 

    with torch.no_grad(): # idk what this block of code doing
        acc = (logits.argmax(dim=1) == y).float().mean().item() * 100

    return loss.item(), class_loss.item(), recon_loss.item(), acc


def evaluate(loader):
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        x = images.to(DEVICE)
        y = labels.to(DEVICE)

        # Inference needs gradients with respect to hidden density a.
        with torch.enable_grad():
            a = infer(x)

        with torch.no_grad():
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
    """Run without downloading MNIST, just to check shapes and gradients."""
    optimizer = torch.optim.Adam(PARAMS, lr=LR)
    x = torch.rand(8, 1, IMG, IMG)
    y = torch.randint(0, 10, (8,))
    loss, class_loss, recon_loss, acc = train_batch(x, y, optimizer)
    print("smoke test passed")
    print(f"loss={loss:.4f} class={class_loss:.4f} recon={recon_loss:.4f} acc={acc:.1f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--train_subset", type=int, default=TRAIN_SUBSET)
    p.add_argument("--test_subset", type=int, default=TEST_SUBSET)
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args()



def inspect_one_image_density(train_loader):
    """
    Take one MNIST image, print its pixel values,
    convert it to density, and print the result.
    """

    images, labels = next(iter(train_loader))

    # Keep batch dimension: [1, 1, 28, 28]
    x = images[0:1].to(DEVICE)
    label = labels[0].item()

    print("=" * 70)
    print("Inspecting one MNIST image -> density")
    print("=" * 70)

    print(f"Label: {label}")
    print(f"x shape: {x.shape}")

    print()
    print("Original image stats")
    print(f"min pixel value: {x.min().item():.6f}")
    print(f"max pixel value: {x.max().item():.6f}")
    print(f"sum of pixels  : {x.sum().item():.6f}")

    # Convert image to density
    a = image_to_density(x)

    print()
    print("Density stats after image_to_density(x)")
    print(f"a shape: {a.shape}")
    print(f"min density value: {a.min().item():.10f}")
    print(f"max density value: {a.max().item():.10f}")
    print(f"sum of density   : {a.sum().item():.10f}")

    # Show a small 8x8 crop so it is readable
    row_start, row_end = 10, 18
    col_start, col_end = 10, 18

    print()
    print("Original pixel crop x[10:18, 10:18]")
    print(x[0, 0, row_start:row_end, col_start:col_end].cpu())

    print()
    print("Density crop a[10:18, 10:18]")
    print(a[0, 0, row_start:row_end, col_start:col_end].cpu())

    print()
    print("Check mass with sum(dim=(1, 2, 3), keepdim=True)")
    mass = a.sum(dim=(1, 2, 3), keepdim=True)
    print(f"mass shape: {mass.shape}")
    print(f"mass value: {mass}")

    print("=" * 70)


def main():
    args = parse_args()

    print("Simple Fluid Predictive Coding MNIST")
    print(f"device={DEVICE}")

    if args.smoke_test:
        smoke_test()
        return

    optimizer = torch.optim.Adam(PARAMS, lr=LR)
    train_loader = make_loader(train=True, subset=args.train_subset)
    test_loader = make_loader(train=False, subset=args.test_subset)

    # inspect_one_image_density(train_loader)

    for epoch in range(1, args.epochs + 1):
        for batch_idx, (images, labels) in enumerate(train_loader):
            loss, class_loss, recon_loss, acc = train_batch(images, labels, optimizer)

            if batch_idx % 20 == 0:
                print(
                    f"epoch={epoch} batch={batch_idx:03d} "
                    f"loss={loss:.4f} class={class_loss:.4f} "
                    f"recon={recon_loss:.4f} acc={acc:.1f}%"
                )

        test_acc = evaluate(test_loader)
        print(f"epoch={epoch} test_acc={test_acc:.2f}%")

    


if __name__ == "__main__":
    main()



# things i belive should be fixed

# 1 increase the amount of train data
# 2 update the gradient calculation both for the wight and hidden state
