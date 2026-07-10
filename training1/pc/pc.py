import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.manual_seed(7)

# ============================================================
# 1. Finite differences on a 2-D grid
# ============================================================

def center(f):
    # f: [B, H, W], return interior cells only
    return f[..., 1:-1, 1:-1]

def ddx(f, dx):
    # central difference in x direction
    return (f[..., 1:-1, 2:] - f[..., 1:-1, :-2]) / (2.0 * dx)

def ddy(f, dy):
    # central difference in y direction
    return (f[..., 2:, 1:-1] - f[..., :-2, 1:-1]) / (2.0 * dy)

def laplacian(f, dx, dy):
    return (
        (f[..., 1:-1, 2:] - 2.0 * center(f) + f[..., 1:-1, :-2]) / dx**2
        +
        (f[..., 2:, 1:-1] - 2.0 * center(f) + f[..., :-2, 1:-1]) / dy**2
    )

# ============================================================
# 2. Navier-Stokes residual on hidden field
# ============================================================

def navier_stokes_losses(field, dx, dy, nu=0.01):
    """
    field: [B, 3, H, W]
        channel 0 = u_x
        channel 1 = u_y
        channel 2 = pressure p

    Steady incompressible Navier-Stokes, density rho=1, no external force:

        (u · grad)u + grad p - nu * Laplacian(u) = 0
        div(u) = 0

    The residual is computed at interior grid cells.
    """
    ux = field[:, 0]
    uy = field[:, 1]
    p  = field[:, 2]

    ux_c = center(ux)
    uy_c = center(uy)

    dux_dx = ddx(ux, dx)
    dux_dy = ddy(ux, dy)
    duy_dx = ddx(uy, dx)
    duy_dy = ddy(uy, dy)

    dp_dx = ddx(p, dx)
    dp_dy = ddy(p, dy)

    lap_ux = laplacian(ux, dx, dy)
    lap_uy = laplacian(uy, dx, dy)

    # Momentum residual:
    #
    #     (u · grad)u + grad p - nu * Laplacian(u)
    #
    r_x = ux_c * dux_dx + uy_c * dux_dy + dp_dx - nu * lap_ux
    r_y = ux_c * duy_dx + uy_c * duy_dy + dp_dy - nu * lap_uy

    # Incompressibility / mass conservation residual:
    #
    #     div(u) = du_x/dx + du_y/dy
    #
    r_div = dux_dx + duy_dy

    momentum_loss = (r_x.pow(2) + r_y.pow(2)).mean()
    divergence_loss = r_div.pow(2).mean()

    return momentum_loss, divergence_loss


def pc_energy(field, observed, mask, dx, dy,
              lambda_mom=0.05,
              lambda_div=0.50,
              nu=0.01):
    """
    Predictive-coding energy + physics energy.

    observed/mask: [B, 3, H, W]

    mask=1 means the channel/cell is observed.
    mask=0 means unobserved.
    """
    prediction_error_loss = (
        ((field - observed) * mask).pow(2).sum()
        / (mask.sum() + 1e-8)
    )

    momentum_loss, divergence_loss = navier_stokes_losses(field, dx, dy, nu)

    energy = (
        prediction_error_loss
        + lambda_mom * momentum_loss
        + lambda_div * divergence_loss
    )

    parts = {
        "prediction": prediction_error_loss.detach(),
        "momentum": momentum_loss.detach(),
        "divergence": divergence_loss.detach(),
    }

    return energy, parts

# ============================================================
# 3. Slow learnable prior network: coordinates -> initial field
# ============================================================

class FieldPrior(nn.Module):
    """
    A tiny network that maps each coordinate (x,y) to [u_x, u_y, p].

    This is the slow part whose weights are learned.

    The PC inference loop refines this network's first guess.
    The learning loop then trains the network to make better first guesses.
    """
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 3),
        )

    def forward(self, coords, H, W):
        out = self.net(coords)              # [H*W, 3]
        return out.T.reshape(1, 3, H, W)    # [B=1, C=3, H, W]

# ============================================================
# 4. Optional toy HJB value function
# ============================================================

class ValueNet(nn.Module):
    """
    Toy value function V(s), where local state s = [u_x, u_y, p].

    This is not required for Navier-Stokes PC.
    It is included to show exactly where an HJB residual would be placed.
    """
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def toy_hjb_loss(value_net, field, observed, mask, r=1.0):
    """
    Stationary HJB for a toy local control problem:

        ds/dtau = a
        running cost c(s) = local prediction error
        control cost = 0.5 * r * ||a||^2

    HJB:

        0 = min_a [ c(s) + grad V(s) · a + 0.5*r*||a||^2 ]

    The minimizing action is:

        a* = -grad V(s) / r

    Substituting a* gives residual:

        R_HJB = c(s) - 0.5/r * ||grad V(s)||^2

    This demonstrates HJB mechanics.

    A real HJB fluid-control problem would define:
        - full grid state
        - physical dynamics
        - real controls
        - boundary forcing
        - terminal cost
    """
    # [B, 3, H, W] -> [B*H*W, 3]
    state = field.detach().permute(0, 2, 3, 1).reshape(-1, 3).clone()
    state.requires_grad_(True)

    obs_flat = observed.detach().permute(0, 2, 3, 1).reshape(-1, 3)
    mask_flat = mask.detach().permute(0, 2, 3, 1).reshape(-1, 3)

    running_cost = ((state - obs_flat) * mask_flat).pow(2).sum(dim=1).detach()

    V_sum = value_net(state).sum()
    grad_V = torch.autograd.grad(V_sum, state, create_graph=True)[0]

    residual = running_cost - 0.5 / r * grad_V.pow(2).sum(dim=1)

    return residual.pow(2).mean()

# ============================================================
# 5. Create a synthetic fluid grid
# ============================================================

H = W = 16

x = torch.linspace(-1.0, 1.0, W)
y = torch.linspace(-1.0, 1.0, H)

dx = float(x[1] - x[0])
dy = float(y[1] - y[0])

Y, X = torch.meshgrid(y, x, indexing="ij")

coords = torch.stack(
    [X.reshape(-1), Y.reshape(-1)],
    dim=1
)

# A simple exact steady rotating flow:
#
#     u_x = -y
#     u_y =  x
#     p   = 0.5 * (x^2 + y^2)
#
# This has div(u)=0.
# It also satisfies the steady momentum equation because
# the pressure gradient balances the circular acceleration.
#
true_field = torch.stack(
    [
        -Y,
        X,
        0.5 * (X**2 + Y**2)
    ],
    dim=0
).unsqueeze(0)  # [1, 3, H, W]

# Noisy observations:
#   - observe velocities everywhere
#   - observe pressure only at 10% of cells
#
observed = true_field + 0.05 * torch.randn_like(true_field)

mask = torch.zeros_like(true_field)
mask[:, 0:2] = 1.0
mask[:, 2] = (torch.rand(1, H, W) < 0.10).float()

# ============================================================
# 6. Two-loop predictive coding training
# ============================================================

prior = FieldPrior(width=32)
value_net = ValueNet(width=32)

weight_optimizer = torch.optim.Adam(prior.parameters(), lr=3e-3)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

outer_steps = 12
inner_inference_steps = 20

for outer in range(outer_steps + 1):

    # ------------------------------------------------------------
    # Fast PC inference loop:
    #
    # weights are frozen
    # hidden field is optimized
    # ------------------------------------------------------------
    with torch.no_grad():
        first_guess = prior(coords, H, W)

    hidden = first_guess.detach().clone().requires_grad_(True)
    hidden_optimizer = torch.optim.Adam([hidden], lr=5e-2)

    for inner in range(inner_inference_steps):
        hidden_optimizer.zero_grad()

        energy, parts = pc_energy(
            hidden,
            observed,
            mask,
            dx,
            dy
        )

        energy.backward()
        hidden_optimizer.step()

    # ------------------------------------------------------------
    # Slow learning loop:
    #
    # train the prior weights so the next first_guess is closer to
    # the settled hidden state
    # ------------------------------------------------------------
    weight_optimizer.zero_grad()

    next_guess = prior(coords, H, W)

    weight_loss = F.mse_loss(
        next_guess,
        hidden.detach()
    )

    weight_loss.backward()
    weight_optimizer.step()

    # ------------------------------------------------------------
    # Optional HJB learning:
    #
    # train a value function on the settled hidden state.
    #
    # This does NOT magically make PC solve HJB.
    # It is an added value-function objective.
    # ------------------------------------------------------------
    value_optimizer.zero_grad()

    hjb = toy_hjb_loss(
        value_net,
        hidden.detach(),
        observed,
        mask
    )

    hjb.backward()
    value_optimizer.step()

    if outer % 4 == 0:
        with torch.no_grad():
            settled_energy, settled_parts = pc_energy(
                hidden,
                observed,
                mask,
                dx,
                dy
            )

            mse_to_true = F.mse_loss(
                hidden,
                true_field
            )

        print(
            f"outer={outer:02d} | "
            f"settled_E={settled_energy.item():.5f} | "
            f"MSE_to_true={mse_to_true.item():.5f} | "
            f"pred={settled_parts['prediction'].item():.5f} | "
            f"mom={settled_parts['momentum'].item():.5f} | "
            f"div={settled_parts['divergence'].item():.5f} | "
            f"HJB_toy={hjb.item():.5f}"
        )

print("\nFinal hidden field shape:", tuple(hidden.shape))
print("Channel meanings: hidden[:,0]=u_x, hidden[:,1]=u_y, hidden[:,2]=pressure")

