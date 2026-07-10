# | x₁ | x₂ | y |
# | -- | -- | - |
# | 0  | 0  | 0 |
# | 0  | 1  | 1 |
# | 1  | 0  | 1 |
# | 1  | 1  | 1 |

import torch
import torch.nn as nn

# ----------------------------
# Tiny OR dataset
# ----------------------------

X = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])

Y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [1.]
])


# ------------------------------------
# Weight matrix:
#
# 2 inputs
#
# ↓
#
# 2 hidden neurons
# ------------------------------------

W1 = nn.Parameter(torch.randn(2,2)*0.2)
print(W1.shape)

# ------------------------------------
# Weight matrix:
#
# 2 hidden neurons
#
# ↓
#
# 1 output neuron
# ------------------------------------

W2 = nn.Parameter(torch.randn(2,1)*0.2)


# -----------------------------------------
# Initial hidden-state guess
#
# One hidden vector for every training
# sample.
#
# Shape:
#
# (4 samples, 2 hidden neurons)
# -----------------------------------------

hidden = torch.zeros(4,2)

# --------------------------------------
# Hidden predicts the input
#
# hidden
#
# ↓
#
# predicted_input
# --------------------------------------

def predict_input(hidden):

    return hidden @ W1.T


# -------------------------------------
# Output predicts hidden
#
# output
#
# ↓
#
# predicted_hidden
# -------------------------------------

def predict_hidden(output):

    return output @ W2.T


# ------------------------------------
# Compute prediction errors
# ------------------------------------

predicted_input = predict_input(hidden)

epsilon_input = X - predicted_input


# ------------------------------------
# Predict hidden state
# ------------------------------------

output = hidden @ W2

predicted_hidden = predict_hidden(output)

epsilon_hidden = hidden - predicted_hidden

print("Actual Input")
print(X)

print()

print("Predicted Input")
print(predicted_input)

print()

print("Input Error")
print(epsilon_input)


# -------------------------------------
# Energy Function
# -------------------------------------

energy = 0.5*(
    epsilon_input.pow(2).sum()
    +
    epsilon_hidden.pow(2).sum()
)


# -------------------------------------
# Hidden states become variables
# -------------------------------------


