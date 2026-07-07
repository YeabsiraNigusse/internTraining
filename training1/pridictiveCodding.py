

# | Backpropagation             | Predictive Coding           |
# | --------------------------- | --------------------------- |
# | Gradient                    | Prediction error            |
# | Backward pass               | Error propagation           |
# | Fixed activations           | Optimized activations       |
# | One forward pass            | Iterative inference         |
# | Error exists only at output | Error exists at every layer |


# ------------------------------------------
# Step 1: Initialize latent variables
# ------------------------------------------
hidden = initial_guess()

# ------------------------------------------
# Step 2: Inference loop
# Optimize the hidden state while keeping
# the weights fixed.
# ------------------------------------------
for _ in range(num_inference_steps):

    # Predict the input from the hidden state
    predicted_input = W1 @ hidden

    # Predict the hidden state from the output
    predicted_hidden = W2 @ output

    # Compute prediction errors
    eps_input = input - predicted_input
    eps_hidden = hidden - predicted_hidden

    # Compute gradient of the energy
    grad_hidden = -W1.T @ eps_input + eps_hidden

    # Update the hidden state
    hidden = hidden - eta_inference * grad_hidden

# ------------------------------------------
# Step 3: Weight learning
# Once inference converges, update weights.
# ------------------------------------------
W1 += eta_weight * eps_input @ hidden.T
W2 += eta_weight * eps_hidden @ output.T