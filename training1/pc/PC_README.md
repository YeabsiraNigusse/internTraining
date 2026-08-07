# Predictive Coding Notes For This Repo

This note explains how `engine/pc.py` works in this codebase. It is meant as
a revision guide for reading the code line by line.

The most important idea is:

```text
Predictive coding in this repo is generative and top-down.

The input image is z_0.
Higher latent states z_1, z_2, ... predict the layer below them.
```

So the direction is:

```text
z_L -> z_{L-1} -> ... -> z_2 -> z_1 -> z_0
```

not the normal feedforward classifier direction:

```text
x -> hidden1 -> hidden2 -> class_output
```

## 1. What Kind Of PC Is This?

The main `pc.py` implementation is a generative predictive coding model.
It reconstructs the input image.

In `train.py`, the dataloader returns:

```python
for batch_x, batch_y in train_loader:
```

but `batch_y` is not used by `model.forward`. The model receives only:

```python
output = model.forward(batch_x, epoch_config, update=True, prng_key=sub_key)
```

The reported BCE loss is:

```python
bce_loss = compute_binary_cross_entropy(reconstruction, batch_x)
```

So the target is the image `batch_x`, not the class label.

The top latent has dimension 10 in the default config, but it is not trained as
a supervised class output. It is just the top latent state of the generative
hierarchy.

## 2. Active Architecture

The default config in `config.py` is:

```python
"layer_dims": [784, 1568, 1568, 64, 10]
```

That means:

```text
z_0: 784       input image, 28 x 28 flattened
z_1: 1568      hidden spatial state, 8 x 14 x 14 flattened
z_2: 1568      hidden spatial state, 8 x 14 x 14 flattened
z_3: 64        hidden vector
z_4: 10        top latent vector
```

The active prediction chain is:

```text
z_4 --layer_3--> z_3 --layer_2--> z_2 --layer_1--> z_1 --layer_0--> z_0
```

or with predictions:

```text
mu_3 predicts z_3 from z_4
mu_2 predicts z_2 from z_3
mu_1 predicts z_1 from z_2
mu_0 predicts z_0 from z_1
```

The first two prediction layers are convolutional. The upper two are fully
connected.

```text
layer_0: z_1 -> z_0   convolution, predicts image pixels
layer_1: z_2 -> z_1   convolution, predicts lower spatial latent
layer_2: z_3 -> z_2   fully connected
layer_3: z_4 -> z_3   fully connected
```

Important implementation note: `model.py` initializes skip and lateral
parameters, but `engine/pc.py::compute_prediction` does not currently add skip
or lateral terms into the predictions. So the actual active PC prediction path
is the plain chain above.

## 3. Parameters Versus Latent States

There are two different kinds of values:

```text
Parameters:
  W_0, b_0, W_1, b_1, ...
  These are learned across batches.

Latent states:
  z_1, z_2, z_3, ...
  These are temporary hidden states for the current batch.
```

`z_0` is special:

```text
z_0 = input image x
```

It is clamped. The inference loop does not update it.

The hidden states are not initially computed with a feedforward pass like:

```text
z_1 = act(x @ W)
```

Instead they are initialized by `utils/initializer.py`.

The default config uses:

```python
"latent_init": "zeros"
```

So for a batch:

```text
z_0 = batch_x
z_1 = zeros
z_2 = zeros
z_3 = zeros
z_4 = zeros
```

Then inference updates `z_1`, `z_2`, `z_3`, and `z_4` so that the generative
model can explain `z_0`.

## 4. Parameter Initialization

Parameters are created in `model.py::build_network_parameters`.

For the first two layers, the code creates convolution kernels:

```python
if idx == 0:
    W = jax.random.normal(w_key, (1, C1, k, k)) * scale
    b = jnp.zeros((1))
elif idx == 1:
    W = jax.random.normal(w_key, (C1, C2, k, k)) * scale
    b = jnp.zeros((C1))
```

With the default config:

```text
C1 = 8
C2 = 8
k = 3
```

So:

```text
layer_0 W shape: (1, 8, 3, 3)
layer_1 W shape: (8, 8, 3, 3)
```

For later fully connected layers, `utils/initializer.py::initialize_weights`
creates matrix weights:

```python
W = jax.random.normal(w_key, (in_dim, out_dim)) * scale
b = jnp.zeros((out_dim))
```

The comment in `initializer.py` is important:

```text
Weights are shaped for top-down generative projection.
```

That means a fully connected prediction is:

```text
mu_k = z_{k+1} @ W_k + b_k
```

where `W_k` has shape:

```text
parent_dim x child_dim
```

For the current architecture:

```text
layer_2 W shape: (64, 1568)    z_3 -> z_2
layer_3 W shape: (10, 64)      z_4 -> z_3
```

## 5. One Batch: Full High-Level Flow

For a batch of images `batch_x`, `model.forward` does:

```text
1. Initialize z_0, z_1, ..., z_L
2. Clamp z_0 = batch_x
3. Initialize momentum buffers v_0, v_1, ..., v_L to zero
4. Repeat inference for N steps:
   a. Compute top-down predictions mu_k
   b. Compute errors eps_k = z_k - mu_k
   c. Compute latent update directions for z_1, ..., z_L
   d. Update latent states with momentum
5. Recompute reconstruction mu_0
6. If update=True, compute local weight gradients
7. Update weights and biases
```

The inference loop in `model.py` is:

```python
for step in range(inf_steps):
    predictions = compute_prediction(...)
    errors = compute_errors(...)
    z_grads = compute_latent_gradients(...)
    self.z_cache, self.v_dict = run_inference(...)
```

## 6. Prediction Calculation

Prediction is implemented in `engine/pc.py::compute_prediction`.

For each layer index `k`:

```python
z_parent = z_dict[f"z_{k+1}"]
weights = params_dict[f"layer_{k}"]
```

So:

```text
layer_k predicts z_k from z_{k+1}
```

### Fully Connected Case

For a fully connected layer:

```python
mu_linear = jnp.dot(z_parent, W) + b
```

Then:

```python
if is_bottom:
    mu = jax.nn.sigmoid(mu_linear)
else:
    mu = act_fx(mu_linear)
```

So in formula form:

```text
a_k = z_{k+1} @ W_k + b_k

if k == 0:
    mu_0 = sigmoid(a_0)
else:
    mu_k = act(a_k)
```

With default `act_fx = "tanh"`:

```text
mu_1 = tanh(a_1)
mu_2 = tanh(a_2)
mu_3 = tanh(a_3)
```

### Current Repo Predictions

For the current default architecture:

```text
a_0 = conv(z_1, W_0) + b_0
mu_0 = sigmoid(a_0)

a_1 = conv(z_2, W_1) + b_1
mu_1 = tanh(a_1)

a_2 = z_3 @ W_2 + b_2
mu_2 = tanh(a_2)

a_3 = z_4 @ W_3 + b_3
mu_3 = tanh(a_3)
```

## 7. Error Calculation

Error calculation is implemented in `engine/pc.py::compute_errors`.

The code is:

```python
for k in range(num_layers - 1):
    mu = predictions[f"mu_{k}"]
    target = z_dict[f"z_{k}"]
    errors[f"eps_{k}"] = target - mu
```

Formula:

```text
eps_k = z_k - mu_k
```

For the current architecture:

```text
eps_0 = z_0 - mu_0 = x   - prediction_from_z_1
eps_1 = z_1 - mu_1       = z_1 - prediction_from_z_2
eps_2 = z_2 - mu_2       = z_2 - prediction_from_z_3
eps_3 = z_3 - mu_3       = z_3 - prediction_from_z_4
```

There is no `eps_4`, because no layer above `z_4` predicts `z_4`.

## 8. Latent State Updates

Latent gradients are computed in `engine/pc.py::compute_latent_gradients`.

For every prediction relationship:

```text
z_{k+1} predicts z_k
```

the code computes an update direction for the parent latent `z_{k+1}`.

For a fully connected layer, the key code is:

```python
error = errors[f"eps_{k}"]
mu = predictions[f"mu_linear_{k}"]
error = error * d_actfx(mu)
bu_base = jnp.dot(error, W.T)
```

Meaning:

```text
d_k = eps_k * act_prime(a_k)
bottom_up_k = d_k @ W_k.T
```

`bottom_up_k` says:

```text
Change z_{k+1} so it predicts z_k better.
```

Then the code subtracts the error from the layer above:

```python
if k < num_layers - 2:
    td_error = errors[f"eps_{k+1}"]
else:
    td_error = jnp.zeros_like(bu_base)

z_delta = bu_base - td_error
z_grads[f"z_{k+1}"] = z_delta
```

Formula:

```text
delta_z_{k+1} = bottom_up_k - eps_{k+1}
```

If there is no layer above, the second term is zero.

### Why Each Hidden State Uses Two Errors

An internal latent state is both:

```text
1. A parent that predicts the layer below
2. A child that is predicted by the layer above
```

So it receives two pressures:

```text
1. Predict the lower layer better
2. Stay close to the prediction from the upper layer
```

### Update For z_1

`z_1` predicts `z_0`, and `z_1` is predicted by `z_2`.

```text
d_0 = eps_0 * activation_prime(a_0)
bottom_up_0 = d_0 @ W_0.T

delta_z_1 = bottom_up_0 - eps_1
```

In words:

```text
z_1 changes to reconstruct the input better,
but it is also pulled toward what z_2 predicts z_1 should be.
```

### Update For z_2

`z_2` predicts `z_1`, and `z_2` is predicted by `z_3`.

```text
d_1 = eps_1 * activation_prime(a_1)
bottom_up_1 = d_1 @ W_1.T

delta_z_2 = bottom_up_1 - eps_2
```

### Update For z_3

`z_3` predicts `z_2`, and `z_3` is predicted by `z_4`.

```text
d_2 = eps_2 * activation_prime(a_2)
bottom_up_2 = d_2 @ W_2.T

delta_z_3 = bottom_up_2 - eps_3
```

### Update For The Final Top Latent z_4

`z_4` predicts `z_3`, but no higher layer predicts `z_4`.

```text
d_3 = eps_3 * activation_prime(a_3)
bottom_up_3 = d_3 @ W_3.T

delta_z_4 = bottom_up_3
```

So the top latent is corrected only by how well it helps predict the layer
below it.

This is the answer to the common question:

```text
What happens to the final layer?
```

It is not compared to a label in this repo. It moves because its prediction of
`z_3` is wrong.

### Applying The Latent Update

The actual state update happens in `engine/pc.py::run_inference`.

The code skips `z_0`:

```python
if layer_key == "z_0":
    continue
```

Then for every hidden latent:

```python
g = z_grads.get(layer_key, jnp.zeros_like(z))
g = g + l1_latents * jnp.sign(z)
g = jnp.clip(g, -grad_clip, grad_clip)

v_new[layer_key] = polyak_mu * v + inf_lr * g
z_new[layer_key] = z + v_new[layer_key]
```

Formula:

```text
g_k = delta_z_k + L1_term
g_k = clip(g_k)

v_k <- polyak_mu * v_k + inference_lr * g_k
z_k <- z_k + v_k
```

So during inference:

```text
Weights are fixed.
z_0 is fixed.
Only z_1, z_2, ..., z_L are updated.
```

## 9. Weight Gradient Calculation

After inference, if `update=True`, the model updates weights.

Weight gradients are computed in `engine/pc.py::compute_weight_grads`.

For a fully connected layer, the code is:

```python
error = errors[f"eps_{k}"]
z_parent = z_dict[f"z_{k+1}"]
mu = predictions[f"mu_linear_{k}"]

error_deriv = error * d_actfx(mu)

w_grad = jnp.dot(z_parent.T, error_deriv) / batch_size
b_grad = jnp.sum(error_deriv, axis=0) / batch_size
```

Formula:

```text
d_k = eps_k * activation_prime(a_k)

dW_k = z_{k+1}.T @ d_k / batch_size
db_k = sum_batch(d_k) / batch_size
```

This is local. Each weight only uses:

```text
1. The parent latent z_{k+1}
2. The prediction error eps_k for the child it predicts
```

### W_0 Update

`W_0` predicts `z_0` from `z_1`.

```text
a_0 = z_1 @ W_0 + b_0
mu_0 = sigmoid(a_0)
eps_0 = z_0 - mu_0

d_0 = eps_0 * activation_prime(a_0)

dW_0 = z_1.T @ d_0 / batch_size
db_0 = sum_batch(d_0) / batch_size
```

For the actual default architecture, `W_0` is convolutional, so this same idea
is implemented as a convolutional correlation between `z_1` feature maps and
`eps_0` image error maps.

### W_1 Update

`W_1` predicts `z_1` from `z_2`.

```text
a_1 = z_2 @ W_1 + b_1
mu_1 = act(a_1)
eps_1 = z_1 - mu_1

d_1 = eps_1 * activation_prime(a_1)

dW_1 = z_2.T @ d_1 / batch_size
db_1 = sum_batch(d_1) / batch_size
```

For the actual default architecture, `W_1` is also convolutional.

### W_2 Update

`W_2` predicts `z_2` from `z_3`.

```text
a_2 = z_3 @ W_2 + b_2
mu_2 = act(a_2)
eps_2 = z_2 - mu_2

d_2 = eps_2 * activation_prime(a_2)

dW_2 = z_3.T @ d_2 / batch_size
db_2 = sum_batch(d_2) / batch_size
```

### W_3 Update

`W_3` predicts `z_3` from `z_4`.

```text
a_3 = z_4 @ W_3 + b_3
mu_3 = act(a_3)
eps_3 = z_3 - mu_3

d_3 = eps_3 * activation_prime(a_3)

dW_3 = z_4.T @ d_3 / batch_size
db_3 = sum_batch(d_3) / batch_size
```

Again, `W_3` is not updated from class labels in this repo. It is updated from
the local error `eps_3`.

## 10. Applying Weight Updates

Parameter updates happen in `engine/pc.py::update_parameters`.

For normal PC, without causal coding gates:

```python
new_W = W + lr * (dW - decay * W)
new_b = b + lr * db
```

Formula:

```text
W_k <- W_k + lr_pc * (dW_k - weight_decay * W_k)
b_k <- b_k + lr_pc * db_k
```

The sign is plus because:

```text
eps_k = target - prediction
```

With that error convention, `dW_k` points in the direction that improves the
local prediction.

## 11. Difference Between Latent Updates And Weight Updates

This is the key distinction:

```text
Latent update:
  Internal z values use neighboring errors.

Weight update:
  Each W uses only its own local prediction error.
```

For the current architecture:

```text
Latent updates:
  z_1 uses eps_0 and eps_1
  z_2 uses eps_1 and eps_2
  z_3 uses eps_2 and eps_3
  z_4 uses eps_3 only

Weight updates:
  W_0 uses z_1 and eps_0 only
  W_1 uses z_2 and eps_1 only
  W_2 uses z_3 and eps_2 only
  W_3 uses z_4 and eps_3 only
```

This is why predictive coding can look like backpropagation during inference,
but the parameter update itself is local.

## 12. What Is A Convolution?

A convolution is a layer that applies a small shared filter across a spatial
grid.

For an image-like tensor:

```text
batch x channels x height x width
```

a convolution kernel has shape:

```text
output_channels x input_channels x kernel_height x kernel_width
```

For example, `layer_0` has:

```text
W_0 shape = (1, 8, 3, 3)
```

That means:

```text
1 output channel
8 input channels
3 x 3 local spatial window
```

At each output location, the convolution looks at a local `3 x 3` patch from
all 8 input channels and computes a weighted sum.

So a fully connected layer does:

```text
output_dimension_j = weighted sum over all input dimensions
```

but a convolutional layer does:

```text
output_pixel_at_location = weighted sum over nearby input pixels/channels
```

The same small kernel is reused at every spatial location. This is called
weight sharing.

## 13. Why Use Convolution Here?

MNIST images have spatial structure. Nearby pixels matter together.

A fully connected layer from `z_1` to `z_0` would need a separate parameter for
many pairs of latent units and pixels.

A convolution says:

```text
Use the same local pattern detector/predictor everywhere on the image grid.
```

This gives fewer parameters and preserves the spatial layout.

In this repo:

```text
z_1 is stored flat as shape:
  batch x 1568

but it represents:
  batch x 8 x 14 x 14
```

So before applying convolution, `pc.py` reshapes:

```python
z_maps = z_parent.reshape(batch, C_in, H, H)
```

For `layer_0`, the parent `z_1` has spatial size `14 x 14`, but the image is
`28 x 28`. So the code upsamples first:

```python
z_up = jax.image.resize(z_maps, (batch, C_in, H*2, H*2), method='bilinear')
mu_linear = conv2d_same(z_up, W) + b
mu = jax.nn.sigmoid(mu_linear)
```

So:

```text
z_1: 8 x 14 x 14
upsample -> 8 x 28 x 28
conv W_0 -> 1 x 28 x 28
flatten -> 784
```

For `layer_1`:

```text
z_2: 8 x 14 x 14
conv W_1 -> 8 x 14 x 14
flatten -> 1568
```

## 14. Convolution In Error Projection

For fully connected layers, projecting error upward uses:

```python
bu_base = error @ W.T
```

For convolutional layers, there is no simple 2D matrix transpose in the code.
Instead, the equivalent operation is a transposed convolution / gradient with
respect to the input feature map.

In `engine/pc.py`, this is:

```python
bu_base_maps = conv2d_grad_wrt_x(error, W)
```

Conceptually:

```text
The child-layer error is spread back through the convolution kernel
to tell the parent feature map how to change.
```

For `layer_0`, the parent was upsampled from `14 x 14` to `28 x 28` before
prediction. So after projecting the error through the convolution, the code
resizes the update back down to the actual parent latent size:

```python
bu_base_maps = jax.image.resize(
    bu_base_maps,
    (batch_size, C_in, H_target, H_target),
    method='bilinear'
)
```

## 15. Convolution In Weight Updates

For fully connected layers:

```text
dW = z_parent.T @ error_deriv
```

For convolutional layers, the same idea becomes:

```text
dW = correlation between parent feature patches and child error maps
```

The code does this with `jax.lax.conv_general_dilated` inside
`compute_weight_grads`.

Conceptually, for each kernel position:

```text
gradient = sum over batch and spatial positions of:
  parent_patch_value * local_prediction_error
```

So the local Hebbian idea is still the same:

```text
parent activity times child prediction error
```

The difference is that convolution shares the same small kernel over all spatial
locations, so the gradient accumulates evidence from every place where that
kernel was used.

## 16. Important Implementation Details To Remember

1. `z_0` is the input image and is clamped.

2. `z_1`, `z_2`, etc. are latent states. They are initialized, then updated by
   inference. They are not computed once by a feedforward encoder.

3. Labels are not used in the main PC training path.

4. The top latent is updated only from the error below it.

5. Weight updates are local. `W_k` uses `z_{k+1}` and `eps_k`, not all errors
   in the network.

6. `mu_0` uses sigmoid because it predicts image pixels between 0 and 1.
   Hidden predictions use the configured activation, which defaults to `tanh`.

7. In the current code, the derivative used in `compute_latent_gradients` and
   `compute_weight_grads` is `d_actfx` for every layer. With the default config,
   this means `tanh'` is used even for the bottom layer path, even though
   `mu_0` itself is computed with sigmoid. Conceptually, the bottom path would
   usually use `sigmoid'` for a sigmoid output.

8. `model.forward` recomputes the reconstruction after the inference loop, but
   the weight update uses the `predictions` and `errors` variables from the last
   loop iteration. Those were computed just before the final latent update
   inside the loop.

## 17. Short Mental Model

For every batch:

```text
Clamp:
  z_0 = x

Initialize:
  z_1, z_2, ..., z_L = zeros/random/warm-start

Infer:
  Repeat:
    Higher states predict lower states.
    Errors are target minus prediction.
    Hidden states move to reduce neighboring errors.

Learn:
  Each weight learns from:
    parent latent activity * local child prediction error
```

The whole system is trying to find hidden causes `z_1...z_L` that explain the
input image, then update the generative weights so future explanations become
better.
