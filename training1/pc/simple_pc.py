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
H_LR = 0.8
W_LR = 0.02

train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)

W_x = torch.randn(HIDDEN, 784, device=DEVICE) / HIDDEN**0.5
b_x = torch.zeros(784, device=DEVICE)
W_y = torch.randn(10, HIDDEN, device=DEVICE) / 10**0.5
b_h = torch.zeros(HIDDEN, device=DEVICE)

for _ in range(EPOCHS):
    for images, labels in train_loader:
        x = images.view(images.size(0), -1).to(DEVICE)
        y = F.one_hot(labels.to(DEVICE), 10).float()

        h = y @ W_y + b_h

        for _ in range(INFER_STEPS):
            x_hat = h @ W_x + b_x
            h_hat = y @ W_y + b_h
            eps_x = x - x_hat
            eps_h = h - h_hat
            grad_h = -(eps_x @ W_x.t()) / 784 + eps_h / HIDDEN
            h = (h - H_LR * grad_h).clamp(-5, 5)

        x_hat = h @ W_x + b_x
        h_hat = y @ W_y + b_h
        eps_x = x - x_hat
        eps_h = h - h_hat

        W_x += W_LR * (h.t() @ eps_x) / x.size(0)
        b_x += W_LR * eps_x.mean(0)
        W_y += W_LR * (y.t() @ eps_h) / x.size(0)
        b_h += W_LR * eps_h.mean(0)



# practical flow of how pc algrithm works

# pc starts from input, label and wight intialized for the hidden layer and output layer
# so when we start we have 

# assume the intput vector is 3 by 5 and we have 3 class to pridict

# x1 = [1, 0, 1, 0, 0]   label = class 0
# x2 = [0, 1, 0, 1, 0]   label = class 1
# x3 = [0, 0, 1, 0, 1]   label = class 2

# and we have one-shot labels

# y1 = [1, 0, 0]
# y2 = [0, 1, 0]
# y3 = [0, 0, 1]

# so 

# X = [
#     [1, 0, 1, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 0, 1, 0, 1],
# ]

# Y = [
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
# ]


# Initial weights

# We use:

# W_x.shape == [3, 5] so our hidden layer will be 1 by 3
# cuz the hidden state pridict the input to match that if we decided our weight to be 3 by 5 out hidden state will be 1 by 3

# so 1 by 3 * 3 by 5 = 1 by 5 which is the image

# W_x =
# [
#   [ 0.10, -0.20,  0.00,  0.10,  0.20],
#   [-0.10,  0.10,  0.20,  0.00, -0.20],
#   [ 0.00,  0.20, -0.10,  0.10,  0.10],
# ]

# W_y.shape == [3, 3]

# W_y =
# [
#   [ 0.20,  0.00, -0.10],
#   [-0.10,  0.10,  0.20],
#   [ 0.00, -0.20,  0.10],
# ]


# H_LR = 0.5      # hidden-state inference learning rate
# W_LR = 0.1      # weight learning rate
# INFER_STEPS = 2
# image_dim = 5
# hidden_dim = 3


# so the hidden state dimention will be 1 by 3

# Image 1: class 0

# Image:

# x = [1, 0, 1, 0, 0]

# y = [1, 0, 0]

# h = y @ W_y

# h0 = [0.20, 0.00, -0.10]

# now we can start the inference loop, mind that we get the h from the output pridiction as intialization


# Inference step 0

# Predict image from hidden:

# x_hat = h @ W_x

# x_hat = [0.02, -0.06, 0.01, 0.01, 0.03]

# Predict hidden from label:

# h_hat = y @ W_y

# h_hat = [0.20, 0.00, -0.10]

# eps_x = x - x_hat
# eps_h = h - h_hat


# eps_x = [0.98, 0.06, 0.99, -0.01, -0.03]
# eps_h = [0.00, 0.00, 0.00]


# E_x = 0.19451
# E_h = 0.00000

# grad_h = [-0.0158, -0.0224, 0.0182]

# h = h - H_LR * grad_h

# h_new = [0.20, 0.00, -0.10] - 0.5 * [-0.0158, -0.0224, 0.0182]

# h_new = [0.2079, 0.0112, -0.1091]

# Inference step 1

# Now use:

# h = [0.2079, 0.0112, -0.1091]

# Predict image:

# x_hat = h @ W_x

# x_hat = [0.0197, -0.0623, 0.0132, 0.0099, 0.0284]


# h_hat = y @ W_y

# this is similar with the first inference iteration cuz we are using the same y and ouput wight
# this changes when the wight gets updated and/or the label changes

# till that the inside the the same inference loop h_hat will stay the same.
# but eps_h changes since h will be changed in every iteration due to the grad


# h_hat = [0.20, 0.00, -0.10]


# eps_x = x - x_hat
# eps_h = h - h_hat

# eps_x = [0.9803, 0.0623, 0.9869, -0.0099, -0.0284]
# eps_h = [0.0079, 0.0112, -0.0091]

# E_x = 0.19397
# E_h = 0.000045
# E   = 0.194016

# grad_h = [-0.0131, -0.0185, 0.0150]

# h_new = [0.2079, 0.0112, -0.1091] - 0.5 * [-0.0131, -0.0185, 0.0150]

# h_new = [0.2145, 0.0205, -0.1166]

# After 2 inference steps, the settled hidden state for image 1 is:

# h = [0.2145, 0.0205, -0.1166]

# Notice what happened:

# initial h:  [0.2000, 0.0000, -0.1000]
# settled h:  [0.2145, 0.0205, -0.1166]

# The hidden state moved slightly so that it could better explain the image.



# x_hat = h @ W_x + b_x
# h_hat = y @ W_y + b_h

# eps_x = x - x_hat
# eps_h = h - h_hat


# eps_x = [0.9806, 0.0642, 0.9842, -0.0098, -0.0271]

# eps_h = [0.0145, 0.0205, -0.0166]



# W_x += W_LR * h.T @ eps_x
# W_y += W_LR * y.T @ eps_h

# h.T @ eps_x

# has shape:

# [3, 1] @ [1, 5] = [3, 5]

# So each hidden neuron updates its connection to each image pixel.

# After image 1:

# W_x become from the following 

# W_x =
# [
#   [ 0.10, -0.20,  0.00,  0.10,  0.20],
#   [-0.10,  0.10,  0.20,  0.00, -0.20],
#   [ 0.00,  0.20, -0.10,  0.10,  0.10],
# ]


# to 

# W_x =
# [
#   [ 0.1210, -0.1986,  0.0211,  0.0998,  0.1994],
#   [-0.0980,  0.1001,  0.2020, -0.0000, -0.2001],
#   [-0.0114,  0.1993, -0.1115,  0.1001,  0.1003],
# ]


# After image 1:

# W_y become from the following



# W_y =
# [
#   [ 0.20,  0.00, -0.10],
#   [-0.10,  0.10,  0.20],
#   [ 0.00, -0.20,  0.10],
# ]

# to

# W_y =
# [
#   [ 0.2014,  0.0020, -0.1017],
#   [-0.1000,  0.1000,  0.2000],
#   [ 0.0000, -0.2000,  0.1000],
# ]



# Biases after image 1:

# b_x = [0.0981, 0.0064, 0.0984, -0.0010, -0.0027]
# b_h = [0.0014, 0.0020, -0.0017]


# for image 2, what changes is that the hidden state become new according to the W_y and y, then we repeat the above same process
# till all the image completed


    