import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for MNIST.

    Architecture:
        784 → 128 → 10

    The hidden layer allows the model to learn intermediate features
    instead of directly mapping pixels to digit classes.
    """

    def __init__(self):
        super().__init__()

        # First linear layer:
        # 784 input pixels → 128 hidden features

        # maybe math could help me here??
        # how exactlly is this 128 features learned in a way that they start to learn certain features
        # and the composition of multiple linear classifiers(without activation function) will be just another linear classifier
        # can we prove that??

        # One single neuron can only calculate one specific linear combination of the data. 
        # By having 64 neurons, the layer can look at the exact same input data from 64 different geometric perspectives simultaneously

        # by having multiple neorons(for below case 128) for a single data point(784 pixle is a single image) we help all those neurons
        # to learn different features for a single image

        # and this need to be studied mathimatically(i mean it is stillvegue for me how specific neuron like #42 neuron learn curve feature of specifc image 
        # and based on this neuron on and off status the final activation function pridiction will be affected, like how exactlly mathimatically??)
        # or in another question what mathis we are doing to make specific neuron learn specific feature of the image

        # like do you get my point, let us say i have x = [2.1 3.4 6.3 8.4] pixle value of image
        # for the same image i have different parameter/weights with similar vector size like

        #  w1 [1.3 4.3 5.4 2.1]
        #  w2 [1.5 1.3 3.4 1.1]
        #  w3 [6.3 6.3 6.4 9.1]
        #  w4 [8.3 7.3 7.4 3.1]

        # so what we are saying is thid different parameters is learning about different features of a single image
        # and for my below case my hidden layer is 128 neorns or w1 to w128 for a singel image learning different features of a single image

        # so it is still calssifing but features instead of digits

        # and also we are adding activation to remove the linear compsitionfeature of the output of the neurons
        # so neurons that give negative value will be turned into zero(0)


        self.fc1 = nn.Linear(784, 128)

        # Activation function (explained below)
        self.relu = nn.ReLU() # ReLU(x)=max(0,x)

        # Second linear layer:
        # 128 hidden features → 10 output logits
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        # Flatten images from (batch, 1, 28, 28)
        # to (batch, 784)
        x = x.view(x.size(0), -1)

        # First linear transformation
        x = self.fc1(x)

        # Apply the non-linear activation
        x = self.relu(x)

        # Produce the final logits
        logits = self.fc2(x)

        return logits
    

# Understanding Tensor Shapes

# One of the biggest sources of confusion is tensor dimensions.

# Let's follow them carefully for a batch of 64 images.

# Stage	        Shape	        Meaning

# Input images	(64, 1, 28, 28)	64 grayscale images
# Flatten	    (64, 784)	    Each image becomes a vector
# First Linear	(64, 128)	    128 hidden features per image
# ReLU	        (64, 128)	    Same shape, negative values clipped
# Second Linear	(64, 10)	    10 logits per image
# Labels	    (64,)           One digit label per image


# so single pass computes 

# 1  First Linear Layer (fc1: 784 → 128)
#  Weights: 784 * 128 = 100,352
#  Biases: One bias per hidden neuron = 128
#  Layer Total: 100,352 + 128 = 100,480

# 2. Second Linear Layer (fc2: 128 → 10)

# Weights: 128 * 10 = 1,280
# Biases: One bias per output digit = 10
# Layer Total: 1,280 + 10 = 1,290

# Grand Total
#     Total Parameters = 100,480 + 1,290 = 101,770


    
