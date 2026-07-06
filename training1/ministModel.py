import torch.nn as nn

class LinearClassifier(nn.Module):
    """
    The simplest possible classifier.

    It flattens a 28*28 image into a vector of length 784
    and applies a single linear transformation:
        logits = W x + b
    """

    def __init__(self):
        super().__init__()

        # A fully connected layer:
        # input: 784 pixels
        # output: 10 digit scores
        self.linear = nn.Linear(784, 10)

    def forward(self, x):

        # x has shape:
        # (batch_size, 1, 28, 28)

        # Flatten each image into a vector of length 784.
        x = x.view(x.size(0), -1)

        # Compute logits = W x + b
        logits = self.linear(x)

        return logits
    
model = LinearClassifier()

print(model)



