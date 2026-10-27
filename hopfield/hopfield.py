import numpy as np


class Hopfield:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))

    def train(self, patterns):
        for p in patterns:
            self.W += np.outer(p, p)       # Hebbian rule
        np.fill_diagonal(self.W, 0)         # no self-connections
        self.W /= self.n

    def energy(self, x):
        return -0.5 * x @ self.W @ x

    def recall(self, x, sweeps=10):
        x = x.copy()
        for _ in range(sweeps):
            for i in np.random.permutation(self.n):   # asynchronous updates
                x[i] = 1 if self.W[i] @ x >= 0 else -1
        return x



# demo

rng = np.random.default_rng(0)
n = 100
patterns = rng.choice([-1, 1], size=(3, n)) 

# so why are the numbers -1, 0 and 1

net = Hopfield(n)
net.train(patterns) # what exactlly are we training here

noisy = patterns[0].copy()
flip = rng.choice(n, size=15, replace=False)   # corrupt 15% of the bits
noisy[flip] *= -1

out = net.recall(noisy)
print("energy before:", net.energy(noisy))
print("energy after: ", net.energy(out))
print("recovered correctly:", np.array_equal(out, patterns[0]))

# pattern recognition, complition and noise reduction
