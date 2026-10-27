import numpy as np
from hopfield import Hopfield

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

class ModernHopfield:

    def __init__(self, beta=1.0):
        self.beta = beta
        self.P = None

    def store(self, patterns):
        self.P = np.array(patterns, dtype=float)

    def energy(self, x):
        scores = self.beta * self.P @ x
        m = scores.max()

        lse = m + np.log(np.exp(scores - m).sum())
        return -lse / self.beta + 0.5 * x @ x

    def recall(self, x, steps=1):
        x = x.astype(float)
        for _ in range(steps):
            w = softmax(self.beta * self.P @ x)
            x = w @ self.P

        return x


# --- Tiny example from above ---
net = ModernHopfield(beta=1.0)
net.store([[1, 1, -1, -1], [1, -1, 1, -1]])
x = np.array([-1, 1, -1, -1])
print(net.recall(x))                  # ~ [ 1.  0.964 -0.964 -1. ]
print(net.energy(x), net.energy(net.recall(x)))



# --- Capacity test: classical vs modern ---
rng = np.random.default_rng(0)
n, P_count = 100, 50                  # 50 patterns, far beyond classical limit (~14)
patterns = rng.choice([-1, 1], size=(P_count, n))

noisy = patterns[0].copy()
noisy[rng.choice(n, size=20, replace=False)] *= -1   # flip 20% of the bits

classical = Hopfield(n)               # your class from before
classical.train(patterns)
out_c = classical.recall(noisy)
print("classical correct:", np.array_equal(out_c, patterns[0]))

modern = ModernHopfield(beta=1.0)
modern.store(patterns)
out_m = np.sign(modern.recall(noisy))
print("modern correct:   ", np.array_equal(out_m, patterns[0]))


# i might need explanation on softmax and its probability defination

