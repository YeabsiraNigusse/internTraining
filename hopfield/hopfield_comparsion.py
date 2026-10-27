import numpy as np
from hopfield import Hopfield
from modern_hopfield import ModernHopfield
# assumes your Hopfield class and the ModernHopfield class are already defined above

rng = np.random.default_rng(0)
n = 100
flip = 20          # corrupt 20% of the bits
trials = 20        # repeat several times per setting for a stable average

print("patterns | classical | modern")
for P_count in [5, 10, 14, 20, 50, 100, 200, 500, 1000, 10000, 100000, 1000000]:
    ok_c = ok_m = 0
    for _ in range(trials):
        patterns = rng.choice([-1, 1], size=(P_count, n))

        noisy = patterns[0].copy()
        noisy[rng.choice(n, size=flip, replace=False)] *= -1

        c = Hopfield(n)
        c.train(patterns)
        ok_c += np.array_equal(c.recall(noisy), patterns[0])

        m = ModernHopfield(beta=4.0)
        m.store(patterns)
        ok_m += np.array_equal(np.sign(m.recall(noisy)), patterns[0])

    print(f"{P_count:8d} | {ok_c/trials:9.2f} | {ok_m/trials:6.2f}")

