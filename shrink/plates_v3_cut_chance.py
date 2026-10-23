"""PLATES v3 -- develop at a fixed place, by chance. As v2 (each neuron keeps its top b places), but each weight
rounds up with chance equal to its remainder (add a random whole number below 2^s, shift; fixed seed). The answer is
right on average for every single weight; the price is twice the scatter of nearest rounding (a uniform remainder
gives s^2/12 nearest, s^2/6 by chance).
Prediction before running: worse than v2 at the same b -- by about half a place -- and no cheaper to store."""
from plates_lib import *
import sys
bs = [int(x) for x in sys.argv[1:]] or [9, 8, 7]
for b in bs:
    s = 15 - b; rng = np.random.default_rng(1000 + b)
    run_version(f"v3 cut by chance b={b}", lambda n, q, e, s=s, rng=rng: (cut(q, s, chance=True, rng=rng), s), side_per_col=1,
                note=f"keep top {b} places of each neuron (cut below place {s}), rounding by chance, seed {1000+b}")
