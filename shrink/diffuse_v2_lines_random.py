"""DIFFUSE v2 -- first-order error carry down each neuron's column, lines in a RANDOM order (one fixed permutation
of the lines per plate, seed = 1000 + plate index; shared by all the plate's neurons). No similarity used.
v1 lost 5-7 words at the same size. If v1 lost because the natural order puts unrelated lights side by side, a random
order is no better (unrelated either way). If v1 lost because of something the natural order does that a random one
does not (e.g. a bright line always receiving the carry of the same coarse neighbour), the words change.
Prediction before running: same as v1 within the judge's noise (+-2 words): 23-26 at A=8."""
from diffuse_lib import *
args = sys.argv[1:] or ["v12:8", "v12:7", "v12:6"]
PERM = {n: np.random.default_rng(1000 + k).permutation(plate(n).shape[0]) for k, n in enumerate(NAMES)}
for a in args:
    rule, A = a.split(":"); A = int(A)
    run(f"v2 carry down lines, random order, {rule} A={A}", rule, A,
        lambda n, q, s: develop_carry(q, s, axis=0, order=PERM[n], kind=1, unit=False), note="first order, raw units, seed 1000+k")
