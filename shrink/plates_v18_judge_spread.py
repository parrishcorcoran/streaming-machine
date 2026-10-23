"""PLATES v18 -- control: how far apart can two equally good developments land on the 30 words? Rounding by chance
at half a place finer has exactly the scatter of nearest rounding (uniform remainder: by chance s^2/6 at step s, nearest
s^2/12; half a place finer = step / root 2 -> (s^2/2)/6 = s^2/12). So v12's exposure at A=7 by chance, three seeds,
carries the same scatter power as v12 A=6 nearest (29/30) in three different random patterns.
Prediction before running: the three land within a few words of each other (27-30); that spread is the resolution of
the 30-word judge, and differences between versions smaller than it are not findings."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
A = int(sys.argv[1]) if len(sys.argv) > 1 else 7
for seed in (11, 12, 13):
    rng = np.random.default_rng(seed)
    def dev(n, q, e, rng=rng):
        s = half_places(q, A); return cut(q, s, chance=True, rng=rng), s
    run_version(f"v18 control: v12 A={A} by chance, seed {seed}", dev, side_per_col=2, extra_side=extra,
                note=f"same scatter power as v12 A={A-1} nearest; seed {seed}")
