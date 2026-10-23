"""PLATES v12 -- the meters read to half a place. v7 rounds each neuron's and each line's brightness to a whole place
before choosing the step, so two weights of the same relative size can get steps a factor 2 apart. Here each meter
is read to half a place (the place of its top 1 and whether the next place down is lit), the two are added, and only
then the step is taken as the whole place under the sum. The step is still a power of two (a shift).
A = 2a matches v7 a; an odd A sits between two whole-place exposures.
Prediction before running: A=8 is v7 a=4's size with the same or one better words; A=7 lands near 51-52 MB at ~28-29."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
As = [int(x) for x in sys.argv[1:]] or [8, 7, 6]
for A in As:
    def dev(n, q, e, A=A):
        s = half_places(q, A); return cut(q, s), s
    run_version(f"v12 half-place meters A={A}", dev, side_per_col=2, extra_side=extra,
                note=f"neuron + line meters in half places, step = whole place under (sum - {A})/2, nearest")
