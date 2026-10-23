"""PLATES v9 -- develop, then give every neuron back its total. qkv and fc carry the gauge's DC block folded in
(each neuron's weights sum to zero over the lines, so a light common to every line passes nothing); proj and mproj
take light that is all one sign on many lines (after the bend). Rounding each weight breaks the totals: the sum of
the rounding errors grows like the root of the line count. Here, after v7's development, the weights whose rounding
was nearest a tie are flipped one step (largest-remainder rule: cheapest extra scatter first) until each neuron's
total is back as close as the steps allow. Nothing extra is stored.
Prediction before running: if light common to all lines is a real part of the arriving light, words at a=3 rise
above v7's 27 for the same size; if not, nothing changes (the flips cost almost no scatter either way)."""
from plates_lib import *
import sys
extra = sum(plate(n).shape[0] for n in NAMES)
As = [int(x) for x in sys.argv[1:]] or [3, 4]
for a in As:
    def dev(n, q, e, a=a):
        s = v7_places(q, a); v = cut(q, s); v = keep_sums(q, v, s)
        return v, s
    run_version(f"v9 v7 a={a} + neuron totals kept", dev, side_per_col=2, extra_side=extra,
                note=f"v7 a={a}, then largest-remainder flips so each neuron's sum over lines is restored")
