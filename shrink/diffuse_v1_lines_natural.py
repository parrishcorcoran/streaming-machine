"""DIFFUSE v1 -- first-order error carry down each neuron's column, lines in their natural order (0, 1, 2, ...).
Each weight's rounding error e (in the plate's own units) is subtracted from the next line's weight of the same
neuron before that one is developed, so the developed column's running sum never strays more than half a step from
the true running sum: the neuron's total over its lines is exact to half a step (plain rounding: ~ root(lines)/3.5
steps). The total weight error becomes t_i = e_i - e_(i-1).
The pile for one token's light x is sum_i t_i x_i = sum_i e_i (x_i - x_(i+1)).  So the carry wins only where the light
on consecutive lines is alike: plain scatter ~ sum x^2 = lines x (mean^2 + spread^2); carried ~ lines x 2 spread^2
(for lines whose light is unrelated from one to the next). The carry wins iff mean^2 > spread^2, i.e. only for a light
that is mostly common to all lines.
Prediction before running: the stream's light into qkv/fc is far from common-mode (its size is set by a few bright
lines), so the carry doubles the pile scatter there; at the same A the words fall roughly to one A lower (A=8 -> ~28)
and the file grows by ~1% (the stored numbers wander by up to a step more)."""
from diffuse_lib import *
args = sys.argv[1:] or ["v12:8", "v12:7", "v12:6"]
for a in args:
    rule, A = a.split(":"); A = int(A)
    run(f"v1 carry down lines, natural order, {rule} A={A}", rule, A,
        lambda n, q, s: develop_carry(q, s, axis=0, kind=1, unit=False), note="first order, raw units")
