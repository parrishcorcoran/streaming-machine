"""DIFFUSE v3 -- the carry runs along each LINE instead: the rounding error of weight (i, j) goes to weight (i, j+1),
the next neuron on the same line (natural neuron order), the way a fax/printer carries error along its scan line.
Now every neuron's own column keeps independent-ish errors t_ij = e_ij - e_i,(j-1): each pile's scatter is doubled
for ANY light (the two errors on a line are both lit by the same x_i), but the errors of neighbouring neurons cancel
in pairs: neuron j's pile error = sum_i e_ij x_i - sum_i e_i,(j-1) x_i. That pays only if what reads the plate's
neurons adds neighbouring neurons with like weights. Nothing in the machine does: a q.k head multiplies neurons in
pairs, the next plate's lines read neuron j with weights that are unrelated from j to j+1; the gauge removes only the
part common to all 768 stream lines.
Prediction before running: worse than plain at the same A, about like v1 (2x pile scatter): ~25 at A=8."""
from diffuse_lib import *
args = sys.argv[1:] or ["v12:8", "v12:7", "v12:6"]
for a in args:
    rule, A = a.split(":"); A = int(A)
    run(f"v3 carry across neurons on each line, {rule} A={A}", rule, A,
        lambda n, q, s: develop_carry(q, s, axis=1, kind=1, unit=False), note="first order, raw units, along lines")
