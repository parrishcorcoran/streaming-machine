"""DIFFUSE v5 -- second-order noise shaping down each neuron's column: the next line gets -(2e - e_prev), so the total
weight error is t_i = e_i - 2 e_(i-1) + e_(i-2). The column's running sum AND its running first moment are held: a
light that is constant OR changes in a straight ramp from line to line sees no error at all. For light unrelated line
to line the pile scatter is 1 + 4 + 1 = 6 times plain (in power), against 2 for first order.
Prediction before running: worse than v4 (first order, step units) by roughly one more place of exposure: ~20-24 at
A=8 (plain A=6 is 29, A=5 is 22, so 6x the power is like plain A~5.4)."""
from diffuse_lib import *
args = sys.argv[1:] or ["v12:8:1", "v12:7:1", "v12:8:0"]
for a in args:
    rule, A, u = a.split(":"); A = int(A); u = bool(int(u))
    run(f"v5 second-order carry down lines ({'steps' if u else 'raw units'}), {rule} A={A}", rule, A,
        lambda n, q, s: develop_carry(q, s, axis=0, kind=2, unit=u), note="carry -(2e - e_prev)")
