"""DIFFUSE v4 -- error carry WITH round one's meters: the carry is measured in steps. v1/v2 carried the error in the
plate's own units, so a coarse line's error (up to half ITS step) landed whole on the next line even when the meter
had given that line a step 4-8 times finer (a small plate row: the gauge's gain is small there and the light large).
The meter's protection of those lines was undone, which is why v1/v2 lost more than the factor 2 of a first-order
carry (A=8 fell to 21-25, like plain A=5). Here the error leaves as a fraction of its own step (e << (16 - s), a
shift) and arrives as the same fraction of the next weight's step (>> (16 - s'), rounded): every weight's total error
stays within one of its own steps, so each line keeps its exposure.
What is left is the first-order carry itself: pile scatter sum_i e_i (y_i - y_(i+1)) in step units, 2x plain for
light unrelated line to line.
Prediction before running: better than v1/v2 by 2-4 words, still 1-2 below plain at the same A (A=8 ~28, A=6 ~27);
v14 A=6 ~27 (plain 29)."""
from diffuse_lib import *
args = sys.argv[1:] or ["v12:8", "v12:7", "v12:6", "v14:6"]
for a in args:
    rule, A = a.split(":"); A = int(A)
    run(f"v4 carry down lines in steps (meters kept), {rule} A={A}", rule, A,
        lambda n, q, s: develop_carry(q, s, axis=0, kind=1, unit=True), note="first order, carry in fractions of a step")
