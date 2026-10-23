"""DIFFUSE v9 -- the dark frame again, measured where the judged word lives. v8 found no gain; the look
(diffuse_look_standing.py) shows why it might have missed: position 0 (the first word) sees nearly the same light in
every text (its error is 95-97% standing) and is bright, so v8's per-neuron mean was pulled toward position 0's
offset, which is NOT the offset of positions 1-4. A neuron has one constant, so here it is measured on positions 1-4
only (128 windows x 4 = 512 positions: the mean is one shift, >> 9), where 33-58% of the plain error power stands.
Also the chained form: the developed machine runs its own light forward and each bank's constant takes up the
standing error carried to it from every bank before (error diffusion along the chain of plates, standing part only).
Prediction before running: if the standing error mattered, A=5 (plain 22) rises to 25+ and A=6 to 30. If words stay
at plain's level, the words are read by the moving part of the light alone."""
from diffuse_dark import *
import diffuse_dark as dd
dd.POS = 1
args = sys.argv[1:] or ["v12:6:0", "v12:5:0", "v12:6:1", "v12:5:1"]
WT = windows()
for a in args:
    rule, A, ch = a.split(":"); A = int(A); ch = bool(int(ch))
    run_dark(f"v9 plain + dark-frame constants on positions 1-4{' chained' if ch else ''}, {rule} A={A}", rule, A,
             lambda n, q, s: develop_plain(q, s), WT, chained=ch, note="constants from positions 1-4 (>> 9)")
