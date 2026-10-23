"""DIFFUSE v8 -- the dark frame: carry each neuron's STANDING-light error into its constant (see diffuse_dark.py).
Why, from v1-v6: a neuron's pile error for one word is sum_i t_i x_i. Split each word's light into the part that is
the same for every word (standing, x0) and the part that moves (y): error = sum t x0 + sum t y. A plain rounding
error t is unrelated from line to line, so it is read by the standing light in proportion to the standing light's
share of the power -- NOT 1/768 of it. If most of the light stands still (the next word only barely touches
everything), most of each neuron's error is one fixed offset, the same on every word: a dark frame. Carried into the
neuron's constant it is gone, for one whole number per neuron (82,944 numbers, about 0.1-0.2 MB coded).
Developing is round one's plain nearest (v12 meters); the plates in the file are identical to v0's; only the
constants are added.
Prediction before running: if the standing light holds most of the power, A=6 and A=5 gain several words (29 -> 30,
22 -> 26+); if the light mostly moves, nothing changes. Control: the same constants measured with random word ids
(a light that is not text) -- if the gain is the standing light it should mostly survive (the standing part is the
machine's, not the text's)."""
from diffuse_dark import *
args = sys.argv[1:] or ["v12:7", "v12:6", "v12:5", "v12:6:rand", "v12:5:rand"]
WT = windows()
for a in args:
    parts = a.split(":"); rule, A = parts[0], int(parts[1]); rand = len(parts) > 2
    wins = windows(seed=77) if rand else WT
    run_dark(f"v8 plain + dark-frame constants{' (random-word light)' if rand else ''}, {rule} A={A}", rule, A,
             lambda n, q, s: develop_plain(q, s), wins, note="constants = minus mean standing-light error per neuron")
