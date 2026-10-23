"""DIFFUSE v12 -- is v9's chained dark frame (24 at v12 A=5 against plain 22) real or the judge's scatter (+-2 words
between equally good developments, see round one's v18 by-chance seeds 26-28)? The same chained constants (positions
1-4) at other exposures and with a different calibration light. Every run is reported, none picked.
Prediction before running: if the carry of the standing error along the chain of plates is real, each run is >= plain
at the same exposure (v12 A=4: plain 23; v14 A=5: 26; v14 A=6: 29); with random-word light (a light that is not text)
the gain should shrink, because the standing part of text light and of random-word light differ."""
from diffuse_dark import *
import diffuse_dark as dd
dd.POS = 1
args = sys.argv[1:] or ["v12:4:text", "v14:5:text", "v14:6:text", "v12:5:rand"]
for a in args:
    rule, A, light = a.split(":"); A = int(A)
    wins = windows() if light == "text" else windows(seed=77)
    run_dark(f"v12 chained dark-frame constants (positions 1-4, {light} light), {rule} A={A}", rule, A,
             lambda n, q, s: develop_plain(q, s), wins, chained=True, note="v9 chained, robustness check")
