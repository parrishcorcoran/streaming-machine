"""PLATES v2 -- develop at a fixed place, nearest. Every neuron keeps its top b places (its power of two is set by its
brightest weight, which sits at place 14), everything below place 15-b is rounded away to the nearest (add half,
shift). Stored: the developed whole numbers (photon coder), each neuron's power of two, one b per plate.
Judged by words(plates=all 48) out of 30.
Prediction before running: stored bits ~ b - 1.5 per weight (a bell-shaped spread whose brightest is ~6x its typical
size); 30/30 holds down to b = 7 (~5.5 bits, ~58 MB), 28/30 near b = 5, broken by b = 3."""
from plates_lib import *
import sys
bs = [int(x) for x in sys.argv[1:]] or [9, 8, 7, 6, 5, 4]
for b in bs:
    s = 15 - b
    run_version(f"v2 cut nearest b={b}", lambda n, q, e, s=s: (cut(q, s), s), side_per_col=1,
                note=f"keep top {b} places of each neuron (cut below place {s}), nearest")
