"""COMBINED v2b -- CONTROL for v2: is a words count the exposure, or the one grain pattern that fell?
v2 gave c=35: 30, c=36: 25, c=37: 28 with plates v12 A=8. c=37 is COARSER than c=36 on 57 lines (gain >= 2) and the
same on the other 711, yet it kept 3 more words. If the words were set by the exposure, coarser could not be better.
Test: the same exposures on grids shifted by a fraction of a step (a phase per neuron and per line from a fixed seed:
same grain power, another pattern). Prediction before running: c=34 and c=35 phases spread 28-30 (text 12, lead
0.0074, a coin each time; 16 and 7 sometimes); c=36 and c=37 phases overlap, both spread over ~25-29."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
for c in (34, 35, 36, 37):
    w = judge_phase(f"v2b CONTROL plates v12 A=8 + tables c={c}, shifted grids", cfg_all("v12", 8), c)
    print(f"c={c}: phases 1-4 words {w}", flush=True)
