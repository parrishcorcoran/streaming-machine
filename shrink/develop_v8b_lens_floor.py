"""DEVELOP family, version 8b: why the lens exposure of version 8 fell to 18/30 at c = 33 and 35, and the repair.
Version 8 measured: T printed from E with E exposed for the lens: c=31 29, c=33 18, c=35 18 -- far below T's own word cut
at the same top step (v3: b=10 28, b=8 28), while E cut per word and printed gave b=12 30, b=10 28, b=8 23.
Reading of the negative: one negative serves two prints through two different lenses. The top reads line i through
gain g_i; the ENTRY reads every line with gain 1. Exposure c - place(g_i) puts the 8 lines with the faintest gains
(gain 2^-8..2^-4) at steps of 2^13..2^17 -- they go fully dark, even their values near 2^17. The top does not miss them;
the entry does (E alone tolerated a word step of 2^12 -- v3 b=4, 30/30 -- but never a dark line).
Repair (a fixed rule): no line exposed coarser than 2^12, the step E alone was measured to tolerate.
Predicted before running: c=33 cap 12 -> 26-29 (back to what the top allows, like T's own b=10: 28); c=35 cap 12 ->
24-28. If these stay near 18, the reading above is wrong.
    python3 develop_v8b_lens_floor.py words|store
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from develop_lib import run
SPECS = [dict(name=f"v8b E lens exposure c={c} cap 12, T printed", E=dict(how="lines", c=c, cap=12), T=dict(how="print"))
         for c in (31, 33, 35)]
if __name__ == "__main__": run(SPECS, sys.argv[1])
