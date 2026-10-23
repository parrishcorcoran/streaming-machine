"""TABLEDIFFUSE v11 -- the low end of the frontier, read with the luck control. Round one's c=37 (15.00 MB, 28/30) was
one realization; v0's screen phases put the cube at c=37 at {28,27,24,23,27} (mean 25.8). Here: the cube at c=36
(phases 1-4; round one's grid gave 25), and the two parity screens at c=36 and c=37, 5 realizations each.
At c=36-37 the entry's cap binds (most lines at 2^12), so the E8 grain (x1.72) lands on the entry's coarsest lines too.
Predicted: cube c=36 23-28; D8 c=36 23-28 (13.9 MB), c=37 22-28 (14.0); E8 c=36 20-26 (~12.9), c=37 18-25 (12.5).
    python3 shrink/tablediffuse_v11_frontier_low.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run_phase((36,), (1, 2, 3, 4))
td.run_lattice([dict(name=f"v11 parity {k}, c={c}" + ("" if p is None else f" phase seed={p}"), c=c, kind=k, phase_seed=p)
                for k in ("D8", "E8") for c in (36, 37) for p in (None, 1, 2, 3, 4)])
