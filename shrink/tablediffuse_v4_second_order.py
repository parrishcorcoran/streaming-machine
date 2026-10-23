"""TABLEDIFFUSE v4 -- second-order shaping along a word's lines: the next value is given 2u1 - u2 (the last two
leftovers), so the grain is shaped by (1 - z^-1)^2: the pile over the word's lines AND its first moment cancel.
The relation: the shaped grain power is (1 + 4 + 1) = 6 times a plain rounding's (1/12 -> 6/12 step^2), i.e.
log2(6)/2 = 1.3 places coarser for every reader that does not average neighbouring lines -- and none does.
Predicted: c=32 26-29, c=34 24-28, c=37 10-25. Stored +1-4% (the stored numbers wander up to 2 steps).
    python3 shrink/tablediffuse_v4_second_order.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run([dict(name=f"v4 second order along lines, c={c}", c=c, how="lines2") for c in (32, 34, 37)])
