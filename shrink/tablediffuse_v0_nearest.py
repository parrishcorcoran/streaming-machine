"""TABLEDIFFUSE v0 -- the controls, before any carry.
(a) round one's development (lens-matched exposure, cap 12, nearest; T printed from E) rebuilt with this family's code
    (tablediffuse_ed.c mode 0) on a denser ladder c = 32..37: must give back round one's numbers where they exist
    (c=32 30/30 34.641 MB, c=34 29/30 24.964, c=37 28/30 14.998; v8b c=33 28, c=35 28).
(b) the judge's luck: the same grain power laid at another screen phase (every line's grid shifted by a fixed-seed
    fraction of its step, nearest on that grid). Same exposure, same bits (+1.5 KB of phases), another realization of
    the grain. The spread of words over phases is how far a single count can move by luck alone -- every carry version
    has to be read against it.
Predicted before running:
    (a) c=32 30, 33 28, 34 29, 35 28, 36 27-29, 37 28;   sizes as round one to the byte.
    (b) c=32: 29-30 at every phase; c=34: 27-30; c=37: 26-29.
    python3 shrink/tablediffuse_v0_nearest.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run([dict(name=f"v0 nearest c={c}", c=c, how="nearest") for c in (32, 33, 34, 35, 36, 37)])
td.run_phase((32, 34, 37), (1, 2, 3, 4))
