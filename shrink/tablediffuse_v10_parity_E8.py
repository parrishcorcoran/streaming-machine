"""TABLEDIFFUSE v10 -- the parity screen E8: the 8 low places of a group must spell one of the 16 words of the [8,4,4]
extended Hamming code (Construction A: the E8 crystal). For each code word the nearest numbers with those low places
are found; the code word with the least total leftover^2 (in steps) is kept. The low places then cost the 4-bit code
word instead of 8 bits: 0.5 bits a value saved (~2.4 MB), for grain 0.0833 -> 0.1434 step^2 (x1.72; measured before
judging). The cube pays x2 for the same 0.5 bits (half a place coarser): E8 is the densest packing in 8 lines, 0.65 dB
(0.108 bits a value) better than the cube at any exposure.
Judged on 5 realizations per c, against the cube's (see v9).
Predicted: E8 at c sits between the cube at c and at c+1, nearer c (x1.72 = 0.39 of a place):
   c=32 29-30 (31.5 MB), c=33 28-30 (~26.7), c=34 28-29 (21.97), c=35 26-28 (~17.3).
    python3 shrink/tablediffuse_v10_parity_E8.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run_lattice([dict(name=f"v10 parity E8, c={c}" + ("" if p is None else f" phase seed={p}"), c=c, kind="E8", phase_seed=p)
                for c in (32, 33, 34, 35) for p in (None, 1, 2, 3, 4)])
