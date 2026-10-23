"""TABLEDIFFUSE v9 -- the parity screen D8: halftoning done the other way round.
Error diffusion pushed every value's leftover onto a neighbour -- it ADDED grain (x1.33 to x6) for a cancellation no
reader sees. A press can instead nudge only the cheapest dot: deal the 768 lines into 96 groups of 8 (fixed-seed
shuffle, the same for every word), round all to nearest, and when a group's 8 stored whole numbers add to an odd number,
move the ONE value whose leftover is closest to half a step to its other neighbour. Every group then adds to an even
number, so the 8th number's low place is known from the other 7: 1 bit saved per 8 values (0.125 bits a value,
~0.6 MB), for 1 - 2|u_max| step^2 extra on one value in half the groups: grain 0.0833 -> 0.0901 step^2 (x1.083,
0.06 of a place; measured before judging). A coarser cube would pay x1.19 for the same 0.125 bits.
Coded with the counting coder (v8); the halved 8th number has its own context (768 + line).
Judged on the plain screen and four screen phases (5 realizations), against the cube's (v0/v8):
   cube c=32 {30,30,29,29,30}  c=33 {28,29,29,28,29}  c=34 {29,29,29,30,29}  c=35 {28,27,27,30,28}
Predicted: each D8 realization within +-1 of the cube's at the same c (no common sign); bytes c=32 33.35, c=33 ~28.5,
c=34 23.76, c=35 ~19.0 MB (+1.5 KB with a phase).
    python3 shrink/tablediffuse_v9_parity_D8.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run_lattice([dict(name=f"v9 parity D8, c={c}" + ("" if p is None else f" phase seed={p}"), c=c, kind="D8", phase_seed=p)
                for c in (32, 33, 34, 35) for p in (None, 1, 2, 3, 4)])
