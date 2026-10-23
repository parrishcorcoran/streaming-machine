"""The whole developed machine at once: developed entry table E (exposed for the lens, cap 12) with the top table T
printed from it, AND the developed plates read back from their stored files. Words out of 30; total stored size."""
import os, sys, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine"); sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
os.chdir("/Users/abundancemachine/streaming-machine")
src = open("shrink/plates_best_store.py").read(); exec(src[:src.index("for label, rule, A, k in")])
import develop_lib as dl
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad/"
orig = {n: _tb.W[n] for n in NAMES}
E = _tb._WTE0
TABLE_MB = {32: 34.641, 34: 24.964, 37: 14.998}
PLATE_MB = {"plates_v12_A8.bin": 56.743, "plates_v14_A6.bin": 45.860}
print("unchanged:", _tb.words(), flush=True)
for pf in ("plates_v12_A8.bin", "plates_v14_A6.bin"):
    read_into_machine(SP + pf)
    print(f"  {pf} alone: {_tb.words()}", flush=True)
    for c in (32, 34):
        D, e = dl.develop_lines(E, c, 12); Tp = dl.print_top(D)
        w = _tb.words(entry=D, top=Tp); del D, Tp
        print(f"  {pf} + tables c={c}: right next word {w} / 30   stored {PLATE_MB[pf] + TABLE_MB[c]:.1f} MB (plates {PLATE_MB[pf]} + tables {TABLE_MB[c]})", flush=True)
    for n in NAMES: _tb.W[n] = orig[n]
