"""STEP 2, THE EXACT CHECK. 3 texts (refs 0, 1, 2; first 5 tokens of ids+ref) x 5 words (the answer = the machine's
own word at the last position, its 2nd and 3rd words, and 2 fixed other words).

Forward: the machine untouched (base2.py's own lines) gives every word's whole-number score. The recorded run must give
the same scores; every rounding it made is written down exactly and checked to be at most half a unit (nothing wrapped).
Reverse: word w's row at the top, sent back down every step in Python whole numbers with NO rounding (every dimming a
whole number over a power of two, one power of two per array). What reaches the bottom is R_w (768 numbers at every
position); c_w gathers every bias and every rounding met with the reverse light at its own step.
Check: (embedding numbers met with R_w) + c_w == the machine's score for w, exactly, as rationals.

PREDICTION (written before running): holds exactly, by construction, for all 15.
    python3 reverse/exact_check.py
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")                                # numpy's float products on this Mac raise spurious flags; checked exact below
from machine import *

t0 = time.time(); allok = True
for ti in (0, 1, 2):
    ids = TEXTS[ti]
    S = machine_scores(ids); ans = forward(ids)
    rec = record(ids)
    same = bool((rec["top"]["scores"] == S).all()) and race(S) == ans
    res = residues(rec)
    over = 0
    for d in res[0]:
        for k, (n, e) in d.items():
            e = np.broadcast_to(e, n.shape)
            lim = np.vectorize(lambda z: (1 << int(-z - 1)) if z < 0 else 0, otypes=[object])(e)
            over += int((~((np.abs(n) <= lim).astype(bool))).sum())
    order = np.argsort(-S)
    words = [int(order[0]), int(order[1]), int(order[2]), 1000 + 37 * ti, 30000 + 113 * ti]
    print(f"text {ti}: {''.join(word(i) for i in ids)!r}  answer {word(ans)!r}  "
          f"recorded run = untouched machine: {same}; roundings over half a unit: {over}", flush=True)
    G, c = reverse_exact(rec, words, res)
    m = meet_exact(rec["R0"], G)
    print(f"   R_w: whole numbers up to {G.bits()} places over 2^{-G.e}")
    for w, a, b in zip(words, m, c):
        tot = a + b; ok = (tot == S[w]) and tot.denominator == 1; allok &= ok
        print(f"   {word(w)!r:14s} machine {int(S[w]):>10d} | E.R_w = {float(a):>16.6f} (exact: {a.numerator.bit_length()}-place"
              f" whole number / 2^{a.denominator.bit_length() - 1}) + c_w = {float(b):>16.6f} -> {tot}  "
              f"{'EXACT' if ok else 'DIFFERENT'}", flush=True)
print(f"\nall 15 exact: {allok}   ({time.time() - t0:.0f}s)")
