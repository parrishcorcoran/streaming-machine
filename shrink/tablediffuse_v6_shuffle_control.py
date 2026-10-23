"""TABLEDIFFUSE v6 -- the shuffle control for v1. Take v1's developed table (carry to the next line, natural order) and
move each value's grain to a RANDOM OTHER WORD ON THE SAME LINE (one fixed-seed reshuffle of the 50,257 words per
line): every line keeps exactly its own grain values (the same two-step-wide distribution, the same doubled power), but
the cancellation along each word's lines is gone (a word's grain no longer adds to < 1 step over its lines).
If the cancellation did anything for the words, the control must lose words against v1; if only the grain's power
counts, the control equals v1 within luck. Not a stored form (the moved grain is off the grid): a judge-only control.
Also: v1 at c=37 and nearest at c=37 judged print by print (E developed with the top as stored; the top printed from
the developed E with the entry as stored), to see which print the carry breaks.
Predicted: control = v1 within luck at every c (c=32 28-30, c=34 25-29, c=37 14-22).
Prints at c=37: nearest entry-only 30, top-only 29 (round one); v1 entry-only 25-29, top-only 22-28.
    python3 shrink/tablediffuse_v6_shuffle_control.py
"""
import sys, time, numpy as np; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
tb = td.machine(); E = tb._WTE0
for c in (32, 34, 37):
    t0 = time.time(); D, e = td.develop(E, c, "lines")
    if c == 37:
        row = dict(name="v6 prints of v1 c=37 (carry to next line)", spec=dict(c=c, how="lines")); row.update(td.judge(D, parts=True))
        row["secs"] = round(time.time() - t0, 1); td.record(row)
    rng = np.random.default_rng(600 + c)
    for i in range(768):                                # D <- E + (D - E) of a random other word, line by line
        g = D[:, i].astype(np.int64) - E[:, i]; D[:, i] = (E[:, i] + g[rng.permutation(E.shape[0])]).astype(np.int32)
    row = dict(name=f"v6 CONTROL v1's grain on random words of the same line, c={c}", spec=dict(c=c, how="lines", shuffle=600 + c))
    row.update(td.judge(D)); row["grain_steps2"] = round(td.grain_steps2(D, E, e), 4)
    row["secs"] = round(time.time() - t0, 1); td.record(row); del D
t0 = time.time(); D, e = td.develop(E, 37, "nearest")
row = dict(name="v6 prints of nearest c=37", spec=dict(c=37, how="nearest")); row.update(td.judge(D, parts=True))
row["secs"] = round(time.time() - t0, 1); td.record(row)
